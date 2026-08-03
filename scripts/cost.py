"""What recent calls cost, from a terminal.

    make cost                     # last 24h
    make cost ARGS="--hours 168"
    make cost ARGS="<call_id>"    # one call, priced line by line

The dashboard at /platform/spend on the hub is the same data with charts. This
exists because the first question after a surprising invoice is usually asked
over SSH, and because it needs no hub to answer.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

# Run as a file rather than a module (`make cost`), so the repo root is not on
# the path yet. `inspect_lead.py` sidesteps this by talking to psycopg directly;
# this one reuses `costing`, because the pricing arithmetic must have exactly
# one implementation and re-deriving it in a script is how the CLI and the
# dashboard come to disagree.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from backend import costing, db  # noqa: E402
from backend.config import settings  # noqa: E402
from backend.logger import setup_logging  # noqa: E402


def _money(usd) -> str:
    if usd is None:
        return "     —    "
    return f"${usd:>8.4f}"


def _rupees(usd) -> str:
    return "—" if usd is None else f"₹{usd * settings.inr_per_usd:,.2f}"


def overview(hours: int) -> None:
    economics = costing.unit_economics(hours)
    if not economics:
        print("No data. Is DATABASE_URL set and the schema migrated?")
        return

    print(f"\n  Last {hours}h — {economics['calls']} call(s)\n")
    print(f"    total spend             {_money(economics['cost_usd'])}"
          f"   {_rupees(economics['cost_usd'])}")
    print(f"    per call                {_money(economics['cost_per_call_usd'])}"
          f"   {_rupees(economics['cost_per_call_usd'])}")
    # The one the business runs on: a call that captured nothing still cost
    # money, so this is larger than the per-call average by the funnel's loss.
    print(f"    per DELIVERED lead      {_money(economics['cost_per_delivered_usd'])}"
          f"   {_rupees(economics['cost_per_delivered_usd'])}")
    print(f"\n    captured {economics['captured']}/{economics['calls']}"
          f" ({economics['capture_rate'] or 0}%)"
          f"   delivered {economics['delivered']}"
          f" ({economics['delivery_rate'] or 0}% of captured)"
          f"   flagged {economics['flagged']}")

    lag = costing.latency(hours)
    gap, ttft = lag["turn_gap"], lag["ttft"]
    print(f"\n    turn gap  p50 {gap['p50']}s  p95 {gap['p95']}s"
          f"      ttft  p50 {ttft['p50']}s  p95 {ttft['p95']}s")

    print("\n  Where the money goes")
    for row in costing.by_provider(hours):
        failed = f"   {row['failed']} failed" if row["failed"] else ""
        print(f"    {row['provider']:<14}{_money(row['cost_usd'])}"
              f"   {row['calls']} call(s){failed}")

    print("\n  Cost of an average call, by model")
    for row in costing.by_model(hours):
        print(f"    {row['model']:<26}{_money(row['cost_per_call_usd'])}"
              f"   over {row['calls']} call(s)")

    _coverage(costing.coverage(hours))

    print("\n  Most expensive calls")
    for row in costing.expensive_calls(hours, limit=8):
        secs = f"{row['duration']:.0f}s" if row["duration"] else "—"
        print(f"    {row['call_id'][:8]}  {_money(row['cost_usd'])}  {secs:>6}"
              f"  {row['turns'] or 0:>3} turns  "
              f"{(row['service'] or '—')[:18]:<18} {row['whatsapp_status']}")
    print()


def _coverage(cover: dict) -> None:
    """What the numbers above do not know. Printed even when clean, because
    "83% confident" is a different reading of the same total than "100%"."""
    print(f"\n  Coverage — {cover['confidence_pct']}% of quantities carry a confirmed rate"
          if cover.get("confidence_pct") is not None else "\n  Coverage — nothing metered")
    for label, key, note in (
        ("unmetered calls", "unmetered_calls", "no usage at all; totals are a floor"),
        ("unpriced units", "unpriced_units", "costing zero on every chart"),
        ("placeholder rates", "placeholder_units", "a rate nobody has confirmed"),
        ("derived quantities", "estimated_units", "inferred, not reported"),
        ("rates by prefix", "inherited_units", "priced from an ancestor model"),
    ):
        if cover.get(key):
            print(f"    {label:<20}{cover[key]:>5}   ({note})")
    for row in cover.get("unpriced") or []:
        model = f"/{row['model']}" if row["model"] else ""
        print(f"      no rate: {row['provider']}{model} ({row['unit']})")


def one_call(call_id: str) -> None:
    detail = costing.for_call(call_id)
    if not detail.get("metered"):
        print(f"\n  {call_id} recorded no usage.\n"
              "  Its call.ended never arrived, or came from an agent built "
              "before metering.\n")
        return

    print(f"\n  {call_id}   {_money(detail['cost_usd'])}   {_rupees(detail['cost_usd'])}\n")
    print(f"    {'provider/model':<32}{'operation':<15}{'unit':<27}"
          f"{'quantity':>12}  {'cost':>10}")
    for unit in detail["units"]:
        # The operation earns its column: LiveKit bills the agent session and
        # the SIP leg as two separate per-minute charges with the same provider,
        # the same empty model and the same unit. Without it they render as one
        # line printed twice.
        name = f"{unit['provider']}/{unit['model']}" if unit["model"] else unit["provider"]
        flags = []
        if unit["unpriced"]:
            flags.append("NO RATE")
        if unit["placeholder"]:
            flags.append("placeholder")
        if unit["estimated"]:
            flags.append("derived")
        if unit["rate_inherited"]:
            flags.append("rate by prefix")
        if unit["ok"] is False:
            flags.append("FAILED")
        suffix = ("   " + ", ".join(flags)) if flags else ""
        print(f"    {name[:31]:<32}{(unit['operation'] or '—')[:14]:<15}"
              f"{unit['unit']:<27}{unit['quantity']:>12,.0f}  "
              f"{_money(unit['cost_usd'])}{suffix}")

    turns = costing.turns_for_call(call_id)
    if turns:
        total = sum(t["input_tokens"] for t in turns) or 1
        cached = sum(t["cached_tokens"] for t in turns)
        print(f"\n    {len(turns)} responses, {total:,} input tokens, "
              f"{round(100 * cached / total)}% from cache")
        # The slope is the diagnosis: a realtime model re-bills the whole
        # conversation each turn, so what matters is how fast it grew.
        print(f"    context grew {turns[0]['input_tokens']:,} → "
              f"{turns[-1]['input_tokens']:,} tokens over the call")
        barged = sum(1 for t in turns if t["cancelled"])
        if barged:
            print(f"    {barged} response(s) barged into — check turn detection")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="What recent calls cost.")
    parser.add_argument("call_id", nargs="?", help="one call, priced line by line")
    parser.add_argument("--hours", type=int, default=24)
    args = parser.parse_args()

    setup_logging()
    if not settings.database_url:
        print("DATABASE_URL is not set.")
        return 1
    try:
        one_call(args.call_id) if args.call_id else overview(args.hours)
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

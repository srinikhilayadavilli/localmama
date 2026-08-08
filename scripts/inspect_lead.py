"""What the backend made of a call. Reads the database directly.

    python scripts/inspect_lead.py                 # the 10 most recent leads
    python scripts/inspect_lead.py <call_id>       # one lead, in full
    python scripts/inspect_lead.py --review        # only what needs a human
    python scripts/inspect_lead.py --owed          # either channel still owed

`smoke.py` proves the API accepts a call. This shows what came out the other
end — the English values, the confidence scores, the vendors matched, and why a
lead was flagged. That is the part you cannot see over HTTP, and the part that
tells you whether the pipeline is actually any good.
"""

from __future__ import annotations

import argparse
import os
import sys

import psycopg
from dotenv import load_dotenv

load_dotenv()

BOLD, DIM, GREEN, YELLOW, RESET = "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[0m"


def connect():
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        sys.exit("DATABASE_URL is not set.")
    return psycopg.connect(dsn, connect_timeout=10)


def show_one(cur, call_id: str) -> None:
    cur.execute(
        "SELECT call_id, caller_phone, language, raw, name, service, city, status,"
        " confirmed, confidence, needs_review, review_reason, vendors,"
        " whatsapp_status, whatsapp_error, handoff_status, handoff_error,"
        " started_at, ended_at, processed_at"
        " FROM localmama.leads WHERE call_id = %s",
        (call_id,),
    )
    row = cur.fetchone()
    if row is None:
        print(f"No lead for {call_id!r}.")
        return
    (cid, phone, lang, raw, name, service, city, status, confirmed, confidence,
     review, reason, vendors, wa, wa_err, hook, hook_err,
     started, ended, processed) = row

    print(f"\n{BOLD}{cid}{RESET}   {status}"
          f"{'  ' + YELLOW + 'NEEDS REVIEW' + RESET if review else ''}")
    print(f"  caller     {phone or '(anonymous)'}   language={lang or '?'}")
    print(f"  confirmed  {confirmed}"
          f"{DIM}   (None = no read-back happened){RESET}")

    print(f"\n  {BOLD}as the caller said it{RESET}")
    for k, v in (raw or {}).items():
        print(f"    {k:<10} {v}")

    print(f"\n  {BOLD}as it was stored{RESET}")
    for k, v in (("name", name), ("service", service), ("city", city)):
        score = (confidence or {}).get(k, {})
        mark = ""
        if score:
            s = score.get("score", 0)
            mark = f"{DIM}  score={s:.2f}{RESET}" if score.get("evidence") else \
                   f"{DIM}  (no transcript evidence){RESET}"
            if score.get("note"):
                mark += f"{DIM}  {score['note']}{RESET}"
        print(f"    {k:<10} {v or '—'}{mark}")

    if reason:
        print(f"\n  {YELLOW}flagged:{RESET} {reason}")

    print(f"\n  {BOLD}vendors matched{RESET}  ({len(vendors or [])})")
    for v in (vendors or []):
        print(f"    {v.get('title')}  {v.get('phone')}  {DIM}{v.get('category') or ''}{RESET}")
    if not vendors:
        print(f"    {DIM}none — the message falls back to 'our team is "
              f"shortlisting'{RESET}")

    # Two channels while the webhook is being proven. They are separate rows
    # of truth: WhatsApp is what the caller received, the webhook is what the
    # receiver was told.
    print(f"\n  whatsapp   {wa}{('  ' + str(wa_err)) if wa_err else ''}")
    print(f"  webhook    {hook}{('  ' + str(hook_err)) if hook_err else ''}")
    print(f"  timing     started={started}  ended={ended}  processed={processed}")
    if processed is None:
        print(f"    {YELLOW}! never processed — the pipeline did not run or "
              f"failed{RESET}")
    print()


def show_recent(cur, where: str = "", limit: int = 10) -> None:
    cur.execute(
        "SELECT call_id, status, name, service, city, needs_review,"
        " whatsapp_status, handoff_status, created_at FROM localmama.leads"
        f" {where} ORDER BY created_at DESC LIMIT %s",
        (limit,),
    )
    rows = cur.fetchall()
    if not rows:
        print("Nothing to show.")
        return
    print(f"\n  {'call_id':<26} {'status':<11} {'name':<14} {'service':<16} "
          f"{'city':<14} {'whatsapp':<10} {'webhook':<9}")
    print(f"  {DIM}{'-' * 100}{RESET}")
    for cid, status, name, service, city, review, wa, hook, _ in rows:
        flag = f" {YELLOW}!{RESET}" if review else ""
        print(f"  {cid[:24]:<26} {status or '?':<11} {(name or '—')[:12]:<14} "
              f"{(service or '—')[:14]:<16} {(city or '—')[:12]:<14} "
              f"{wa or '—':<10} {hook or '—':<9}{flag}")
    print(f"\n  {DIM}{YELLOW}!{RESET}{DIM} = needs review. "
          f"Full detail: python scripts/inspect_lead.py <call_id>{RESET}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("call_id", nargs="?")
    parser.add_argument("--review", action="store_true", help="only flagged leads")
    parser.add_argument("--owed", action="store_true", help="only owed handoffs")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    with connect() as conn, conn.cursor() as cur:
        if args.call_id:
            show_one(cur, args.call_id)
        elif args.review:
            show_recent(cur, "WHERE needs_review", args.limit)
        elif args.owed:
            show_recent(cur, "WHERE whatsapp_status IN ('pending','sending')"
                            "    OR handoff_status IN ('pending','sending')", args.limit)
        else:
            show_recent(cur, "", args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())

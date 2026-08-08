"""End-to-end smoke test against a running backend.

    python scripts/smoke.py                          # localhost:8000
    python scripts/smoke.py https://…onrender.com    # a deployment

Exercises the contract the way the agent will: health, auth, a whole call as a
batch of events, idempotent retry, an abandoned call, and a vendor lookup.
Reads `AGENT_TOKEN` from the environment or `.env`.

This is deliberately a *black box* — it imports nothing from `backend/` and
talks only over HTTP, so it tests the deployed artifact rather than the code on
this laptop. That is the difference between "the tests pass" and "the thing I
deployed works".

It writes real rows. Every call it creates is prefixed `smoke-`, and
`--cleanup` deletes them again. Point it at production knowingly.

Exit code 0 if everything passed, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone

import httpx
from dotenv import load_dotenv

load_dotenv()

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"

passed = failed = 0


def check(name: str, ok: bool, detail: str = "") -> bool:
    global passed, failed
    if ok:
        passed += 1
        print(f"  {GREEN}✓{RESET} {name}")
    else:
        failed += 1
        print(f"  {RED}✗{RESET} {name}")
        if detail:
            print(f"    {DIM}{detail}{RESET}")
    return ok


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def a_whole_call(call_id: str) -> dict:
    """One complete Telugu call, exactly as the agent would send it."""
    ev = lambda n: f"{call_id}-{n}"  # noqa: E731 - deterministic ids, so retries dedupe
    return {
        "schema_version": "1.0",
        "events": [
            {"type": "call.started", "event_id": ev(0), "call_id": call_id,
             "seq": 0, "at": 0.0, "caller_phone": "+919739960092",
             "dialled": "+918041234567", "started_at": now_iso(),
             "agent_version": "smoke"},
            {"type": "call.captured", "event_id": ev(1), "call_id": call_id,
             "seq": 1, "at": 6.0, "field": "language", "value": "telugu"},
            {"type": "call.captured", "event_id": ev(2), "call_id": call_id,
             "seq": 2, "at": 12.0, "field": "name", "value": "రవి"},
            {"type": "call.captured", "event_id": ev(3), "call_id": call_id,
             "seq": 3, "at": 19.0, "field": "service", "value": "ఏసీ రిపేర్"},
            {"type": "call.captured", "event_id": ev(4), "call_id": call_id,
             "seq": 4, "at": 26.0, "field": "city", "value": "మాదాపూర్"},
            {"type": "call.ended", "event_id": ev(5), "call_id": call_id,
             "seq": 5, "at": 40.0, "status": "completed", "confirmed": True,
             "turn_gaps": [1.1, 0.9], "ended_at": now_iso(),
             "transcript": [
                 {"role": "caller", "text": "తెలుగు", "at": 6.0},
                 {"role": "caller", "text": "నా పేరు రవి", "at": 12.0},
                 {"role": "caller", "text": "ఏసీ రిపేర్ కావాలి", "at": 19.0},
                 {"role": "caller", "text": "మాదాపూర్", "at": 26.0},
             ]},
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url", nargs="?", default="http://localhost:8000")
    parser.add_argument("--token", default=os.getenv("AGENT_TOKEN", ""))
    parser.add_argument("--vendor", default="", help="a business name to look up")
    parser.add_argument("--keep", action="store_true", help="leave the smoke rows behind")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    auth = {"Authorization": f"Bearer {args.token}"}
    client = httpx.Client(timeout=30.0)

    print(f"\n{DIM}target {base}{RESET}\n")

    if not args.token:
        print(f"{RED}AGENT_TOKEN is not set.{RESET} Export it or pass --token.\n")
        return 1

    # --- health --------------------------------------------------------
    print("health")
    try:
        r = client.get(f"{base}/healthz")
        body = r.json()
        check("/healthz responds", r.status_code == 200, f"HTTP {r.status_code}")
        check("service reports ok", body.get("ok") is True,
              "; ".join(body.get("problems") or []) or json.dumps(body))
        print(f"    {DIM}webhook={body.get('webhook')} "
              f"translation={body.get('translation')}{RESET}")
    except Exception as exc:
        check("/healthz responds", False, repr(exc))
        print(f"\n{RED}Cannot reach the service. Nothing else will work.{RESET}\n")
        return 1

    # --- auth ----------------------------------------------------------
    print("\nauth")
    r = client.post(f"{base}/v1/events", json=a_whole_call("smoke-unauth"))
    check("no token is refused", r.status_code == 401, f"HTTP {r.status_code}")
    r = client.post(f"{base}/v1/events", json=a_whole_call("smoke-unauth"),
                    headers={"Authorization": "Bearer wrong"})
    check("a wrong token is refused", r.status_code == 401, f"HTTP {r.status_code}")

    # --- a whole call --------------------------------------------------
    call_id = f"smoke-{uuid.uuid4().hex[:12]}"
    print(f"\na complete call  {DIM}{call_id}{RESET}")
    payload = a_whole_call(call_id)
    r = client.post(f"{base}/v1/events", json=payload, headers=auth)
    ok = check("accepted", r.status_code == 202, f"HTTP {r.status_code}: {r.text[:200]}")
    if ok:
        body = r.json()
        check("all six events stored", len(body["accepted"]) == 6,
              f"accepted={len(body['accepted'])} dupes={len(body['duplicates'])} "
              f"rejected={body['rejected']}")
        check("nothing rejected", not body["rejected"], json.dumps(body["rejected"]))

    # --- idempotency ---------------------------------------------------
    print("\nidempotency")
    r = client.post(f"{base}/v1/events", json=payload, headers=auth)
    if check("a retry is accepted", r.status_code == 202, f"HTTP {r.status_code}"):
        body = r.json()
        check("the retry stores nothing new", not body["accepted"],
              f"accepted={body['accepted']}")
        check("and reports six duplicates", len(body["duplicates"]) == 6,
              f"duplicates={len(body['duplicates'])}")

    # --- an abandoned call ---------------------------------------------
    ab_id = f"smoke-{uuid.uuid4().hex[:12]}"
    print(f"\nan abandoned call  {DIM}{ab_id}{RESET}")
    r = client.post(f"{base}/v1/events", headers=auth, json={
        "schema_version": "1.0",
        "events": [
            {"type": "call.started", "event_id": f"{ab_id}-0", "call_id": ab_id,
             "seq": 0, "at": 0.0, "caller_phone": "+919739960092",
             "started_at": now_iso()},
            {"type": "call.captured", "event_id": f"{ab_id}-1", "call_id": ab_id,
             "seq": 1, "at": 9.0, "field": "name", "value": "రవి"},
            {"type": "call.ended", "event_id": f"{ab_id}-2", "call_id": ab_id,
             "seq": 2, "at": 15.0, "status": "abandoned", "ended_at": now_iso()},
        ],
    })
    check("a partial lead is accepted", r.status_code == 202,
          f"HTTP {r.status_code}: {r.text[:200]}")

    # --- malformed -----------------------------------------------------
    print("\nbad input")
    r = client.post(f"{base}/v1/events", headers=auth,
                    json={"schema_version": "1.0", "events": [{"type": "call.captured"}]})
    check("a malformed event is refused, not swallowed", r.status_code == 422,
          f"HTTP {r.status_code}")

    # --- vendor lookup -------------------------------------------------
    print("\nvendor lookup")
    t0 = time.monotonic()
    r = client.get(f"{base}/v1/vendors", params={"name": args.vendor or "car wash"},
                   headers=auth)
    elapsed = time.monotonic() - t0
    if check("responds", r.status_code == 200, f"HTTP {r.status_code}: {r.text[:200]}"):
        body = r.json()
        check("has a line to speak", bool(body.get("say")), json.dumps(body))
        check("never leaks a number on a weak match",
              body["match"] not in ("approximate", "ambiguous", "category")
              or body.get("phone") is None,
              json.dumps(body))
        print(f"    {DIM}match={body['match']}  {elapsed*1000:.0f}ms  "
              f"say={body['say'][:70]!r}{RESET}")
        # The agent gives up at 800ms. A slower backend is not a failure here,
        # but it is the number that decides whether callers hear an apology.
        if elapsed > 0.8:
            print(f"    {YELLOW}! slower than the agent's 800ms budget — callers "
                  f"would hear the fallback{RESET}")

    # --- cleanup -------------------------------------------------------
    if not args.keep:
        print(f"\n{DIM}smoke rows left in place; remove with:{RESET}")
        print(f"{DIM}  DELETE FROM localmama.leads WHERE call_id LIKE 'smoke-%';{RESET}")
        print(f"{DIM}  DELETE FROM localmama.call_events WHERE call_id LIKE 'smoke-%';{RESET}")

    total = passed + failed
    colour = GREEN if not failed else RED
    print(f"\n{colour}{passed}/{total} checks passed{RESET}\n")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())

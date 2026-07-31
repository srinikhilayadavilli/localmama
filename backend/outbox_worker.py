"""Retry WhatsApp handoffs that did not get through, and expire transcripts.

    python -m backend.outbox_worker            # run forever, on a timer
    python -m backend.outbox_worker --once     # one pass, then exit
    python -m backend.outbox_worker --status   # report without sending

A lead whose message never went out is work still owed to a customer. Inside
the agent this was a daemon thread in the voice process, which is the wrong
place twice over: it competed with a live call for the event loop, and it only
ran while a worker happened to be up.

Here it is its own Render process. Safe to run alongside others: rows are
*claimed* with `UPDATE ... RETURNING` and `FOR UPDATE SKIP LOCKED`, so two
workers sweeping at the same moment take disjoint batches rather than sending
one customer the same message twice.

One attempt per lead per pass. The sweep *is* the retry — trying three times
with backoff on every lead only makes each pass longer while a provider is
down, and 29 owed leads at three attempts each is minutes of work to learn one
fact.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from . import db, store
from .config import settings
from .logger import get_logger, setup_logging
from .services import brain, whatsapp

logger = get_logger("localmama.outbox")


async def drain(limit: int = 50) -> tuple[int, int]:
    """Retry every owed handoff. Returns (sent, still_failing)."""
    owed = store.claim_owed_handoffs(limit=limit)
    if not owed:
        return 0, 0

    sent = failed = 0
    for row in owed:
        vendors = row.get("vendors") or []
        options = " · ".join(
            f"{v['title']} {brain.spoken_phone(v.get('phone'))}".strip()
            for v in vendors if v.get("title") and v.get("phone")
        )
        result = await whatsapp.send(
            row["caller_phone"],
            name=row.get("name"), service=row.get("service"), city=row.get("city"),
            options=options, attempts=1,
        )
        ok = bool(result.get("ok"))
        error = "" if ok else str(result.get("reason") or result.get("error") or "")
        store.mark_whatsapp(row["call_id"], ok, error)
        if ok:
            sent += 1
            logger.info("sent %s (attempt %d)", row["call_id"][:8], row["attempts"] + 1)
        else:
            failed += 1
            logger.warning("still failing %s (attempt %d): %s",
                           row["call_id"][:8], row["attempts"] + 1, error[:120])
    return sent, failed


async def reprocess() -> int:
    """Finish any call that ended without being turned into a lead.

    The pipeline runs in a background task after the API has answered, so a
    deploy, a restart or an uncaught exception in between leaves a call with
    every value captured and nothing done with it — no vendors, no message, and
    invisible to the outbox because it was never processed.

    Those are the leads that vanish without anyone noticing. Idempotent, so a
    lead that was merely mid-flight gets processed twice and is none the worse.
    """
    from . import pipeline

    owed = store.unprocessed_leads()
    for call_id in owed:
        try:
            await pipeline.process(call_id)
            logger.info("recovered %s, which ended without being processed",
                        call_id[:8])
        except Exception:  # noqa: BLE001 - one bad lead, not the sweep
            logger.exception("could not recover %s", call_id[:8])
    return len(owed)


async def one_pass() -> None:
    recovered = await reprocess()
    sent, failed = await drain()
    if recovered or sent or failed:
        logger.info("sweep: recovered %d, sent %d, still owed %d",
                    recovered, sent, failed)
    store.expire_transcripts()


async def forever() -> None:
    interval = settings.outbox_sweep_seconds
    logger.info("sweeping every %.0fs", interval)
    while True:
        try:
            await one_pass()
        except Exception:  # noqa: BLE001 - a sweep must never kill the worker
            logger.exception("sweep failed; continuing")
        await asyncio.sleep(interval)


def main() -> int:
    parser = argparse.ArgumentParser(description="Retry owed WhatsApp handoffs.")
    parser.add_argument("--once", action="store_true", help="one pass, then exit")
    parser.add_argument("--status", action="store_true", help="report without sending")
    args = parser.parse_args()

    setup_logging()
    if not db.available():
        print("DATABASE_URL is not set, so there is no outbox to read.")
        return 1

    if args.status:
        owed = store.pending_handoffs()
        print(f"{len(owed)} lead(s) owed a WhatsApp message")
        for row in owed:
            print(f"   {row['call_id'][:8]}  {row['caller_phone']:16} "
                  f"{row['name']} / {row['service']} / {row['city']}  "
                  f"({row['status']}, attempts: {row['attempts']})"
                  + (f"  last error: {row['error'][:60]}" if row.get("error") else ""))
        return 0

    if not settings.whatsapp_available:
        logger.warning(
            "WhatsApp is not configured; owed leads will be marked skipped rather "
            "than retried forever."
        )

    asyncio.run(one_pass() if args.once else forever())
    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

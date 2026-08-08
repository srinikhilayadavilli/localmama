"""Retry lead handoffs that did not get through, and expire transcripts.

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

from . import costing, db, store
from .config import settings
from .logger import get_logger, setup_logging
from .services import brain, meter, webhook, whatsapp

logger = get_logger("localmama.outbox")


async def drain(limit: int = 50) -> tuple[int, int]:
    """Retry every owed handoff on both channels. Returns (delivered, failing).

    Swept per channel. A lead whose WhatsApp landed and whose webhook failed is
    owed the webhook and nothing else — one shared sweep would message that
    customer again to fix a problem they never had.
    """
    sent = failed = 0
    for channel in ("whatsapp", "webhook"):
        s, f = await _drain_channel(channel, limit=limit)
        sent += s
        failed += f
    return sent, failed


async def _drain_channel(channel: str, limit: int = 50) -> tuple[int, int]:
    owed = store.claim_owed_handoffs(channel, limit=limit)
    if not owed:
        return 0, 0

    sent = failed = 0
    for row in owed:
        # The claim returns the whole lead, so the sweep builds the same body
        # the live path does. `subject` is reconstructed the way the pipeline
        # chose it — the caller's own words for the trade when we have them.
        subject = row.get("service_said") or row.get("service") or ""
        # Attributed to the call it is owed to, so a lead that took nine
        # sweeps to deliver carries nine attempts against its own cost rather
        # than against nothing.
        with meter.for_call(row["call_id"]):
            if channel == "whatsapp":
                options = " · ".join(
                    f"{v['title']} {brain.spoken_phone(v.get('phone'))}".strip()
                    for v in (row.get("vendors") or [])
                    if v.get("title") and v.get("phone")
                )
                result = await whatsapp.send(
                    row["caller_phone"], name=row.get("name"), service=subject,
                    city=row.get("city"), options=options, attempts=1,
                )
                detail = result.get("messageId")
            else:
                result = await webhook.send(row, subject=subject, attempts=1)
                detail = result.get("status")
        ok = bool(result.get("ok"))
        error = "" if ok else str(result.get("reason") or result.get("error") or "")
        store.mark_handoff(row["call_id"], channel, ok, error, detail=detail)
        if ok:
            sent += 1
            logger.info("%s delivered %s (attempt %d)",
                        channel, row["call_id"][:8], row["attempts"] + 1)
        else:
            failed += 1
            logger.warning("%s still failing %s (attempt %d): %s",
                           channel, row["call_id"][:8], row["attempts"] + 1, error[:120])
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
    # Drain first, then recover. The other order gives a lead recovered in this
    # pass two goes at once: `reprocess` runs the pipeline, whose `notify`
    # already tries three times with backoff, and then `drain` finds whatever
    # came back `pending` and tries again — four attempts in a sweep that
    # promises one. Recovered leads are still delivered in this pass, by that
    # `notify`; they simply are not also swept until the next one.
    #
    # It cost little with one channel. With two there are two ways to come back
    # `pending`, so the same lead can burn attempts on a provider that is down
    # twice as fast, and `OUTBOX_MAX_ATTEMPTS` is what stops a lead being
    # retried forever.
    sent, failed = await drain()
    recovered = await reprocess()
    if recovered or sent or failed:
        logger.info("sweep: delivered %d, still owed %d, recovered %d",
                    sent, failed, recovered)
    store.expire_transcripts()
    costing.prune_turns()


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
    parser = argparse.ArgumentParser(description="Retry owed lead handoffs.")
    parser.add_argument("--once", action="store_true", help="one pass, then exit")
    parser.add_argument("--status", action="store_true", help="report without sending")
    args = parser.parse_args()

    setup_logging()
    if not db.available():
        print("DATABASE_URL is not set, so there is no outbox to read.")
        return 1

    if args.status:
        for channel in ("whatsapp", "webhook"):
            owed = store.pending_handoffs(channel)
            print(f"{len(owed)} lead(s) owed a {channel}")
            _report(owed)
        return 0

    if not settings.whatsapp_available:
        logger.warning(
            "WhatsApp is not configured; leads stay pending rather than being "
            "marked terminal, because this is a state of the deployment and "
            "not of the lead."
        )
    if not settings.webhook_available:
        logger.warning(
            "No webhook is configured (WEBHOOK_URL / WEBHOOK_SECRET); owed leads "
            "stay pending rather than being marked terminal, because this is a "
            "state of the deployment and not of the lead."
        )

    asyncio.run(one_pass() if args.once else forever())
    db.close()
    return 0


def _report(owed: list) -> None:
    for row in owed:
        print(f"   {row['call_id'][:8]}  {row['caller_phone']:16} "
              f"{row['name']} / {row['service']} / {row['city']}  "
              f"({row['status']}, attempts: {row['attempts']})"
              + (f"  last error: {row['error'][:60]}" if row.get("error") else ""))


if __name__ == "__main__":
    sys.exit(main())

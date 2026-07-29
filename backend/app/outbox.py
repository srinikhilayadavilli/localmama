"""Retry WhatsApp handoffs that did not get through.

    python -m backend.app.outbox            # retry what is owed
    python -m backend.app.outbox --status   # just report

A lead whose message never went out is work still owed to a customer, and until
now it was lost the moment the call ended: three in-call retries, then nothing.
That is a poor fit for the actual failure mode — the provider being down for
hours — where the right answer is to try again later, not to try harder now.

So the outbox state lives on the lead row in Postgres rather than in a process
that dies when the caller hangs up. `whatsapp_status` is `pending` until a send
succeeds (`sent`) or there was no number to send to (`skipped`).

It runs at worker startup, then every `OUTBOX_SWEEP_SECONDS`, and can be run by
hand. Safe to run concurrently: rows are *claimed* with `UPDATE ... RETURNING`
and `FOR UPDATE SKIP LOCKED`, so two workers sweeping at the same moment take
disjoint batches rather than sending one customer the same message twice.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

from .config import settings
from .logger import get_logger, setup_logging
from .models import ConversationStatus, Lead, utcnow
from .services import lead_store, whatsapp

logger = get_logger("localmama.outbox")


def _as_lead(row: dict) -> Lead:
    """Rebuild just enough of a Lead for the WhatsApp template.

    The transcript is deliberately not loaded: the template needs a name, a
    service and a city, and pulling whole transcripts to send a four-field
    message would make the sweep far heavier than the work it does.
    """
    from .languages import Language

    language = None
    if row.get("language"):
        try:
            language = Language(row["language"])
        except ValueError:
            language = None
    return Lead(
        session_id=row["session_id"],
        selected_language=language,
        user_name=row.get("name"),
        requested_service=row.get("service"),
        city_or_area=row.get("city"),
        conversation_status=ConversationStatus.COMPLETED,
        transcript=[],
        started_at=utcnow(),
        completed_at=utcnow(),
    )


async def drain(limit: int = 50) -> tuple[int, int]:
    """Retry every owed handoff. Returns (sent, still_failing).

    Claims rows before sending, so two workers sweeping at the same moment
    cannot send one customer the same message twice.
    """
    owed = lead_store.claim_whatsapp(limit=limit)
    if not owed:
        return 0, 0

    logger.info("outbox: %d lead(s) owed a WhatsApp message", len(owed))
    sent = failed = 0
    for row in owed:
        # One attempt per lead: the sweep itself is the retry.
        result = await whatsapp.send(_as_lead(row), row["caller_phone"], attempts=1)
        ok = bool(result.get("ok"))
        error = "" if ok else str(result.get("reason") or result.get("error") or "")
        lead_store.mark_whatsapp(row["session_id"], ok, error)
        if ok:
            sent += 1
            logger.info("outbox: sent %s (attempt %d)", row["session_id"][:8],
                        row["attempts"] + 1)
        else:
            failed += 1
            logger.warning("outbox: still failing %s (attempt %d): %s",
                           row["session_id"][:8], row["attempts"] + 1, error[:120])
    return sent, failed


def start_periodic_sweep() -> "threading.Thread | None":
    """Retry owed handoffs on a timer, for the life of the worker.

    Runs in a daemon thread rather than the worker's event loop. LiveKit
    creates job processes with `spawn`/`forkserver`, never plain `fork`, so a
    thread here is not inherited by a call in progress and cannot interfere
    with one.

    The startup sweep alone left a gap that matters: if the provider recovers
    an hour into a deployment, nothing goes out until the next restart. This
    closes it without needing a scheduler the deployment does not have.

    Returns None when disabled or unconfigured, so the caller can log which.
    """
    import threading

    interval = settings.outbox_sweep_seconds
    if interval <= 0 or not lead_store.available():
        return None

    def _loop() -> None:
        # Sweep immediately, then on the interval. The first pass used to run
        # inline in main() before the worker registered, so a backlog the
        # provider could not accept held up accepting calls — 29 owed handoffs
        # at three attempts each is minutes of a worker that answers nothing,
        # and it grows with the backlog. Answering the phone matters more than
        # draining the outbox promptly.
        first = True
        while True:
            if not first:
                time.sleep(interval)
            first = False
            try:
                # Its own event loop: this thread is not the worker's.
                sent, failed = asyncio.run(drain())
                if sent or failed:
                    logger.info("outbox sweep: sent %d, still owed %d", sent, failed)
            except Exception as exc:  # noqa: BLE001 - a sweep must never kill the worker
                logger.warning("outbox sweep failed: %s", exc)

    thread = threading.Thread(target=_loop, name="outbox-sweep", daemon=True)
    thread.start()
    logger.info("outbox sweep every %.0fs", interval)
    return thread


def whatsapp_configured() -> bool:
    return settings.whatsapp_available


def main() -> int:
    parser = argparse.ArgumentParser(description="Retry owed WhatsApp handoffs.")
    parser.add_argument("--status", action="store_true", help="report without sending")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    setup_logging()
    if not lead_store.available():
        print("DATABASE_URL is not set, so there is no outbox to read.")
        return 1

    owed = lead_store.pending_whatsapp(limit=args.limit)
    print(f"{len(owed)} lead(s) owed a WhatsApp message")
    for row in owed:
        print(f"   {row['session_id'][:8]}  {row['caller_phone']:16} "
              f"{row['name']} / {row['service']} / {row['city']}  "
              f"(attempts: {row['attempts']})")

    if args.status:
        return 0
    if not owed:
        return 0
    if not whatsapp_configured():
        print("\nWhatsApp is not configured, so nothing can be sent yet.")
        return 1

    sent, failed = asyncio.run(drain(limit=args.limit))
    print(f"\nsent {sent}, still failing {failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())

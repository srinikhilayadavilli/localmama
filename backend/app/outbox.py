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

This runs at worker startup and can be run by hand or from cron. It is safe to
run concurrently with live calls: each lead is updated by primary key, and a
send that succeeds twice is a duplicate message, not a corrupt record — which is
why `sent` is only ever written after the provider confirms.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

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
    """Retry every owed handoff. Returns (sent, still_failing)."""
    owed = lead_store.pending_whatsapp(limit=limit)
    if not owed:
        return 0, 0

    logger.info("outbox: %d lead(s) owed a WhatsApp message", len(owed))
    sent = failed = 0
    for row in owed:
        result = await whatsapp.send(_as_lead(row), row["caller_phone"])
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


def drain_in_background() -> None:
    """Fire the sweep from an already-running worker, without blocking startup.

    A worker that cannot reach the provider must still take calls, so this never
    waits and never raises into the caller.
    """
    if not (lead_store.available() and whatsapp_configured()):
        return
    try:
        task = asyncio.get_running_loop().create_task(drain())
    except RuntimeError:
        return  # no loop yet; the CLI entry point covers that case
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)


#: Strong refs, so a sweep is not collected mid-flight.
_tasks: set = set()


def whatsapp_configured() -> bool:
    from .config import settings

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

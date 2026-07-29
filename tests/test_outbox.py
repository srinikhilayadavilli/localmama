"""The outbox — what happens when post-call work fails.

Three holes this closes, all of which lost real work:

  1. `add_done_callback(discard)` never inspected `.exception()`, so a failure
     in post-call work reached neither the logs nor anyone looking.
  2. Shutdown did not wait. The agent hangs up ~1.5s after the outro and the
     process exits, while a WhatsApp attempt takes several seconds — so it was
     being killed mid-flight.
  3. There was no retry after the call. Three in-call attempts, then the
     handoff was lost forever — a poor fit for the actual failure mode, a
     provider down for hours.
"""

from __future__ import annotations

import asyncio

import pytest

from backend.app.models import ConversationStatus, Lead, utcnow
from backend.app.languages import Language


def _row(**over):
    row = {"session_id": "s1", "caller_phone": "+919739960092", "name": "Ravi",
           "service": "plumber", "city": "Hyderabad", "language": "english",
           "attempts": 0}
    row.update(over)
    return row


def test_a_row_rebuilds_into_a_sendable_lead() -> None:
    from backend.app.outbox import _as_lead

    lead = _as_lead(_row())
    assert isinstance(lead, Lead)
    assert lead.user_name == "Ravi"
    assert lead.requested_service == "plumber"
    assert lead.selected_language is Language.ENGLISH
    # Transcripts are not loaded: the template needs four fields, and pulling
    # whole transcripts would make the sweep heavier than the work it does.
    assert lead.transcript == []


def test_an_unknown_language_does_not_break_the_sweep() -> None:
    """A bad value must not stop every other owed message going out."""
    from backend.app.outbox import _as_lead

    assert _as_lead(_row(language="klingon")).selected_language is None


@pytest.mark.asyncio
async def test_a_failed_send_stays_owed(monkeypatch) -> None:
    from backend.app import outbox
    from backend.app.services import lead_store, whatsapp

    monkeypatch.setattr(lead_store, "pending_whatsapp", lambda **k: [_row()])
    marks: list = []
    monkeypatch.setattr(lead_store, "mark_whatsapp",
                        lambda sid, ok, error="": marks.append((sid, ok, error)))

    async def fails(lead, phone, options=""):
        return {"ok": False, "error": "ConnectError('')"}

    monkeypatch.setattr(whatsapp, "send", fails)

    sent, failed = await outbox.drain()
    assert (sent, failed) == (0, 1)
    assert marks == [("s1", False, "ConnectError('')")], "must stay pending"


@pytest.mark.asyncio
async def test_a_successful_send_is_recorded_as_terminal(monkeypatch) -> None:
    from backend.app import outbox
    from backend.app.services import lead_store, whatsapp

    monkeypatch.setattr(lead_store, "pending_whatsapp", lambda **k: [_row()])
    marks: list = []
    monkeypatch.setattr(lead_store, "mark_whatsapp",
                        lambda sid, ok, error="": marks.append((sid, ok, error)))

    async def succeeds(lead, phone, options=""):
        return {"ok": True, "messageId": "abc"}

    monkeypatch.setattr(whatsapp, "send", succeeds)

    sent, failed = await outbox.drain()
    assert (sent, failed) == (1, 0)
    assert marks == [("s1", True, "")]


@pytest.mark.asyncio
async def test_nothing_owed_is_not_an_error(monkeypatch) -> None:
    from backend.app import outbox
    from backend.app.services import lead_store

    monkeypatch.setattr(lead_store, "pending_whatsapp", lambda **k: [])
    assert await outbox.drain() == (0, 0)


def test_no_phone_is_skipped_not_retried_forever() -> None:
    """A browser or anonymous caller has no number. That is not a failure, and
    retrying it on every worker start for the life of the deployment is waste."""
    import inspect

    from backend.app.services import lead_store

    source = inspect.getsource(lead_store.mark_whatsapp)
    assert '"skipped"' in source and 'error == "no phone"' in source

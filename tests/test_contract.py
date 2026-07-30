"""The contract is the one thing two deployables share, so it is tested on its
own — no agent, no backend, no database. If these pass, both sides agree.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from contract import (
    SCHEMA_VERSION,
    CallCaptured,
    CallEnded,
    CallStarted,
    CallStatus,
    CapturedField,
    EventBatch,
    Language,
    MatchKind,
    VendorReply,
)


def _now() -> datetime:
    return datetime(2026, 7, 31, 9, 12, 4, tzinfo=timezone.utc)


# --- values cross the boundary raw ----------------------------------------


def test_a_captured_value_keeps_the_callers_script():
    """The whole point of the boundary: nothing is normalised on the way out."""
    event = CallCaptured(
        event_id="e1", call_id="c1", seq=2, at=8.4,
        field=CapturedField.NAME, value="రవి",
    )
    assert event.value == "రవి"


def test_a_transcription_accident_is_capped_not_stored():
    """A stuck recogniser produces a paragraph. It must not become a name."""
    with pytest.raises(ValidationError):
        CallCaptured(
            event_id="e1", call_id="c1", seq=2, at=8.4,
            field=CapturedField.NAME, value="x" * 201,
        )


def test_an_empty_value_is_not_a_capture():
    with pytest.raises(ValidationError):
        CallCaptured(
            event_id="e1", call_id="c1", seq=2, at=8.4,
            field=CapturedField.NAME, value="",
        )


# --- the batch is the transport -------------------------------------------


def test_a_batch_round_trips_all_three_event_types():
    """One endpoint, three shapes, discriminated on `type` — so the agent's
    client needs one retry path rather than three."""
    batch = EventBatch(events=[
        CallStarted(
            event_id="e0", call_id="c1", seq=0, at=0.0,
            caller_phone="+919739960092", dialled="+918041234567",
            started_at=_now(), agent_version="1.4.0",
        ),
        CallCaptured(
            event_id="e1", call_id="c1", seq=1, at=6.2,
            field=CapturedField.LANGUAGE, value=Language.TELUGU.value,
        ),
        CallEnded(
            event_id="e2", call_id="c1", seq=9, at=71.0,
            status=CallStatus.COMPLETED, confirmed=True,
            turn_gaps=[1.1, 0.9], ended_at=_now(),
        ),
    ])

    reparsed = EventBatch.model_validate_json(batch.model_dump_json())

    assert reparsed.schema_version == SCHEMA_VERSION
    assert [e.type for e in reparsed.events] == [
        "call.started", "call.captured", "call.ended",
    ]
    # The discriminator has to survive the round trip as the concrete class,
    # not as a dict — the backend dispatches on it.
    assert isinstance(reparsed.events[1], CallCaptured)
    assert reparsed.events[1].field is CapturedField.LANGUAGE


def test_an_empty_batch_is_refused():
    """Nothing to accept and nothing to acknowledge — a bug on the agent side,
    and cheaper to catch here than to serve a 202 for."""
    with pytest.raises(ValidationError):
        EventBatch(events=[])


# --- the states that are not booleans -------------------------------------


def test_unconfirmed_and_unread_back_are_different_states():
    """`None` means no read-back happened; `False` means the caller said the
    details were wrong. Collapsing them loses the distinction the review queue
    is built on."""
    no_readback = CallEnded(
        event_id="e", call_id="c", seq=1, at=30.0,
        status=CallStatus.ABANDONED, ended_at=_now(),
    )
    disagreed = CallEnded(
        event_id="e", call_id="c", seq=1, at=30.0,
        status=CallStatus.ABANDONED, confirmed=False, ended_at=_now(),
    )
    assert no_readback.confirmed is None
    assert disagreed.confirmed is False


def test_an_abandoned_call_still_closes_the_record():
    """A caller who gave a name and a service and hung up is a lead. The
    backend cannot process one it was never told had ended."""
    ended = CallEnded(
        event_id="e", call_id="c", seq=4, at=22.0,
        status=CallStatus.ABANDONED, ended_at=_now(),
    )
    assert ended.status is CallStatus.ABANDONED


# --- the vendor lookup -----------------------------------------------------


def test_the_agent_is_handed_a_line_to_speak_not_a_phone_number():
    """Digit grouping and 'is this weak enough to withhold' are directory
    policy. They belong on the side that owns the directory."""
    reply = VendorReply(
        match=MatchKind.EXACT,
        say="Pax Jwellers — nine eight seven six five, four three two one zero.",
        instruction="Read it in digit groups and offer to repeat it.",
        title="Pax Jwellers", phone="+919876543210",
    )
    assert "nine eight" in reply.say


def test_a_weak_match_carries_no_number_to_read():
    """An approximate hit is a question, not an answer: nothing structured is
    returned, so the agent has nothing to read out even if it ignores `say`."""
    reply = VendorReply(
        match=MatchKind.APPROXIMATE,
        say="I have a Pax Jwellers — is that the one you mean?",
        instruction="Confirm the name before reading any number.",
    )
    assert reply.phone is None
    assert reply.title is None


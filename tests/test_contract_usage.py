"""Usage on the wire, and the deploy-order hazard it was shaped to avoid.

`EventBatch` is a discriminated union, so pydantic validates it as a whole: an
unrecognised `type` fails the entire batch, not one event. That is why usage
rides as optional fields on `call.ended` rather than as an event of its own —
a new event type deployed to the agent before the backend would have 422'd the
batch carrying `call.ended` and lost the lead in order to add a cost number.

These tests hold that property down in both directions, because it is the kind
of thing that is easy to undo later by adding "just one more event type".
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from contract import (
    CallEnded,
    CallStatus,
    EventBatch,
    TurnMetric,
    Unit,
    UsageRecord,
)


def _ended(**kw) -> CallEnded:
    base = dict(
        event_id="e1", call_id="c1", seq=4, at=91.2,
        status=CallStatus.COMPLETED, confirmed=True,
        ended_at=datetime(2026, 8, 3, 9, 0, tzinfo=timezone.utc),
    )
    return CallEnded(**{**base, **kw})


def test_an_old_agent_sending_no_usage_is_still_valid():
    """Forwards compatibility: the backend deploys first, and every agent still
    running the previous build must keep filing leads."""
    ended = _ended()
    assert ended.usage == []
    assert ended.turns == []


def test_a_new_agents_usage_survives_a_round_trip():
    ended = _ended(usage=[UsageRecord(
        provider="openai", model="gpt-realtime", operation="realtime",
        unit=Unit.AUDIO_INPUT_TOKENS, quantity=12_000.0,
    )])
    parsed = CallEnded.model_validate(ended.model_dump(mode="json"))
    assert parsed.usage[0].unit is Unit.AUDIO_INPUT_TOKENS
    assert parsed.usage[0].quantity == 12_000.0


def test_an_old_backend_ignores_usage_it_does_not_know_about():
    """Backwards compatibility, the direction that actually loses leads: an
    agent deployed ahead of the backend. Pydantic ignores unknown fields by
    default, so the lead lands and only the cost number is missed."""
    payload = _ended().model_dump(mode="json")
    payload["some_future_field"] = {"anything": 1}
    parsed = CallEnded.model_validate(payload)
    assert parsed.call_id == "c1"


def test_a_batch_carrying_usage_validates_as_a_whole():
    batch = EventBatch(events=[_ended(usage=[UsageRecord(
        provider="livekit", operation="telephony", unit=Unit.SECONDS,
        quantity=184.0,
    )])])
    assert EventBatch.model_validate(batch.model_dump(mode="json")).events[0].usage


def test_an_unknown_event_type_fails_the_whole_batch():
    """The hazard itself, asserted rather than described. This is why usage is
    a field: had it been an event, this failure would have taken the
    `call.ended` beside it down too."""
    payload = {
        "schema_version": "1.0",
        "events": [
            _ended().model_dump(mode="json"),
            {"type": "call.usage", "event_id": "e2", "call_id": "c1",
             "seq": 5, "at": 91.3},
        ],
    }
    with pytest.raises(ValidationError):
        EventBatch.model_validate(payload)


def test_a_negative_quantity_is_refused_at_the_boundary():
    """A negative unit reaching the rate card is a call that appears to earn
    money. Rejected here as well as clamped in the meter, because the two
    defend different failures: a bug upstream, and a bug in the meter."""
    with pytest.raises(ValidationError):
        UsageRecord(provider="openai", unit=Unit.AUDIO_INPUT_TOKENS, quantity=-1.0)


def test_an_unknown_unit_is_refused():
    """The unit is the join key against the rate card. A free-text unit that
    matches no row prices at zero, silently — so the vocabulary is closed."""
    with pytest.raises(ValidationError):
        UsageRecord(provider="openai", unit="audio_in", quantity=1.0)


def test_usage_defaults_to_measured_not_estimated():
    """`estimated` has to be opt-in. A default of True would quietly downgrade
    confidence in every measured number; a default of False means only the one
    place that derives a quantity says so."""
    assert UsageRecord(provider="openai", unit=Unit.SECONDS, quantity=1.0).estimated is False


def test_the_turn_series_is_bounded_on_the_wire():
    """A looping model must not make `call.ended` too large to send. The cap is
    enforced in the meter; this is the boundary refusing to carry more."""
    turns = [TurnMetric(ref=f"r{i}", at=float(i)) for i in range(501)]
    with pytest.raises(ValidationError):
        _ended(turns=turns)

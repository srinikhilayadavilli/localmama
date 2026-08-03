"""Transcript retention, against a real Postgres.

The bug these were written for: `expire_transcripts` cleared the lead row and
reported success while the identical transcript sat untouched inside
`call_events.payload`, where the `call.ended` event keeps every event exactly as
received. The sweep ran every five minutes for months and the data never went
anywhere. A half-deletion is worse than none, because the log line saying it
worked is what stops anyone checking.

So most of what is below asserts the *second* copy is gone, and that nothing
else in the payload went with it.

    make test-db          # or: TEST_DATABASE_URL=... pytest
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

import pytest

TEST_DSN = os.environ.get("TEST_DATABASE_URL", "")

pytestmark = pytest.mark.skipif(
    not TEST_DSN,
    reason="set TEST_DATABASE_URL to a scratch Postgres to exercise retention",
)

_TRANSCRIPT = [
    {"role": "caller", "text": "naa peru Ravi", "at": 3.1},
    {"role": "agent", "text": "Ravi gaaru, cheppandi", "at": 4.4},
]


@pytest.fixture()
def store(monkeypatch):
    """A migrated, empty schema plus `store` bound to it."""
    import psycopg

    from backend import db, store as store_module
    from backend.config import settings
    from backend.migrate import available_migrations

    object.__setattr__(settings, "database_url", TEST_DSN)
    object.__setattr__(settings, "transcript_retention_days", 30)
    db.close()

    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("DROP SCHEMA IF EXISTS localmama CASCADE")
        for _, sql in available_migrations():
            conn.execute(sql)

    yield store_module
    db.close()


def _sql(sql: str, params: tuple = ()):
    import psycopg

    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        cur = conn.execute(sql, params)
        return cur.fetchall() if cur.description else None


def _call(call_id: str, *, days_ago: float, ended: bool = True) -> None:
    """A finished call with its transcript in both places, as the API stores it."""
    when = datetime.now(timezone.utc) - timedelta(days=days_ago)
    _sql(
        "INSERT INTO localmama.leads (call_id, agent_id, caller_phone, service,"
        " status, transcript, started_at, ended_at)"
        " VALUES (%s, 'localmama', '+919999', 'plumber', 'completed', %s, %s, %s)",
        (call_id, json.dumps(_TRANSCRIPT), when - timedelta(minutes=3),
         when if ended else None),
    )
    _sql(
        "INSERT INTO localmama.call_events (event_id, call_id, seq, type, payload,"
        " received_at) VALUES (%s, %s, 3, 'call.ended', %s, %s)",
        (f"{call_id}-ended", call_id,
         json.dumps({"type": "call.ended", "call_id": call_id, "seq": 3,
                     "status": "completed", "confirmed": True,
                     "transcript": _TRANSCRIPT, "turn_gaps": [0.9, 1.4]}),
         when),
    )
    # A capture event, which carries the caller's words but mirrors `leads.raw`
    # — deliberately retained as the business record, so it must survive.
    _sql(
        "INSERT INTO localmama.call_events (event_id, call_id, seq, type, payload,"
        " received_at) VALUES (%s, %s, 2, 'call.captured', %s, %s)",
        (f"{call_id}-captured", call_id,
         json.dumps({"type": "call.captured", "call_id": call_id, "seq": 2,
                     "field": "name", "value": "రవి"}),
         when),
    )


def _lead_transcript(call_id: str):
    return _sql("SELECT transcript FROM localmama.leads WHERE call_id = %s",
                (call_id,))[0][0]


def _event_payload(event_id: str) -> dict:
    return _sql("SELECT payload FROM localmama.call_events WHERE event_id = %s",
                (event_id,))[0][0]


def test_an_expired_transcript_is_gone_from_both_copies(store):
    """The bug itself. Clearing the lead row while the event log keeps a
    complete copy is not a deletion, it is a deletion-shaped log line."""
    _call("old", days_ago=45)
    store.expire_transcripts()

    assert _lead_transcript("old") == []
    assert _event_payload("old-ended")["transcript"] == []


def test_a_transcript_inside_the_window_is_untouched(store):
    """It is still evidence: the confidence audit runs on it, and so does
    anyone diagnosing a lead that came out wrong."""
    _call("recent", days_ago=3)
    store.expire_transcripts()

    assert _lead_transcript("recent") == _TRANSCRIPT
    assert _event_payload("recent-ended")["transcript"] == _TRANSCRIPT


def test_redaction_leaves_the_rest_of_the_payload_alone(store):
    """Only the transcript goes. The event row is the deduplication key and the
    forensic record of what the agent actually reported."""
    _call("old", days_ago=45)
    store.expire_transcripts()

    payload = _event_payload("old-ended")
    assert payload["turn_gaps"] == [0.9, 1.4]
    assert payload["status"] == "completed"
    assert payload["confirmed"] is True
    assert payload["seq"] == 3


def test_a_captured_value_survives_because_the_lead_keeps_it_too(store):
    """Scope. `call.captured.value` mirrors `leads.raw`, which is retained
    deliberately — it is the only thing that can be re-processed when the
    normaliser improves. Removing it here would not be a retention policy, just
    an inconsistency pointing the other way."""
    _call("old", days_ago=45)
    store.expire_transcripts()

    assert _event_payload("old-captured")["value"] == "రవి"


def test_an_event_without_a_transcript_is_not_given_one(store):
    """`jsonb_set` creates a missing key by default, so an unguarded update
    would stamp `transcript: []` onto every `call.started` in the log."""
    _call("old", days_ago=45)
    store.expire_transcripts()

    assert "transcript" not in _event_payload("old-captured")


def test_the_events_of_a_call_that_never_closed_are_still_redacted(store):
    """Keyed on the event's own `received_at`, not the lead's `ended_at`. A call
    whose agent died mid-flight has a null `ended_at` — a join on it would leave
    those transcripts in the log forever, which is exactly the population most
    likely to be forgotten."""
    _call("never-closed", days_ago=45, ended=False)
    store.expire_transcripts()

    assert _event_payload("never-closed-ended")["transcript"] == []


def test_rows_are_redacted_rather_than_deleted(store):
    """`event_id` is the deduplication key. A row that has gone would read as a
    new event if it were ever replayed, re-applying a capture and re-running a
    pipeline against a caller who hung up six weeks ago."""
    _call("old", days_ago=45)
    store.expire_transcripts()

    assert _sql("SELECT count(*) FROM localmama.call_events"
                " WHERE call_id = 'old'")[0][0] == 2


def test_the_sweep_is_idempotent(store):
    """It runs every five minutes forever. The second pass must match nothing,
    or the log reports work on every tick and the partial index never empties."""
    _call("old", days_ago=45)
    assert store.expire_transcripts() == 2      # one lead row, one event payload
    assert store.expire_transcripts() == 0


def test_retention_of_zero_keeps_everything(store):
    """Documented as "0 keeps them forever", and a deployment that means it must
    not have the event log quietly swept out from under it."""
    from backend.config import settings

    object.__setattr__(settings, "transcript_retention_days", 0)
    _call("old", days_ago=400)
    assert store.expire_transcripts() == 0
    assert _event_payload("old-ended")["transcript"] == _TRANSCRIPT


def test_a_failure_is_swallowed_rather_than_killing_the_sweep(store, monkeypatch):
    """This runs inside the outbox worker's loop. A retention error must not
    take the WhatsApp retries down with it."""
    from backend import db

    def _boom():
        raise RuntimeError("neon is away")

    monkeypatch.setattr(db, "connection", _boom)
    assert store.expire_transcripts() == 0

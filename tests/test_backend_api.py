"""The API surface, exercised as the agent will exercise it.

The store is faked — these tests are about the contract, the auth and the
acknowledgement semantics, not about SQL. What matters is that a batch the
agent could really send is accepted, that a retry is a no-op, and that the
agent is never told a lead landed when it did not.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from backend import api
from backend.config import settings
from contract import (
    CallCaptured,
    CallEnded,
    CallStarted,
    CallStatus,
    CapturedField,
    EventBatch,
    Language,
)

TOKEN = "test-token"


class FakeStore:
    """Records what the API asked it to do, and remembers event ids."""

    def __init__(self) -> None:
        self.seen: set[str] = set()
        self.calls: dict[str, dict] = {}
        self.raw: dict[str, dict] = {}
        self.processed: list[str] = []

    def record_events(self, events):
        accepted = [e.event_id for e in events if e.event_id not in self.seen]
        duplicates = [e.event_id for e in events if e.event_id in self.seen]
        self.seen.update(accepted)
        return accepted, duplicates

    def upsert_call(self, call_id, **fields):
        self.calls.setdefault(call_id, {}).update(fields)

    def merge_raw(self, call_id, field, value):
        self.raw.setdefault(call_id, {})[field.value] = value


@pytest.fixture()
def client(monkeypatch):
    store = FakeStore()
    monkeypatch.setattr(api, "store", store)
    monkeypatch.setattr(api.db, "available", lambda: True)

    async def _fake_process(call_id):
        store.processed.append(call_id)

    monkeypatch.setattr(api, "_process", _fake_process)
    object.__setattr__(settings, "agent_token", TOKEN)

    with TestClient(api.app) as c:
        c.store = store
        yield c


def _auth() -> dict:
    return {"Authorization": f"Bearer {TOKEN}"}


def _now() -> datetime:
    return datetime(2026, 7, 31, 9, 12, 4, tzinfo=timezone.utc)


def _full_call(call_id: str = "c1") -> EventBatch:
    """One complete call, as the agent would send it."""
    return EventBatch(events=[
        CallStarted(event_id="e0", call_id=call_id, seq=0, at=0.0,
                    caller_phone="+919739960092", started_at=_now()),
        CallCaptured(event_id="e1", call_id=call_id, seq=1, at=6.0,
                     field=CapturedField.LANGUAGE, value=Language.TELUGU.value),
        CallCaptured(event_id="e2", call_id=call_id, seq=2, at=12.0,
                     field=CapturedField.NAME, value="రవి"),
        CallCaptured(event_id="e3", call_id=call_id, seq=3, at=19.0,
                     field=CapturedField.SERVICE, value="ఏసీ రిపేర్"),
        CallCaptured(event_id="e4", call_id=call_id, seq=4, at=26.0,
                     field=CapturedField.CITY, value="మాదాపూర్"),
        CallEnded(event_id="e5", call_id=call_id, seq=5, at=40.0,
                  status=CallStatus.COMPLETED, confirmed=True, ended_at=_now()),
    ])


# --- auth ------------------------------------------------------------------


def test_an_unauthenticated_agent_is_refused(client):
    res = client.post("/v1/events", json=_full_call().model_dump(mode="json"))
    assert res.status_code == 401


def test_a_wrong_token_is_refused(client):
    res = client.post(
        "/v1/events",
        json=_full_call().model_dump(mode="json"),
        headers={"Authorization": "Bearer nope"},
    )
    assert res.status_code == 401


# --- the happy path --------------------------------------------------------


def test_a_whole_call_is_accepted_and_stored(client):
    res = client.post(
        "/v1/events", json=_full_call().model_dump(mode="json"), headers=_auth()
    )
    assert res.status_code == 202
    body = res.json()
    assert len(body["accepted"]) == 6
    assert body["duplicates"] == []
    assert body["rejected"] == {}


def test_values_reach_the_store_in_the_callers_own_script(client):
    """The whole point of the boundary. Nothing is normalised in transit."""
    client.post("/v1/events", json=_full_call().model_dump(mode="json"), headers=_auth())
    assert client.store.raw["c1"] == {
        "language": "telugu", "name": "రవి",
        "service": "ఏసీ రిపేర్", "city": "మాదాపూర్",
    }


def test_the_pipeline_runs_only_once_the_call_has_ended(client):
    """Processing a lead mid-call would match on half of it."""
    batch = _full_call()
    without_end = EventBatch(events=batch.events[:-1])
    client.post("/v1/events", json=without_end.model_dump(mode="json"), headers=_auth())
    assert client.store.processed == []

    client.post("/v1/events", json=batch.model_dump(mode="json"), headers=_auth())
    assert client.store.processed == ["c1"]


# --- retries ---------------------------------------------------------------


def test_a_retried_batch_is_a_no_op(client):
    """The agent may retry any event any number of times. A duplicate must not
    re-apply a capture or re-trigger the pipeline — that would send the caller
    a second WhatsApp message."""
    payload = _full_call().model_dump(mode="json")
    client.post("/v1/events", json=payload, headers=_auth())
    res = client.post("/v1/events", json=payload, headers=_auth())

    body = res.json()
    assert body["accepted"] == []
    assert len(body["duplicates"]) == 6
    assert client.store.processed == ["c1"]      # still once


# --- what the agent must never be told ------------------------------------


def test_a_malformed_event_is_rejected_not_swallowed(client):
    """422, so the agent stops retrying something that will never parse."""
    res = client.post(
        "/v1/events",
        json={"schema_version": "1.0", "events": [{"type": "call.captured"}]},
        headers=_auth(),
    )
    assert res.status_code == 422


def test_no_database_is_a_5xx_so_the_agent_retries(client, monkeypatch):
    """The one thing that must never happen is a 202 for a lead that was
    dropped — the agent would believe it had landed and stop trying."""
    monkeypatch.setattr(api.db, "available", lambda: False)
    res = client.post(
        "/v1/events", json=_full_call().model_dump(mode="json"), headers=_auth()
    )
    assert res.status_code == 503


# --- an abandoned call is still a lead ------------------------------------


def test_an_abandoned_call_is_processed_too(client):
    """A caller who gave a name and hung up is a lead — arguably the most
    interesting kind. It used to be written to an ephemeral disk and lost."""
    batch = EventBatch(events=[
        CallStarted(event_id="a0", call_id="c2", seq=0, at=0.0,
                    caller_phone="+919739960092", started_at=_now()),
        CallCaptured(event_id="a1", call_id="c2", seq=1, at=9.0,
                     field=CapturedField.NAME, value="రవి"),
        CallEnded(event_id="a2", call_id="c2", seq=2, at=15.0,
                  status=CallStatus.ABANDONED, ended_at=_now()),
    ])
    client.post("/v1/events", json=batch.model_dump(mode="json"), headers=_auth())

    assert client.store.processed == ["c2"]
    assert client.store.raw["c2"] == {"name": "రవి"}
    assert client.store.calls["c2"]["status"] == "abandoned"
    assert client.store.calls["c2"]["confirmed"] is None


# --- the vendor lookup ------------------------------------------------------


def test_the_callers_city_reaches_the_lookup(client, monkeypatch):
    """The agent has always sent the city and this endpoint used to drop it, so
    two branches of one business were an "AMBIGUOUS" reply to a caller who had
    already said the thing that separates them."""
    seen = {}

    def _find(name, limit=5, *, city=None):
        seen.update(name=name, city=city)
        return []

    monkeypatch.setattr(api.brain, "available", lambda: True)
    monkeypatch.setattr(api.brain, "looks_like_a_category", lambda q: False)
    monkeypatch.setattr(api.brain, "find_business", _find)

    res = client.get("/v1/vendors", params={"name": "Pax", "city": "Kolkata"},
                     headers=_auth())
    assert res.status_code == 200
    assert seen == {"name": "Pax", "city": "Kolkata"}


def test_a_city_in_the_callers_script_reaches_the_lookup_in_english(client, monkeypatch):
    """The catalogue is English and matching is literal, so a city still in the
    caller's script would narrow nothing — the same trip the name makes."""
    seen = {}

    def _find(name, limit=5, *, city=None):
        seen.update(name=name, city=city)
        return []

    monkeypatch.setattr(api.brain, "available", lambda: True)
    monkeypatch.setattr(api.brain, "looks_like_a_category", lambda q: False)
    monkeypatch.setattr(api.brain, "find_business", _find)

    res = client.get("/v1/vendors", params={"name": "Pax", "city": "కోల్‌కతా"},
                     headers=_auth())
    assert res.status_code == 200
    assert seen["city"].isascii() and seen["city"]


def test_no_city_is_not_an_error(client, monkeypatch):
    """Most callers name a business before they have said where they are."""
    monkeypatch.setattr(api.brain, "available", lambda: True)
    monkeypatch.setattr(api.brain, "looks_like_a_category", lambda q: False)
    monkeypatch.setattr(api.brain, "find_business",
                        lambda name, limit=5, *, city=None: [])

    res = client.get("/v1/vendors", params={"name": "Pax"}, headers=_auth())
    assert res.status_code == 200
    assert res.json()["match"] == "none"

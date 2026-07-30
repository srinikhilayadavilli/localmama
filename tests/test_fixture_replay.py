"""The fixture the README tells you to curl.

A doc that says "you can replay a lead without making a phone call" is worth
exactly as much as a test proving the fixture still parses and still flows
through the API. This is that test.
"""

from __future__ import annotations

import json
import pathlib

import pytest
from fastapi.testclient import TestClient

from backend import api
from backend.config import settings
from backend.pipeline import audit, normalise
from contract import EventBatch

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "telugu-call.json"
TOKEN = "test-token"


def test_the_fixture_matches_the_contract():
    """If the schema moves and the fixture does not, the README's example is a
    404 waiting to happen."""
    batch = EventBatch.model_validate_json(FIXTURE.read_text())
    assert [e.type for e in batch.events] == [
        "call.started", "call.captured", "call.captured",
        "call.captured", "call.captured", "call.ended",
    ]


def test_the_fixture_flows_through_the_api(monkeypatch):
    from tests.test_backend_api import FakeStore

    store = FakeStore()
    monkeypatch.setattr(api, "store", store)
    monkeypatch.setattr(api.db, "available", lambda: True)

    async def _fake_process(call_id):
        store.processed.append(call_id)

    monkeypatch.setattr(api, "_process", _fake_process)
    object.__setattr__(settings, "agent_token", TOKEN)

    with TestClient(api.app) as client:
        res = client.post(
            "/v1/events",
            json=json.loads(FIXTURE.read_text()),
            headers={"Authorization": f"Bearer {TOKEN}"},
        )

    assert res.status_code == 202
    assert len(res.json()["accepted"]) == 6
    assert store.processed == ["fixture-telugu-ac-repair"]
    # Raw, in the caller's own script, all the way to the store.
    assert store.raw["fixture-telugu-ac-repair"]["name"] == "రవి"


@pytest.mark.asyncio
async def test_the_fixture_produces_a_clean_lead():
    """End to end on the parts that need no database: a real Telugu call comes
    out as an English lead that needs no review."""
    batch = EventBatch.model_validate_json(FIXTURE.read_text())
    raw = {e.field.value: e.value for e in batch.events if e.type == "call.captured"}
    ended = batch.events[-1]
    heard = [t.text for t in ended.transcript if t.role == "caller"]

    english = await normalise(raw)
    assert english["name"] == "Ravi"
    assert english["service"] == "ac repair"
    assert "madapur" in english["city"].lower() or "madhapur" in english["city"].lower()

    _, needs_review, reason = audit(raw, heard, english=english)
    assert not needs_review, reason

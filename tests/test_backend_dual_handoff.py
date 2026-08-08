"""Two handoff channels at once, and the ways that could go wrong.

WhatsApp is what the caller actually receives. The webhook is the one being
proven. The whole point of running both is to learn whether the new one works
*without a customer paying for the answer* — so every test here is about one
channel failing to disturb the other.
"""

from __future__ import annotations

import pytest

from backend import pipeline, store


class FakeStore:
    def __init__(self, lead: dict) -> None:
        self.lead = lead
        self.by_channel: dict[str, list] = {}
        self.updates: list[dict] = []

    def get_lead(self, call_id):
        return self.lead

    def agent_for_did(self, dialled):
        return "localmama"

    def transcript_for(self, call_id):
        return self.lead.get("_heard", [])

    def upsert_call(self, call_id, **fields):
        self.updates.append(fields)

    def mark_handoff(self, call_id, channel, ok, error="", detail=None,
                     terminal=False):
        self.by_channel.setdefault(channel, []).append(
            {"ok": ok, "error": error, "detail": detail}
        )


def _lead(**over) -> dict:
    base = {
        "call_id": "c1", "caller_phone": "+919739960092", "confirmed": True,
        "name": "Ravi", "city": "Hyderabad",
        "raw": {"language": "english", "name": "Ravi", "service": "plumber",
                "city": "Hyderabad"},
        "_heard": ["Ravi", "plumber", "Hyderabad"],
    }
    base.update(over)
    return base


@pytest.fixture()
def wired(monkeypatch):
    """Both senders replaced, each independently controllable."""
    state = {"wa": {"ok": True, "messageId": "wamid.1"},
             "hook": {"ok": True, "status": 200},
             "wa_calls": 0, "hook_calls": 0}

    async def _wa(phone, **kw):
        state["wa_calls"] += 1
        out = state["wa"]
        if isinstance(out, BaseException):
            raise out
        return out

    async def _hook(lead, **kw):
        state["hook_calls"] += 1
        out = state["hook"]
        if isinstance(out, BaseException):
            raise out
        return out

    monkeypatch.setattr(pipeline.whatsapp, "send", _wa)
    monkeypatch.setattr(pipeline.webhook, "send", _hook)
    monkeypatch.setattr(pipeline.brain, "matches_for_service", lambda *a, **k: [])
    return state


async def _run(monkeypatch, store_obj):
    monkeypatch.setattr(pipeline, "store", store_obj)
    await pipeline.process("c1")


# --- both fire ------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_lead_goes_out_on_both_channels(monkeypatch, wired):
    s = FakeStore(_lead())
    await _run(monkeypatch, s)

    assert wired["wa_calls"] == 1
    assert wired["hook_calls"] == 1
    assert s.by_channel["whatsapp"][0]["ok"] is True
    assert s.by_channel["webhook"][0]["ok"] is True


@pytest.mark.asyncio
async def test_each_channel_keeps_its_own_detail(monkeypatch, wired):
    """A provider message id and an HTTP status are not the same thing and do
    not belong in the same column."""
    s = FakeStore(_lead())
    await _run(monkeypatch, s)

    assert s.by_channel["whatsapp"][0]["detail"] == "wamid.1"
    assert s.by_channel["webhook"][0]["detail"] == 200


# --- neither can take the other down --------------------------------------


@pytest.mark.asyncio
async def test_a_failing_webhook_does_not_stop_the_whatsapp(monkeypatch, wired):
    """The one that matters. The webhook points at a receiver nobody has
    exercised — it failing must cost the caller nothing."""
    wired["hook"] = {"ok": False, "error": "HTTP 500", "status": 500}
    s = FakeStore(_lead())
    await _run(monkeypatch, s)

    assert s.by_channel["whatsapp"][0]["ok"] is True
    assert s.by_channel["webhook"][0]["ok"] is False


@pytest.mark.asyncio
async def test_a_raising_webhook_does_not_stop_the_whatsapp(monkeypatch, wired):
    """Not merely a bad status — a new endpoint that hangs up mid-request or a
    client that raises. `notify` gathers with return_exceptions for this."""
    wired["hook"] = RuntimeError("connection reset")
    s = FakeStore(_lead())
    await _run(monkeypatch, s)

    assert s.by_channel["whatsapp"][0]["ok"] is True
    assert s.by_channel["webhook"][0]["ok"] is False
    assert "connection reset" in s.by_channel["webhook"][0]["error"]


@pytest.mark.asyncio
async def test_a_raising_whatsapp_does_not_stop_the_webhook(monkeypatch, wired):
    """And the same in reverse, so the experiment still gets its evidence on a
    day CampaignBot is down."""
    wired["wa"] = RuntimeError("campaignbot unreachable")
    s = FakeStore(_lead())
    await _run(monkeypatch, s)

    assert s.by_channel["whatsapp"][0]["ok"] is False
    assert s.by_channel["webhook"][0]["ok"] is True


@pytest.mark.asyncio
async def test_a_lead_with_no_phone_closes_both(monkeypatch, wired):
    s = FakeStore(_lead(caller_phone=None))
    await _run(monkeypatch, s)

    assert wired["wa_calls"] == 0
    assert wired["hook_calls"] == 0
    assert s.by_channel["whatsapp"][0]["error"] == "no phone"
    assert s.by_channel["webhook"][0]["error"] == "no phone"


# --- the columns stay apart ------------------------------------------------


def test_the_channels_own_different_columns():
    """The reason a WhatsApp success is never re-sent to fix a webhook failure.
    If these ever overlap, the sweep messages a customer twice."""
    wa = store._CHANNELS["whatsapp"]
    hook = store._CHANNELS["webhook"]

    assert set(wa.values()).isdisjoint(set(hook.values()))
    assert wa["status"] == "whatsapp_status"
    assert hook["status"] == "handoff_status"


def test_an_unknown_channel_is_refused_not_interpolated():
    """These column names go into SQL by f-string. The whitelist is the guard."""
    with pytest.raises(ValueError):
        store._cols("'; DROP TABLE localmama.leads; --")


# --- what the review found -------------------------------------------------


@pytest.mark.asyncio
async def test_the_subject_is_stored_not_re_derived(monkeypatch, wired):
    """The sweep used to rebuild it as `service_said or service`, which is a
    different decision to the one `_what_the_message_is_about` made. Storing it
    is what stops a retry naming a trade the live path refused to name."""
    s = FakeStore(_lead(
        raw={"language": "english", "name": "Ravi"},
        _heard=["Ravi"],
        asked_vendors=[{"title": "Brimmies Cafe", "phone": "+919876543210"}],
    ))
    await _run(monkeypatch, s)

    written = [u for u in s.updates if "handoff_subject" in u]
    assert written and written[-1]["handoff_subject"] == "Brimmies Cafe"


@pytest.mark.asyncio
async def test_an_unverified_service_still_reaches_the_webhook(monkeypatch, wired):
    """The copy guard is about what we say to the *caller*. Your own system
    receives the review verdict with the lead and can triage it — suppressing
    both would hide exactly the leads that most need a human."""
    s = FakeStore(_lead(
        confirmed=None,
        raw={"language": "tamil", "name": "பணிகுமார்",
             "service": "இலெக்ட்ரீஷியன்", "city": "சென்னை"},
        _heard=["என்னோட பெயர் பணிகுமார்", "సరఫరా చేయనో"],
    ))
    await _run(monkeypatch, s)

    assert s.by_channel["whatsapp"][0]["ok"] is False
    assert s.by_channel["webhook"][0]["ok"] is True


@pytest.mark.asyncio
async def test_a_terminal_rejection_is_passed_through(monkeypatch, wired):
    """A 4xx means the receiver refuses this lead, not that it failed to
    receive it. Without the flag it was recorded `pending` and re-POSTed 25
    times before being dropped past the attempt ceiling."""
    wired["hook"] = {"ok": False, "error": "HTTP 400", "status": 400,
                     "terminal": True}
    s = FakeStore(_lead())
    marks = []
    s.mark_handoff = lambda cid, ch, ok, error="", detail=None, terminal=False: (
        marks.append((ch, terminal))
    )
    await _run(monkeypatch, s)

    assert ("webhook", True) in marks
    assert ("whatsapp", False) in marks

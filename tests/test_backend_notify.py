"""When a caller gets a WhatsApp message, and when they must not.

Written because a real phone received:

    Hi there,
    As Requested, here are some the service options in your area:

That was a caller who picked a language and hung up. The lead is worth keeping;
the automatic message had nothing to say.
"""

from __future__ import annotations

import pytest

from backend import pipeline


class FakeStore:
    def __init__(self, lead: dict) -> None:
        self.lead = lead
        self.marks: list[tuple] = []
        self.updates: list[dict] = []

    def get_lead(self, call_id):
        return self.lead

    def transcript_for(self, call_id):
        return self.lead.get("_heard", [])

    def upsert_call(self, call_id, **fields):
        self.updates.append(fields)

    def mark_whatsapp(self, call_id, ok, error="", message_id=""):
        self.marks.append((ok, error))


@pytest.fixture()
def sent(monkeypatch):
    """Records every WhatsApp send the pipeline attempts."""
    calls = []

    async def _send(phone, **kw):
        calls.append({"phone": phone, **kw})
        return {"ok": True}

    monkeypatch.setattr(pipeline.whatsapp, "send", _send)
    monkeypatch.setattr(pipeline.brain, "matches_for_service", lambda *a, **k: [])
    return calls


def _lead(**over) -> dict:
    base = {
        "call_id": "c1", "caller_phone": "+919739960092", "confirmed": True,
        "raw": {}, "_heard": [],
    }
    base.update(over)
    return base


@pytest.mark.asyncio
async def test_a_language_only_call_sends_nothing(monkeypatch, sent):
    """The bug. `raw` is truthy — it holds the language — so this sails past
    the empty-lead guard and used to message the caller about "the service"."""
    store = FakeStore(_lead(raw={"language": "telugu"}))
    monkeypatch.setattr(pipeline, "store", store)

    await pipeline.process("c1")

    assert sent == []
    assert store.marks == [(False, "incomplete lead")]


@pytest.mark.asyncio
async def test_a_name_only_call_sends_nothing(monkeypatch, sent):
    """A caller who gave their name and hung up. Still a lead worth chasing,
    still nothing an automatic message can usefully say."""
    store = FakeStore(_lead(raw={"language": "telugu", "name": "Ravi"}))
    monkeypatch.setattr(pipeline, "store", store)

    await pipeline.process("c1")
    assert sent == []


@pytest.mark.asyncio
async def test_a_lead_with_a_service_is_sent(monkeypatch, sent):
    """The service is what the message is about, so it is the bar."""
    store = FakeStore(_lead(
        raw={"language": "telugu", "name": "Ravi", "service": "plumber",
             "city": "Hyderabad"},
        _heard=["my name is Ravi", "plumber", "Hyderabad"],
    ))
    monkeypatch.setattr(pipeline, "store", store)

    await pipeline.process("c1")

    assert len(sent) == 1
    assert sent[0]["service"] == "plumber"
    assert sent[0]["phone"] == "+919739960092"


@pytest.mark.asyncio
async def test_a_call_that_captured_nothing_sends_nothing(monkeypatch, sent):
    store = FakeStore(_lead(raw={}))
    monkeypatch.setattr(pipeline, "store", store)

    await pipeline.process("c1")
    assert sent == []


@pytest.mark.asyncio
async def test_an_unverifiable_service_is_not_messaged_about(monkeypatch, sent):
    """The call that prompted this. The caller asked for a plumber; the line
    was bad enough that STT produced Telugu on a Tamil call and the model heard
    "electrician". The audit scored it 0.00 and flagged the lead — and the
    message went out anyway, telling them about electricians.

    Naming the wrong trade to a customer is worse than saying nothing."""
    store = FakeStore(_lead(
        confirmed=None,                     # the read-back never got an answer
        raw={"language": "tamil", "name": "பணிகுமார்",
             "service": "இலெக்ட்ரீஷியன்", "city": "சென்னை"},
        _heard=["என்னோட பெயர் பணிகுமார்", "సరఫరా చేయనో", "பெனமெய்ல்"],
    ))
    monkeypatch.setattr(pipeline, "store", store)

    await pipeline.process("c1")

    assert sent == []
    assert store.marks == [(False, "service unverified")]


@pytest.mark.asyncio
async def test_a_confirmed_read_back_beats_a_garbled_transcript(monkeypatch, sent):
    """This suppressed two good leads. A caller said "plumber" and the
    transcriber wrote "अप्पिरंबर"; another said "సలూన్ కాడ్ బ్యూటీ" and the model
    heard "beautician". Both times the model was right and the transcript was
    garbage — and the audit, which can only see that the two disagree, threw
    the lead away.

    A read-back the caller answered is better evidence than a second decode of
    the same bad audio."""
    store = FakeStore(_lead(
        confirmed=True,                     # they heard it and said yes
        raw={"language": "english", "name": "Sree Nikhila",
             "service": "plumber", "city": "Mangalore"},
        _heard=["English", "My name is Sree Nikhila", "अप्पिरंबर", "Mangalore"],
    ))
    monkeypatch.setattr(pipeline, "store", store)

    class H:
        title, phone, category, city = "Infinity Enterprises", "+91", "Plumbing", None

    monkeypatch.setattr(pipeline.brain, "matches_for_service", lambda q, **k: [H()])

    await pipeline.process("c1")

    assert len(sent) == 1
    assert "Infinity Enterprises" in sent[0]["options"]


@pytest.mark.asyncio
async def test_a_shaky_name_still_gets_a_message(monkeypatch, sent):
    """Only the service gates the send. A name we cannot verify still produces
    a useful message — the caller knows who they are."""
    store = FakeStore(_lead(
        raw={"language": "telugu", "name": "Suresh", "service": "plumber",
             "city": "Hyderabad"},
        _heard=["my name is Ravi", "plumber", "Hyderabad"],
    ))
    monkeypatch.setattr(pipeline, "store", store)

    await pipeline.process("c1")
    assert len(sent) == 1


# --- businesses the caller asked for by name ------------------------------


@pytest.mark.asyncio
async def test_a_business_asked_for_by_name_leads_the_message(monkeypatch, sent):
    """They named it. Whatever we matched off the service is a suggestion
    beside that, so it comes second."""
    store = FakeStore(_lead(
        raw={"language": "english", "name": "Ravi", "service": "plumber",
             "city": "Hyderabad"},
        _heard=["Ravi", "plumber", "Hyderabad"],
        asked_vendors=[{"title": "Brimmies Cafe", "phone": "+919876543210"}],
    ))
    monkeypatch.setattr(pipeline, "store", store)
    monkeypatch.setattr(pipeline.brain, "matches_for_service", lambda *a, **k: [])

    await pipeline.process("c1")

    assert len(sent) == 1
    assert "Brimmies Cafe" in sent[0]["options"]
    assert sent[0]["service"] == "plumber"      # the service is still the subject


@pytest.mark.asyncio
async def test_a_business_carries_the_message_with_no_service(monkeypatch, sent):
    """Someone who only wants a number may never answer the service question.
    Refusing to write down the one thing they asked for is backwards, so the
    business name becomes what the message is about."""
    store = FakeStore(_lead(
        raw={"language": "english", "name": "Ravi"},
        _heard=["Ravi"],
        asked_vendors=[{"title": "Brimmies Cafe", "phone": "+919876543210"}],
    ))
    monkeypatch.setattr(pipeline, "store", store)

    await pipeline.process("c1")

    assert len(sent) == 1
    assert sent[0]["service"] == "Brimmies Cafe"
    assert "Brimmies Cafe" in sent[0]["options"]


@pytest.mark.asyncio
async def test_a_business_survives_an_unverifiable_service(monkeypatch, sent):
    """The business matched the catalogue by name, which is evidence
    independent of whatever the service decoded to."""
    store = FakeStore(_lead(
        confirmed=None,
        raw={"language": "tamil", "name": "பணிகுமார்",
             "service": "இலெக்ட்ரீஷியன்", "city": "சென்னை"},
        _heard=["என்னோட பெயர் பணிகுமார்", "సరఫరా చేయనో"],
        asked_vendors=[{"title": "Clean Mates", "phone": "+919876543210"}],
    ))
    monkeypatch.setattr(pipeline, "store", store)

    await pipeline.process("c1")

    assert len(sent) == 1
    assert sent[0]["service"] == "Clean Mates"


@pytest.mark.asyncio
async def test_a_business_with_no_phone_is_not_listed(monkeypatch, sent):
    """A name without a number is a teaser, not an answer."""
    store = FakeStore(_lead(
        raw={"language": "english", "name": "Ravi"},
        _heard=["Ravi"],
        asked_vendors=[{"title": "Brimmies Cafe", "phone": None}],
    ))
    monkeypatch.setattr(pipeline, "store", store)

    await pipeline.process("c1")
    assert sent == []


@pytest.mark.asyncio
async def test_the_message_says_what_the_caller_said(monkeypatch, sent):
    """The template's {{2}}. A caller asked for a "beauty parlor" and the
    message read "here are some salon options" — matched correctly, worded as
    though nobody had listened."""
    store = FakeStore(_lead(
        raw={"language": "english", "name": "Raja", "service": "beauty parlor",
             "city": "Kolkata"},
        _heard=["Raja", "beauty parlor", "Kolkata"],
    ))
    monkeypatch.setattr(pipeline, "store", store)

    await pipeline.process("c1")

    assert len(sent) == 1
    assert "parlo" in sent[0]["service"].lower()
    assert sent[0]["service"].lower() != "salon"
    # …while the lead still stores the canonical label, which is what matched.
    assert store.updates[-1]["service"] == "salon"


# --- when the caller described a problem rather than naming a trade --------


@pytest.mark.asyncio
async def test_the_understood_trade_is_used_when_their_words_match_nothing(
    monkeypatch, sent
):
    """Callers describe: "my geyser is not working" reached a co-working space,
    "my bike is making noise" reached a bike rental. The agent heard the call,
    read "plumber" out of it, and said that back during the read-back — so the
    guess was agreed to while they could still object."""
    store = FakeStore(_lead(
        raw={"language": "english", "name": "Raja",
             "service": "my geyser is not working", "city": "Kolkata"},
        _heard=["Raja", "my geyser is not working", "Kolkata"],
        service_inferred="plumber",
    ))
    monkeypatch.setattr(pipeline, "store", store)

    def _match(q, **kw):
        class H:
            title, phone, category, city = "Infinity Enterprises", "+91", "Plumbing", None
        return [H()] if "plumb" in q.lower() else []

    monkeypatch.setattr(pipeline.brain, "matches_for_service", _match)

    await pipeline.process("c1")

    assert len(sent) == 1
    assert "Infinity Enterprises" in sent[0]["options"]
    # …and flagged, because nobody actually said "plumber".
    assert store.updates[-1]["needs_review"] is True
    assert "understood as" in store.updates[-1]["review_reason"]


@pytest.mark.asyncio
async def test_their_own_words_beat_the_interpretation(monkeypatch, sent):
    """A literal hit on what they said is better evidence than a reading of it,
    so the interpretation is only ever a fallback."""
    store = FakeStore(_lead(
        raw={"language": "english", "name": "Raja", "service": "plumber",
             "city": "Kolkata"},
        _heard=["Raja", "plumber", "Kolkata"],
        service_inferred="salon",
    ))
    monkeypatch.setattr(pipeline, "store", store)

    class H:
        title, phone, category, city = "Infinity Enterprises", "+91", "Plumbing", None

    monkeypatch.setattr(pipeline.brain, "matches_for_service", lambda q, **k: [H()])

    await pipeline.process("c1")

    assert len(sent) == 1
    # Not flagged: the match came from their own word, not the reading.
    assert "understood as" not in (store.updates[-1].get("review_reason") or "")

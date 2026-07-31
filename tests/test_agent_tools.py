"""The agent's tools: local writes and queued events, nothing else.

The property under test throughout is that no tool touches the network or
normalises a value. If one ever starts to, these break — which is the point,
because that is how six seconds of Sarvam calls got onto the call path last
time.
"""

from __future__ import annotations

import pytest

from agent.tools import Recorder
from contract import CallCaptured, CallStatus, CapturedField, Language


class FakeQueue:
    """Stands in for the event queue; records what would have been sent."""

    def __init__(self) -> None:
        self.sent: list = []

    def send(self, event) -> None:
        self.sent.append(event)

    def start(self) -> None:
        pass


@pytest.fixture()
def rec():
    return Recorder(FakeQueue(), caller_phone="+919739960092")


def tools(rec) -> dict:
    return {t.info.name: t for t in rec.build_tools()}


def captures(rec) -> list[CallCaptured]:
    return [e for e in rec.events.sent if isinstance(e, CallCaptured)]


# --- values leave raw ------------------------------------------------------


@pytest.mark.asyncio
async def test_a_name_is_stored_and_sent_in_the_callers_own_script(rec):
    """No transliteration here. That is the backend's job, and doing it here
    cost up to two seconds of a live call per field."""
    await tools(rec)["set_name"](name="రవి")
    assert rec.session.captured[CapturedField.NAME] == "రవి"
    assert captures(rec)[0].value == "రవి"


@pytest.mark.asyncio
async def test_a_service_is_not_translated_or_canonicalised(rec):
    await tools(rec)["set_service"](service="నాకు ఎలక్ట్రీషియన్ కావాలి")
    assert captures(rec)[0].value == "నాకు ఎలక్ట్రీషియన్ కావాలి"


@pytest.mark.asyncio
async def test_an_empty_value_is_refused_without_an_event(rec):
    reply = await tools(rec)["set_name"](name="   ")
    assert "again" in reply.lower()
    assert captures(rec) == []


# --- the evidence the backend audits against -------------------------------


@pytest.mark.asyncio
async def test_the_callers_own_turns_are_kept_as_evidence(rec):
    """Shipped with `call.ended`. An independent decode of the same audio the
    model heard, and the only evidence the backend's audit has that a captured
    value came from the caller."""
    rec.note_caller_speech("मेरा नाम राहुल है")
    rec.note_caller_speech("  ")            # empty finals are not turns
    rec.note_caller_speech("हाँ")
    assert [t.text for t in rec.session.transcript] == ["मेरा नाम राहुल है", "हाँ"]
    assert all(t.role == "caller" for t in rec.session.transcript)


@pytest.mark.asyncio
async def test_a_corrected_value_is_marked_as_such(rec):
    """A field the caller fixed during the read-back is better evidence, not
    worse, and the backend scores it accordingly."""
    t = tools(rec)
    await t["set_name"](name="Ramesh")
    await t["set_name"](name="Ravi")
    assert [c.corrected for c in captures(rec)] == [False, True]


# --- completeness is enforced locally, with no network ---------------------


@pytest.mark.asyncio
async def test_save_refuses_while_a_field_is_missing(rec):
    t = tools(rec)
    await t["set_language"](language="telugu")
    await t["set_name"](name="రవి")
    reply = await t["save_lead"]()
    assert "missing" in reply.lower()
    assert "service" in reply and "city" in reply
    assert rec.session.saved is False


async def _capture_everything(rec) -> dict:
    t = tools(rec)
    await t["set_language"](language="telugu")
    await t["set_name"](name="రవి")
    await t["set_service"](service="ఏసీ రిపేర్")
    await t["set_city"](city="మాదాపూర్")
    return t


@pytest.mark.asyncio
async def test_a_confirmed_call_records_that_the_caller_answered(rec):
    """The read-back happened and the caller replied to it."""
    t = await _capture_everything(rec)
    rec.note_agent_speech("So that's an AC repair in Madhapur for Ravi?")
    rec.note_caller_speech("అవును")                      # "yes"

    reply = await t["save_lead"]()
    assert "saved" in reply.lower()
    assert rec.session.saved is True
    assert rec.session.confirmed is True


@pytest.mark.asyncio
async def test_saving_without_the_caller_answering_is_not_a_confirmation(rec):
    """A real call: Mami read the details back and called save_lead two seconds
    later, with no caller turn in between. The lead went out marked confirmed,
    having been confirmed by nobody — and the details were wrong.

    Reaching this tool is not evidence. A caller turn after it is."""
    t = await _capture_everything(rec)
    rec.note_agent_speech("So that's an AC repair in Madhapur for Ravi?")

    await t["save_lead"]()
    assert rec.session.saved is True
    assert rec.session.confirmed is False


@pytest.mark.asyncio
async def test_saving_twice_is_a_no_op(rec):
    """The model re-confirms at the end of a call, or retries after a slow
    response. Doing this twice would send the caller two WhatsApp messages."""
    t = tools(rec)
    for tool, value in (("set_language", "telugu"), ("set_name", "రవి"),
                        ("set_service", "ఏసీ"), ("set_city", "మాదాపూర్")):
        await t[tool](**{tool.removeprefix("set_"): value})

    await t["save_lead"]()
    reply = await t["save_lead"]()
    assert "already saved" in reply.lower()


# --- language, which is the one thing the agent does resolve ---------------


@pytest.mark.asyncio
async def test_a_language_is_resolved_before_it_is_sent(rec):
    """The agent has to resolve this one: it needs to know what to speak next."""
    await tools(rec)["set_language"](language="తెలుగు")
    assert rec.session.language is Language.TELUGU
    assert captures(rec)[0].value == "telugu"


@pytest.mark.asyncio
async def test_an_unsupported_language_is_refused(rec):
    reply = await tools(rec)["set_language"](language="french")
    assert "not a supported language" in reply
    assert captures(rec) == []


@pytest.mark.asyncio
async def test_switching_language_mid_call_has_to_be_asked_for_twice(rec):
    """A caller said "ఏది?" — Telugu for "which?" — and the model switched the
    whole conversation to Hindi off that one word. A genuine request survives
    being asked to confirm; a misheard one does not."""
    t = tools(rec)
    await t["set_language"](language="telugu")

    first = await t["set_language"](language="hindi")
    assert "Do NOT switch yet" in first
    assert rec.session.language is Language.TELUGU

    await t["set_language"](language="hindi")
    assert rec.session.language is Language.HINDI


# --- what the state line tells the model -----------------------------------


@pytest.mark.asyncio
async def test_every_reply_restates_what_is_held_and_what_is_needed(rec):
    """A reply naming only the field just written made the model re-anchor and
    re-ask for details it already had."""
    t = tools(rec)
    await t["set_language"](language="telugu")
    reply = await t["set_name"](name="రవి")
    assert "HELD:" in reply and "name=రవి" in reply
    assert "STILL NEEDED:" in reply and "service" in reply


# --- the call record -------------------------------------------------------


def test_an_abandoned_call_still_closes_the_record(rec):
    rec.announce_start()
    rec.announce_end(CallStatus.ABANDONED)
    ended = rec.events.sent[-1]
    assert ended.status is CallStatus.ABANDONED
    # No read-back happened, which is not the same as the caller disagreeing.
    assert ended.confirmed is None


# --- the order is a default, not a gate -----------------------------------


@pytest.mark.asyncio
async def test_a_service_can_be_recorded_before_anything_else(rec):
    """A caller said "I need a plumber" 3.9 seconds in, before the greeting had
    finished. The flow threw it away and asked again 38 seconds later, on a
    moment of the line so bad that the transcriber emitted Telugu on a Tamil
    call and the model heard "electrician".

    The prompt now tells the model to record what it hears when it hears it.
    Nothing in the tools may stand in the way of that."""
    t = tools(rec)
    reply = await t["set_service"](service="plumber")

    assert rec.session.captured[CapturedField.SERVICE] == "plumber"
    assert "service=plumber" in reply
    # And the model is told it no longer needs to ask.
    assert "STILL NEEDED: language, name, city" in reply


@pytest.mark.asyncio
async def test_an_out_of_order_call_still_saves(rec):
    """Captured backwards, and complete is still complete."""
    t = tools(rec)
    await t["set_city"](city="Chennai")
    await t["set_service"](service="plumber")
    await t["set_name"](name="Panikumar")
    await t["set_language"](language="tamil")
    rec.note_agent_speech("So that's a plumber in Chennai for Panikumar?")
    rec.note_caller_speech("ஆமா")

    assert "saved" in (await t["save_lead"]()).lower()
    assert rec.session.confirmed is True


# --- a business asked for stands in for the service ------------------------


class _Reply:
    def __init__(self, match, title=None, phone=None):
        self.match, self.title, self.phone = match, title, phone
        self.say, self.instruction = f"{title} — 98363 18445", "Read it clearly."


@pytest.mark.asyncio
async def test_a_found_business_removes_the_service_question(rec, monkeypatch):
    """A caller rang for Pax Jwellers' number, got it, and was then asked what
    trade they needed — a question they had already answered by naming the
    business. save_lead required a service, so the model had to keep asking."""
    from agent import tools as tools_mod
    from contract import MatchKind

    async def _found(*a, **k):
        return _Reply(MatchKind.EXACT, "Pax Jwellers", "9836318445")

    monkeypatch.setattr(tools_mod, "lookup_vendor", _found)
    t = tools(rec)
    await t["set_language"](language="hindi")
    await t["set_name"](name="Ravi")

    assert CapturedField.SERVICE in rec.session.missing()
    reply = await t["lookup_vendor_contact"](business="Pax")
    assert CapturedField.SERVICE not in rec.session.missing()
    assert "Do NOT ask what service" in reply
    assert "asked about=Pax Jwellers" in reply

    await t["set_city"](city="Kolkata")
    rec.note_agent_speech("So that's Pax Jwellers in Kolkata for Ravi?")
    rec.note_caller_speech("haan")
    assert "saved" in (await t["save_lead"]()).lower()


@pytest.mark.asyncio
async def test_a_weak_lookup_does_not_stand_in_for_the_service(rec, monkeypatch):
    """An approximate hit is still a question, and a category names no
    business. Neither is what the caller asked for."""
    from agent import tools as tools_mod
    from contract import MatchKind

    async def _vague(*a, **k):
        return _Reply(MatchKind.APPROXIMATE, "Pax Jwellers")

    monkeypatch.setattr(tools_mod, "lookup_vendor", _vague)
    t = tools(rec)
    await t["lookup_vendor_contact"](business="Pax Jewellers")
    assert CapturedField.SERVICE in rec.session.missing()

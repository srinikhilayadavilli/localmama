"""A value only counts if the caller was heard saying it.

The bug this exists for: on a Hindi call the lead came back with a name nobody
had spoken. On the speech-to-speech path the model is the transcriber, and
`set_name` took whatever string it passed — a realtime model on a bad line does
not fail loudly, it produces a fluent, plausible Indian name.

So every stored value is checked against the transcript of the caller's own
audio, which comes from a different model than the one hearing the call. Two
independent decodes agree on a name that was said and disagree on one that was
invented.

**The check runs at save, and the ordering is why.** The first version of this
checked each value as the tool received it, and it made the agent unusable: the
realtime model calls `set_name` the moment the caller answers, while the
transcript is produced by a separate, slower pass that lands *afterwards*. Every
value was therefore compared against the previous turn, matched nothing, and was
refused — the caller heard "sorry, I could not understand" on every question.

The tests below encode that ordering: `answers()` calls the tool BEFORE feeding
the transcript, exactly as production does. The original tests fed the
transcript first, which is why they passed while live calls failed.
"""

from __future__ import annotations

import pytest

from backend.app.config import settings
from backend.app.realtime_tools import LeadRecorder
from backend.app.services import grounding
from backend.app.services.grounding import Verdict


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(type(settings), "data_dir", property(lambda self: tmp_path),
                        raising=False)
    from backend.app import persistence
    monkeypatch.setattr(persistence.settings.__class__, "data_dir",
                        property(lambda self: tmp_path), raising=False)
    return tmp_path


def tools(rec: LeadRecorder) -> dict:
    return {t.info.name: t for t in rec.build_tools()}


def heard(rec: LeadRecorder, *turns: str) -> LeadRecorder:
    for turn in turns:
        rec.note_caller_speech(turn)
    return rec


async def answers(rec: LeadRecorder, tool, kwargs: dict, transcript: str) -> str:
    """One caller turn, in the order production actually produces it.

    The tool call comes first and the transcript lands after it — that is the
    sequence, not an edge case, and anything that judges a value inside the tool
    is judging it against the previous turn.
    """
    reply = await tool(**kwargs)
    rec.note_caller_speech(transcript)
    return reply


# --- the check itself -----------------------------------------------------


def test_a_name_the_caller_said_is_heard():
    assert grounding.check("राहुल", ["मेरा नाम राहुल है"]) is Verdict.HEARD


def test_a_name_the_caller_did_not_say_is_not():
    assert grounding.check("Suresh", ["मेरा नाम राहुल है"]) is Verdict.NOT_HEARD


def test_the_comparison_sees_through_script():
    """The realtime model hands over "Rahul"; the transcript reads "राहुल".
    Comparing those literally rejects the caller's own name on every Hindi
    call."""
    assert grounding.check("Rahul", ["मेरा नाम राहुल है"]) is Verdict.HEARD
    assert grounding.check("राहुल", ["mera naam Rahul hai"]) is Verdict.HEARD


def test_two_spellings_of_the_same_name_still_match():
    """Two decoders of the same phone audio do not spell a name identically."""
    assert grounding.check("Rahool", ["my name is Rahul"]) is Verdict.HEARD
    assert grounding.check("Lakshmi", ["my name is Laxmi"]) is Verdict.HEARD


def test_every_word_of_a_name_must_be_heard():
    """Half a name invented is a whole name wrong."""
    assert grounding.check("Ravi Kumar", ["I am Ravi"]) is Verdict.NOT_HEARD
    assert grounding.check("Ravi Kumar", ["I am Ravi Kumar"]) is Verdict.HEARD


def test_a_service_needs_only_one_content_word():
    """A description may carry words the caller never said — "plumber service"
    for "मुझे प्लंबर चाहिए" — as long as it is anchored in one they did."""
    assert grounding.check(
        "plumber service", ["मुझे प्लंबर चाहिए"], every_word=False
    ) is Verdict.HEARD
    assert grounding.check(
        "plumber service", ["मुझे प्लंबर चाहिए"], every_word=True
    ) is Verdict.NOT_HEARD


def test_a_service_out_of_nowhere_is_still_refused():
    assert grounding.check(
        "plumber", ["मुझे एसी ठीक करवाना है"], every_word=False
    ) is Verdict.NOT_HEARD


def test_nothing_transcribed_is_not_evidence_of_anything():
    """A provider with transcription off must not have every value refused."""
    assert grounding.check("Ravi", []) is Verdict.NO_TRANSCRIPT
    assert grounding.check("Ravi", ["   "]) is Verdict.NO_TRANSCRIPT


# --- the regression: transcripts arrive after the tool call ---------------


@pytest.mark.asyncio
async def test_a_whole_call_works_when_transcripts_lag_the_tools():
    """The outage, as a test.

    Every value here is recorded before its transcript exists, which is the
    order production produces. Checking inside the tool refused all four and the
    caller was asked to repeat themselves until they hung up.
    """
    rec = LeadRecorder()
    t = tools(rec)

    await answers(rec, t["set_language"], {"language": "हिंदी"}, "हिंदी")
    await answers(rec, t["set_name"], {"name": "राहुल"}, "मेरा नाम राहुल है")
    await answers(rec, t["set_service"], {"service": "प्लंबर"}, "मुझे प्लंबर चाहिए")
    await answers(rec, t["set_city"], {"city": "मुंबई"}, "मैं मुंबई में हूँ")
    rec.note_caller_speech("हाँ, सही है")          # the read-back

    assert rec.snapshot() == {
        "language": "hindi", "name": "Rahul", "service": "plumber", "city": "Mumbai",
    }
    assert "saved" in (await t["save_lead"]()).lower()


@pytest.mark.asyncio
async def test_a_value_is_never_refused_at_capture():
    """Capture is unconditional now. Whatever the model passes is recorded and
    judged later, because at this instant there is nothing to judge it against."""
    rec = LeadRecorder()
    t = tools(rec)
    reply = await t["set_name"](name="राहुल")
    assert rec.session.user_name == "Rahul"
    assert "did not hear" not in reply.lower()


# --- the audit, at save ---------------------------------------------------


@pytest.mark.asyncio
async def test_an_invented_name_is_caught_before_the_lead_is_saved():
    rec = LeadRecorder()
    t = tools(rec)
    await answers(rec, t["set_language"], {"language": "हिंदी"}, "हिंदी")
    await answers(rec, t["set_name"], {"name": "Priya Sharma"}, "मेरा नाम राहुल है")
    await answers(rec, t["set_service"], {"service": "प्लंबर"}, "मुझे प्लंबर चाहिए")
    await answers(rec, t["set_city"], {"city": "मुंबई"}, "मैं मुंबई में हूँ")
    rec.note_caller_speech("हाँ")

    reply = await t["save_lead"]()
    assert rec.saved is False
    assert "name" in reply.lower()
    assert rec.session.user_name is None, "the bad value must be cleared, not kept"
    # And only that one: a refusal is a re-ask, not a reset.
    assert rec.session.requested_service == "plumber"
    assert rec.session.city_or_area == "Mumbai"


@pytest.mark.asyncio
async def test_the_call_completes_once_the_caller_repeats_it():
    rec = LeadRecorder()
    t = tools(rec)
    await answers(rec, t["set_language"], {"language": "हिंदी"}, "हिंदी")
    await answers(rec, t["set_name"], {"name": "Priya Sharma"}, "मेरा नाम राहुल है")
    await answers(rec, t["set_service"], {"service": "प्लंबर"}, "मुझे प्लंबर चाहिए")
    await answers(rec, t["set_city"], {"city": "मुंबई"}, "मैं मुंबई में हूँ")
    rec.note_caller_speech("हाँ")
    await t["save_lead"]()

    await answers(rec, t["set_name"], {"name": "राहुल"}, "राहुल")
    assert "saved" in (await t["save_lead"]()).lower()
    assert rec.session.user_name == "Rahul"


@pytest.mark.asyncio
async def test_an_invented_city_is_caught_too():
    rec = LeadRecorder()
    t = tools(rec)
    await answers(rec, t["set_language"], {"language": "हिंदी"}, "हिंदी")
    await answers(rec, t["set_name"], {"name": "राहुल"}, "मेरा नाम राहुल है")
    await answers(rec, t["set_service"], {"service": "प्लंबर"}, "मुझे प्लंबर चाहिए")
    await answers(rec, t["set_city"], {"city": "Delhi"}, "मैं मुंबई में हूँ")
    rec.note_caller_speech("हाँ")

    assert "city" in (await t["save_lead"]()).lower()
    assert rec.session.city_or_area is None


@pytest.mark.asyncio
async def test_a_field_with_no_transcript_yet_is_not_judged():
    """Absence of evidence is not evidence. A model that saves the instant the
    last field is captured — no read-back, no further turn — must not have that
    field refused for a transcript that simply has not landed."""
    rec = LeadRecorder()
    t = tools(rec)
    await answers(rec, t["set_language"], {"language": "हिंदी"}, "हिंदी")
    await answers(rec, t["set_name"], {"name": "राहुल"}, "मेरा नाम राहुल है")
    await answers(rec, t["set_service"], {"service": "प्लंबर"}, "मुझे प्लंबर चाहिए")
    await t["set_city"](city="Delhi")          # transcript never arrives

    assert "saved" in (await t["save_lead"]()).lower()
    assert rec.session.city_or_area == "Delhi"


@pytest.mark.asyncio
async def test_the_audit_gives_up_rather_than_trapping_a_caller():
    """The audit compares two imperfect decodes of a phone line. When it is
    wrong, a caller must not be asked for their name forever — it loses ties."""
    from backend.app.realtime_tools import MAX_AUDIT_REFUSALS

    rec = LeadRecorder()
    t = tools(rec)
    await answers(rec, t["set_language"], {"language": "हिंदी"}, "हिंदी")
    await answers(rec, t["set_service"], {"service": "प्लंबर"}, "मुझे प्लंबर चाहिए")
    await answers(rec, t["set_city"], {"city": "मुंबई"}, "मैं मुंबई में हूँ")

    for _ in range(MAX_AUDIT_REFUSALS):
        await answers(rec, t["set_name"], {"name": "Priya"}, "मेरा नाम राहुल है")
        assert "not what the caller said" in (await t["save_lead"]()).lower()

    await answers(rec, t["set_name"], {"name": "Priya"}, "मेरा नाम राहुल है")
    assert "saved" in (await t["save_lead"]()).lower(), "a caller must not be trapped"


@pytest.mark.asyncio
async def test_a_service_the_rules_reach_independently_survives_the_audit():
    """"AC repair" shares no word with "मुझे एसी ठीक करवाना है", but the
    catalogue reaches `ac repair` from the caller's own words too."""
    rec = LeadRecorder()
    t = tools(rec)
    await answers(rec, t["set_language"], {"language": "हिंदी"}, "हिंदी")
    await answers(rec, t["set_name"], {"name": "राहुल"}, "मेरा नाम राहुल है")
    await answers(
        rec, t["set_service"], {"service": "AC repair"}, "मुझे एसी ठीक करवाना है"
    )
    await answers(rec, t["set_city"], {"city": "मुंबई"}, "मैं मुंबई में हूँ")
    rec.note_caller_speech("हाँ")

    assert "saved" in (await t["save_lead"]()).lower()
    assert rec.session.requested_service == "ac repair"


@pytest.mark.asyncio
async def test_a_service_from_nowhere_is_caught():
    rec = LeadRecorder()
    t = tools(rec)
    await answers(rec, t["set_language"], {"language": "हिंदी"}, "हिंदी")
    await answers(rec, t["set_name"], {"name": "राहुल"}, "मेरा नाम राहुल है")
    await answers(rec, t["set_service"], {"service": "tutor"}, "मुझे प्लंबर चाहिए")
    await answers(rec, t["set_city"], {"city": "मुंबई"}, "मैं मुंबई में हूँ")
    rec.note_caller_speech("हाँ")

    assert "service" in (await t["save_lead"]()).lower()
    assert rec.session.requested_service is None


@pytest.mark.asyncio
async def test_a_call_with_no_transcript_at_all_still_saves():
    """Transcription is a second opinion, not a dependency."""
    rec = LeadRecorder()
    t = tools(rec)
    await t["set_language"](language="english")
    await t["set_name"](name="Ravi")
    await t["set_service"](service="plumber")
    await t["set_city"](city="Pune")
    assert "saved" in (await t["save_lead"]()).lower()

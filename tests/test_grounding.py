"""A value only counts if the caller was heard saying it.

The bug this exists for: on a Hindi call the lead came back with a name nobody
had spoken. On the speech-to-speech path the model is the transcriber, and
`set_name` took whatever string it passed — a realtime model on a bad line does
not fail loudly, it produces a fluent, plausible Indian name.

So every tool value is checked against the transcript of the caller's own
audio, which comes from a different model than the one hearing the call. Two
independent decodes agree on a name that was said and disagree on one that was
invented.
"""

from __future__ import annotations

import pytest

from backend.app.realtime_tools import LeadRecorder
from backend.app.services import grounding
from backend.app.services.grounding import Verdict


def tools(rec: LeadRecorder) -> dict:
    return {t.info.name: t for t in rec.build_tools()}


def heard(rec: LeadRecorder, *turns: str) -> LeadRecorder:
    for turn in turns:
        rec.note_caller_speech(turn)
    return rec


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


# --- through the tools ----------------------------------------------------


@pytest.mark.asyncio
async def test_an_invented_hindi_name_never_reaches_the_session():
    rec = heard(LeadRecorder(), "नमस्ते", "मेरा नाम राहुल है")
    t = tools(rec)

    reply = await t["set_name"](name="Priya Sharma")
    assert rec.session.user_name is None
    assert "did not hear" in reply.lower()

    await t["set_name"](name="राहुल")
    assert rec.session.user_name == "Rahul"


@pytest.mark.asyncio
async def test_an_invented_city_never_reaches_the_session():
    rec = heard(LeadRecorder(), "मुझे मुंबई में चाहिए")
    t = tools(rec)

    await t["set_city"](city="Delhi")
    assert rec.session.city_or_area is None

    await t["set_city"](city="मुंबई")
    assert rec.session.city_or_area == "Mumbai"


@pytest.mark.asyncio
async def test_an_invented_service_never_reaches_the_session():
    rec = heard(LeadRecorder(), "मुझे प्लंबर चाहिए")
    t = tools(rec)

    await t["set_service"](service="tuition classes")
    assert rec.session.requested_service is None

    await t["set_service"](service="प्लंबर")
    assert rec.session.requested_service == "plumber"


@pytest.mark.asyncio
async def test_the_rules_can_corroborate_a_service_no_word_of_which_was_said():
    """"AC repair" shares no word with "मुझे एसी ठीक करवाना है", so grounding
    alone refuses it — but the catalogue reaches `ac repair` from the caller's
    own words too, and two independent routes to the same trade is the
    corroboration being asked for."""
    rec = heard(LeadRecorder(), "मुझे एसी ठीक करवाना है")
    assert grounding.check(
        "AC repair", list(rec.heard), every_word=False
    ) is Verdict.NOT_HEARD

    await tools(rec)["set_service"](service="AC repair")
    assert rec.session.requested_service == "ac repair"


@pytest.mark.asyncio
async def test_corroboration_does_not_admit_a_different_trade():
    """The second chance is for reaching the SAME trade another way, not for
    letting any catalogue label through."""
    rec = heard(LeadRecorder(), "मुझे एसी ठीक करवाना है")
    await tools(rec)["set_service"](service="tutor")
    assert rec.session.requested_service is None


@pytest.mark.asyncio
async def test_a_refused_value_leaves_the_rest_of_the_state_alone():
    """A rejection is a re-ask, not a reset."""
    rec = heard(LeadRecorder(), "मुझे प्लंबर चाहिए", "मुंबई")
    t = tools(rec)
    await t["set_service"](service="प्लंबर")
    await t["set_city"](city="मुंबई")

    reply = await t["set_name"](name="Anjali")
    assert rec.session.requested_service == "plumber"
    assert rec.session.city_or_area == "Mumbai"
    assert "plumber" in reply and "Mumbai" in reply


@pytest.mark.asyncio
async def test_a_value_is_grounded_in_recent_turns_not_only_the_last():
    """The model does not always record on the turn the answer arrived."""
    rec = heard(LeadRecorder(), "मेरा नाम राहुल है", "हाँ", "मुंबई में")
    t = tools(rec)
    await t["set_name"](name="राहुल")
    assert rec.session.user_name == "Rahul"


@pytest.mark.asyncio
async def test_an_old_turn_falls_out_of_the_window():
    """Grounding is evidence about what was just said, not a session-long
    licence to reuse any word the caller ever uttered."""
    rec = heard(LeadRecorder(), "मेरा नाम राहुल है")
    for filler in ("हाँ", "ठीक है", "अच्छा", "जी"):
        rec.note_caller_speech(filler)
    t = tools(rec)
    await t["set_name"](name="राहुल")
    assert rec.session.user_name is None


@pytest.mark.asyncio
async def test_a_call_with_no_transcript_still_works():
    """Transcription is a second opinion, not a dependency: without it the tools
    behave as they did before, rather than refusing the whole call."""
    rec = LeadRecorder()
    t = tools(rec)
    await t["set_name"](name="Ravi")
    assert rec.session.user_name == "Ravi"


@pytest.mark.asyncio
async def test_a_business_the_caller_never_named_is_not_looked_up():
    from backend.app.services import brain

    rec = heard(LeadRecorder(), "पैक्स ज्वेलर्स का नंबर दीजिए")
    t = tools(rec)
    calls: list = []
    original, brain.available = brain.available, lambda: (calls.append(1) or True)
    try:
        reply = await t["lookup_vendor_contact"](business="Speed Autos")
    finally:
        brain.available = original
    assert "did not hear" in reply.lower()

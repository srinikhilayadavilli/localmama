"""The read-back step: last chance to catch a mistranscription.

Speech recognition is nondeterministic — the same audio produced "plumber" once
and "puma" another time on a live LiveKit call, and "puma" reached the lead.
Reading the details back and requiring agreement is the only defence that
catches an error the extractor was confident about.

The load-bearing property: **nothing commits without explicit agreement.**
"""

from __future__ import annotations

import pytest

from backend.app.languages import Language
from backend.app.models import ConversationState, SessionData
from backend.app.prompts.messages import MessageKey, get_message
from backend.app.services import confirmation
from backend.app.services.conversation_manager import ConversationManager

FILLED = ["English", "My name is Ravi", "I need a plumber", "Madhapur Hyderabad"]


async def at_review(session_id="review-test") -> ConversationManager:
    m = ConversationManager(SessionData(session_id=session_id))
    m.start()
    for text in FILLED:
        result = await m.handle(text)
    assert result.state is ConversationState.REVIEW
    return m


# --- yes / no detection ---------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["yes", "yeah", "correct", "that's right", "haan ji", "avunu", "sari",
     "houdu", "হ্যাঁ", "ஆமாம்", "बिल्कुल सही"],
)
def test_affirmatives_across_languages(text):
    assert confirmation.is_affirmative(text) is True


@pytest.mark.parametrize(
    "text",
    ["no", "nope", "that's wrong", "nahi", "galat", "kadu", "illai",
     "ಇಲ್ಲ", "না", "नहीं"],
)
def test_negatives_across_languages(text):
    assert confirmation.is_negative(text) is True
    assert confirmation.is_affirmative(text) is False


def test_negation_beats_affirmation_in_a_mixed_utterance():
    """'yes but no that's wrong' must not be read as agreement."""
    assert confirmation.is_affirmative("yes but no that is wrong") is False


@pytest.mark.parametrize(
    "text,field",
    [
        ("the name is wrong", "user_name"),
        ("change the service", "requested_service"),
        ("the area is wrong", "city_or_area"),
        ("मेरा नाम गलत है", "user_name"),
        ("ప్రాంతం తప్పు", "city_or_area"),
    ],
)
def test_field_callouts(text, field):
    assert confirmation.field_to_correct(text) == field


# --- the read-back gate ---------------------------------------------------


@pytest.mark.asyncio
async def test_call_stops_at_review_before_committing():
    m = await at_review()
    assert m.session.conversation_status.value == "in_progress"
    assert m.session.completed_at is None


@pytest.mark.asyncio
async def test_read_back_states_every_captured_field():
    m = await at_review()
    reply = m.session.transcript[-1].text
    assert "Ravi" in reply and "plumber" in reply and "Madhapur" in reply


@pytest.mark.asyncio
async def test_agreement_completes_the_call():
    m = await at_review()
    result = await m.handle("yes that's right")
    assert result.state is ConversationState.COMPLETED
    assert result.lead is not None
    assert result.lead.requested_service == "plumber"


@pytest.mark.asyncio
@pytest.mark.parametrize("silence", ["...", "umm", "hmm"])
async def test_ambiguity_never_completes_the_call(silence):
    """Only explicit agreement advances — never a shrug."""
    m = await at_review()
    result = await m.handle(silence)
    assert result.state is ConversationState.REVIEW
    assert result.lead is None


@pytest.mark.asyncio
async def test_bare_no_asks_what_to_change():
    m = await at_review()
    result = await m.handle("no")
    assert result.state is ConversationState.REVIEW
    assert get_message(MessageKey.REVIEW_WHAT_TO_CHANGE, Language.ENGLISH) in result.reply
    assert m.session.requested_service == "plumber"   # nothing cleared yet


# --- corrections ----------------------------------------------------------


@pytest.mark.asyncio
async def test_inline_correction_overwrites_and_reads_back_again():
    """The exact live-call failure: the wrong trade was captured."""
    m = await at_review()
    result = await m.handle("no, I need an electrician")

    assert m.session.requested_service == "electrician"
    assert result.state is ConversationState.REVIEW      # confirm the new value
    assert result.lead is None

    result = await m.handle("yes")
    assert result.state is ConversationState.COMPLETED
    assert result.lead.requested_service == "electrician"


@pytest.mark.asyncio
async def test_name_can_be_corrected_inline():
    m = await at_review()
    await m.handle("no, my name is Ravi Kumar")
    assert m.session.user_name == "Ravi Kumar"


@pytest.mark.asyncio
async def test_location_can_be_corrected_inline():
    m = await at_review()
    await m.handle("no, Bangalore")
    assert "Bangalore" in m.session.city_or_area


@pytest.mark.asyncio
async def test_naming_a_wrong_field_reopens_that_question():
    m = await at_review()
    result = await m.handle("the area is wrong")

    assert result.state is ConversationState.ASK_LOCATION
    assert m.session.city_or_area is None
    assert m.session.requested_service == "plumber"   # others untouched

    result = await m.handle("Bangalore")
    assert result.state is ConversationState.REVIEW   # back for confirmation
    assert m.session.city_or_area == "Bangalore"

    result = await m.handle("yes")
    assert result.lead.city_or_area == "Bangalore"


@pytest.mark.asyncio
async def test_correction_cannot_skip_the_second_read_back():
    """A corrected value must itself be confirmed before it becomes a lead."""
    m = await at_review()
    result = await m.handle("no, I need an electrician")
    assert result.state is ConversationState.REVIEW
    assert m.session.conversation_status.value == "in_progress"


@pytest.mark.asyncio
async def test_read_back_is_in_the_session_language():
    m = ConversationManager(SessionData(session_id="review-te"))
    m.start()
    for text in ["Telugu", "నా పేరు రవి", "electrician", "Madhapur Hyderabad"]:
        result = await m.handle(text)
    assert result.state is ConversationState.REVIEW
    assert "మామా" in result.reply or "సరైనదేనా" in result.reply

    result = await m.handle("avunu")
    assert result.state is ConversationState.COMPLETED

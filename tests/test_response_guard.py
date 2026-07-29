"""The contract that makes free-form generation safe.

Generation is allowed to change *how* a turn sounds, never *what the call has
done*. Every test here is a sentence that is fluent, plausible, and would be
damaging if spoken — the guard's job is to reject it and let the deterministic
template through instead.
"""

from __future__ import annotations

import pytest

from backend.app.languages import Language
from backend.app.models import ConversationState, SessionData
from backend.app.services.response_guard import MAX_REPLY_CHARS, validate


def session_at(state: ConversationState, **fields) -> SessionData:
    return SessionData(session_id="guard-test", state=state, **fields)


def ok(reply, session, language=Language.ENGLISH) -> bool:
    return validate(reply, session, language)[0]


def why(reply, session, language=Language.ENGLISH) -> str:
    return validate(reply, session, language)[1]


# --- premature promises: the most damaging failure ------------------------


@pytest.mark.parametrize(
    "reply",
    [
        "Sure Mama, I will send the details to your WhatsApp. What is your name?",
        "Your booking is confirmed! May I know your name?",
        "I have booked an electrician for you. What is your name?",
        "A technician is on the way. What is your name?",
        "I'll send you the best matches shortly. What is your name?",
    ],
)
def test_promises_before_confirmation_are_rejected(reply):
    s = session_at(ConversationState.ASK_NAME, selected_language=Language.ENGLISH)
    assert ok(reply, s) is False


def test_the_same_promise_is_allowed_at_confirmation():
    s = session_at(
        ConversationState.CONFIRMATION,
        selected_language=Language.ENGLISH,
        user_name="Ravi",
        requested_service="plumber",
        city_or_area="Pune",
    )
    reply = "Perfect Ravi, I will send the best plumber details in Pune to your WhatsApp."
    assert ok(reply, s) is True


# --- invented entities ----------------------------------------------------


def test_invented_service_is_rejected():
    s = session_at(ConversationState.ASK_NAME, selected_language=Language.ENGLISH)
    assert ok("I'll find you an electrician. What is your name?", s) is False
    assert "electrician" in why("I'll find you an electrician. What is your name?", s)


def test_invented_city_is_rejected():
    s = session_at(ConversationState.ASK_SERVICE, selected_language=Language.ENGLISH)
    assert ok("Great, service in Mumbai. What service do you need?", s) is False


def test_captured_entities_may_be_referenced():
    s = session_at(
        ConversationState.ASK_LOCATION,
        selected_language=Language.ENGLISH,
        user_name="Ravi",
        requested_service="plumber",
    )
    assert ok("Got it Ravi, a plumber. Which area do you need it in?", s) is True


def test_captured_city_may_be_referenced():
    s = session_at(
        ConversationState.ASK_SERVICE,
        selected_language=Language.ENGLISH,
        city_or_area="Pune",
    )
    assert ok("Thanks! In Pune, what service do you need?", s) is True


# --- fabricated specifics -------------------------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        "Call 9876543210 for help. What is your name?",
        "It will cost around 1500 rupees. What is your name?",
        "Visit https://localmama.example for details. What is your name?",
        "See www.localmama.in. What is your name?",
    ],
)
def test_fabricated_numbers_and_links_are_rejected(reply):
    s = session_at(ConversationState.ASK_NAME, selected_language=Language.ENGLISH)
    assert ok(reply, s) is False


# --- voice safety ---------------------------------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        "**Great!** What is your name?",
        "Here are options:\n- one\n- two\nWhat is your name?",
        "# Welcome\nWhat is your name?",
        "Happy to help 😊 What is your name?",
    ],
)
def test_markdown_and_emoji_are_rejected(reply):
    s = session_at(ConversationState.ASK_NAME, selected_language=Language.ENGLISH)
    assert ok(reply, s) is False


def test_overlong_replies_are_rejected():
    s = session_at(ConversationState.ASK_NAME, selected_language=Language.ENGLISH)
    assert ok("word " * 200 + "What is your name?", s) is False


def test_empty_reply_is_rejected():
    s = session_at(ConversationState.ASK_NAME, selected_language=Language.ENGLISH)
    assert ok("", s) is False
    assert ok("   ", s) is False


# --- staying on task ------------------------------------------------------


def test_a_reply_that_stops_asking_is_rejected():
    """Chatty but useless — the caller is left with no question to answer."""
    s = session_at(ConversationState.ASK_NAME, selected_language=Language.ENGLISH)
    assert ok("That's wonderful to hear, Mama. Have a lovely day.", s) is False


def test_closing_need_not_ask_anything():
    s = session_at(
        ConversationState.CLOSING,
        selected_language=Language.ENGLISH,
        user_name="Ravi",
        requested_service="plumber",
        city_or_area="Pune",
    )
    assert ok("Thank you for choosing Local Mama. Have a wonderful day, Mama.", s) is True


# --- language ------------------------------------------------------------


def test_english_reply_in_a_telugu_session_is_rejected():
    s = session_at(ConversationState.ASK_NAME, selected_language=Language.TELUGU)
    assert ok("Sure, may I know your name?", s, Language.TELUGU) is False


def test_telugu_reply_in_a_telugu_session_is_accepted():
    s = session_at(ConversationState.ASK_NAME, selected_language=Language.TELUGU)
    assert ok("సరే మామా! మీ పేరు తెలుసుకోవచ్చా?", s, Language.TELUGU) is True


def test_code_mixed_indic_reply_is_accepted():
    """Indian speech mixes English constantly; the guard must not punish it."""
    s = session_at(
        ConversationState.ASK_LOCATION,
        selected_language=Language.HINDI,
        requested_service="plumber",
    )
    assert ok("ठीक है मामा, plumber चाहिए। किस area में चाहिए?", s, Language.HINDI) is True


def test_reason_is_reported_for_every_rejection():
    """Failures must be explainable, so they can be studied rather than guessed at."""
    s = session_at(ConversationState.ASK_NAME, selected_language=Language.ENGLISH)
    passed, reason = validate("I will send it to your WhatsApp.", s, Language.ENGLISH)
    assert passed is False and reason


def test_a_good_reply_passes_cleanly():
    s = session_at(ConversationState.ASK_NAME, selected_language=Language.ENGLISH)
    passed, reason = validate(
        "Lovely to meet you! May I know your name, Mama?", s, Language.ENGLISH
    )
    assert passed is True and reason == ""
    assert len("Lovely to meet you! May I know your name, Mama?") < MAX_REPLY_CHARS

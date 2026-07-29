"""Casual conversation must be warm — and must never move the workflow.

The safety property under test is that no amount of chit-chat, flattery,
abuse, or instruction-shaped text can change the agent's position in the flow
or write a field. Warmth is a phrasing choice layered on top of a state machine
that ignores it entirely.
"""

from __future__ import annotations

import pytest

from backend.app.languages import Language
from backend.app.models import ConversationState, SessionData
from backend.app.prompts.messages import MessageKey, get_message
from backend.app.services import smalltalk
from backend.app.services.conversation_manager import ConversationManager


def manager() -> ConversationManager:
    return ConversationManager(SessionData(session_id="smalltalk-test"))


async def at_name_step() -> ConversationManager:
    m = manager()
    m.start()
    await m.handle("English")
    assert m.session.state is ConversationState.ASK_NAME
    return m


# --- classification ------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("hello there!", MessageKey.SMALLTALK_GREETING),
        ("namaste", MessageKey.SMALLTALK_GREETING),
        ("good morning", MessageKey.SMALLTALK_GREETING),
        ("how are you?", MessageKey.SMALLTALK_HOW_ARE_YOU),
        ("kaise ho", MessageKey.SMALLTALK_HOW_ARE_YOU),
        ("who are you?", MessageKey.SMALLTALK_IDENTITY),
        ("are you a robot", MessageKey.SMALLTALK_IDENTITY),
        ("what is your name", MessageKey.SMALLTALK_IDENTITY),
        ("thank you so much", MessageKey.SMALLTALK_THANKS),
        ("dhanyavaad", MessageKey.SMALLTALK_THANKS),
        ("I am so stressed", MessageKey.SMALLTALK_EMPATHY),
        ("my bathroom is flooded", MessageKey.SMALLTALK_EMPATHY),
        ("this is urgent", MessageKey.SMALLTALK_EMPATHY),
        ("you are useless and stupid", MessageKey.SMALLTALK_ABUSE),
    ],
)
def test_classification(text, expected):
    assert smalltalk.classify(text) is expected


@pytest.mark.parametrize("text", ["Ravi", "electrician", "Hyderabad", "Telugu", ""])
def test_real_answers_are_not_smalltalk(text):
    assert smalltalk.classify(text) is None


def test_greeting_does_not_leak_into_a_field():
    """Regression: 'hello there' matched the Telugu locative 'X lo' inside
    'hel-lo' and was captured as a city named 'Hel'."""
    from backend.app.services.entity_extractor import extract

    result = extract("hello there!", expecting=ConversationState.LANGUAGE_SELECTION)
    assert result.city_or_area is None
    assert result.name is None


def test_politeness_is_not_a_name():
    """Regression: 'thank you so much' was stored as the caller's name."""
    from backend.app.services.entity_extractor import extract

    assert extract("thank you so much", expecting=ConversationState.ASK_NAME).name is None


# --- behaviour in a live call --------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "chit_chat",
    ["hello there!", "how are you?", "who are you?", "thank you", "I am so stressed"],
)
async def test_smalltalk_never_advances_state_or_captures(chit_chat):
    m = await at_name_step()
    before = m.session.model_copy(deep=True)

    result = await m.handle(chit_chat)

    assert result.state is ConversationState.ASK_NAME
    assert m.session.state is before.state
    assert m.session.user_name is None
    assert m.session.requested_service is None
    assert m.session.city_or_area is None
    assert m.session.selected_language is before.selected_language


@pytest.mark.asyncio
async def test_smalltalk_acknowledges_then_reasks():
    m = await at_name_step()
    result = await m.handle("how are you?")
    ack = get_message(MessageKey.SMALLTALK_HOW_ARE_YOU, Language.ENGLISH)
    question = get_message(MessageKey.ASK_NAME, Language.ENGLISH)
    assert ack in result.reply
    assert question in result.reply


@pytest.mark.asyncio
async def test_smalltalk_is_answered_in_the_session_language():
    m = manager()
    m.start()
    await m.handle("Telugu")
    result = await m.handle("how are you?")
    assert result.reply.startswith(
        get_message(MessageKey.SMALLTALK_HOW_ARE_YOU, Language.TELUGU)
    )


@pytest.mark.asyncio
async def test_explicit_answer_beats_smalltalk():
    """'thanks, my name is Ravi' is an answer, not chit-chat."""
    m = await at_name_step()
    result = await m.handle("thank you, my name is Ravi")
    assert m.session.user_name == "Ravi"
    assert result.state is ConversationState.ASK_SERVICE


@pytest.mark.asyncio
async def test_abuse_is_de_escalated_not_mirrored():
    m = await at_name_step()
    result = await m.handle("you are useless and stupid")
    assert get_message(MessageKey.SMALLTALK_ABUSE, Language.ENGLISH) in result.reply
    assert m.session.state is ConversationState.ASK_NAME


@pytest.mark.asyncio
async def test_endless_chit_chat_falls_back_to_a_plain_reprompt():
    """A caller who only chats cannot keep the call in small-talk forever."""
    m = await at_name_step()
    for _ in range(smalltalk.MAX_CONSECUTIVE):
        await m.handle("how are you?")
    result = await m.handle("how are you?")
    assert get_message(MessageKey.REPROMPT_NAME, Language.ENGLISH) in result.reply
    assert m.session.state is ConversationState.ASK_NAME


@pytest.mark.asyncio
async def test_call_still_completes_after_lots_of_chit_chat():
    m = manager()
    m.start()
    for text in ["hello!", "how are you?", "English", "thanks!", "My name is Ravi",
                 "my sink is leaking and I am stressed", "plumber", "Pune", "yes"]:
        result = await m.handle(text)
    assert result.state is ConversationState.COMPLETED
    assert result.lead is not None
    assert result.lead.user_name == "Ravi"
    assert result.lead.requested_service == "plumber"
    assert result.lead.city_or_area == "Pune"

"""Free-form phrasing must never be able to damage a call.

These drive a real `ConversationManager` with the generator stubbed, covering
the cases that decide whether this feature is safe to ship: a good generation
is used, a bad one is discarded, and in neither case can the generated text
move the workflow or write a field.
"""

from __future__ import annotations

import pytest

from backend.app.config import Settings
from backend.app.languages import Language
from backend.app.models import ConversationState, SessionData
from backend.app.prompts.messages import MessageKey, get_message
from backend.app.services import response_generator
from backend.app.services.conversation_manager import ConversationManager


@pytest.fixture
def natural_on(monkeypatch):
    """Turn the feature on without needing an API key."""
    monkeypatch.setattr(
        Settings, "natural_replies_available", property(lambda self: True)
    )


def stub_generator(monkeypatch, text):
    """Replace the LLM call with a fixed result (or None for failure)."""

    async def _fake(**kwargs):
        return text

    monkeypatch.setattr(response_generator, "generate", _fake)


def manager() -> ConversationManager:
    return ConversationManager(SessionData(session_id="natural-test"))


async def at_name_step() -> ConversationManager:
    m = manager()
    m.start()
    await m.handle("English")
    assert m.session.state is ConversationState.ASK_NAME
    return m


# --- feature disabled (the default) --------------------------------------


@pytest.mark.asyncio
async def test_disabled_by_default_uses_templates():
    m = await at_name_step()
    result = await m.handle("...")
    assert result.reply == get_message(MessageKey.REPROMPT_NAME, Language.ENGLISH)


# --- valid generation ----------------------------------------------------


@pytest.mark.asyncio
async def test_valid_generation_is_spoken(natural_on, monkeypatch):
    stub_generator(monkeypatch, "Lovely to meet you! May I know your name, Mama?")
    m = await at_name_step()
    result = await m.handle("...")
    assert result.reply == "Lovely to meet you! May I know your name, Mama?"
    assert result.state is ConversationState.ASK_NAME


@pytest.mark.asyncio
async def test_transcript_records_what_was_actually_spoken(natural_on, monkeypatch):
    stub_generator(monkeypatch, "Happy to help! What is your name, Mama?")
    m = await at_name_step()
    await m.handle("...")
    last_agent = [t for t in m.session.transcript if t.role == "agent"][-1]
    assert last_agent.text == "Happy to help! What is your name, Mama?"


# --- invalid generation falls back --------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad",
    [
        "Sure! I will send the details to your WhatsApp now. What is your name?",
        "I'll get an electrician to you. What is your name?",
        "Call 9876543210 meanwhile. What is your name?",
        "**Sure!** What is your name?",
        "Thanks, have a great day.",           # stops asking
        "నమస్తే మామా, మీ పేరు?",                 # wrong language for the session
    ],
)
async def test_unsafe_generation_falls_back_to_template(natural_on, monkeypatch, bad):
    stub_generator(monkeypatch, bad)
    m = await at_name_step()
    result = await m.handle("...")

    assert result.reply == get_message(MessageKey.REPROMPT_NAME, Language.ENGLISH)
    assert result.state is ConversationState.ASK_NAME
    assert m.session.user_name is None


@pytest.mark.asyncio
async def test_generation_failure_falls_back(natural_on, monkeypatch):
    """Timeout, refusal, or API error all surface as None."""
    stub_generator(monkeypatch, None)
    m = await at_name_step()
    result = await m.handle("...")
    assert result.reply == get_message(MessageKey.REPROMPT_NAME, Language.ENGLISH)


# --- generation can never move the workflow ------------------------------


@pytest.mark.asyncio
async def test_generated_text_cannot_capture_a_field(natural_on, monkeypatch):
    """Even a reply asserting the name was captured leaves the session untouched."""
    stub_generator(monkeypatch, "Thanks Ravi, got your name! What is your name?")
    m = await at_name_step()
    await m.handle("...")
    assert m.session.user_name is None
    assert m.session.state is ConversationState.ASK_NAME


@pytest.mark.asyncio
async def test_generated_text_cannot_complete_the_call(natural_on, monkeypatch):
    stub_generator(monkeypatch, "All done, your request is complete. What is your name?")
    m = await at_name_step()
    result = await m.handle("...")
    assert result.state is ConversationState.ASK_NAME
    assert result.lead is None
    assert m.session.conversation_status.value == "in_progress"


@pytest.mark.asyncio
async def test_full_call_still_completes_with_phrasing_on(natural_on, monkeypatch):
    """Phrasing is cosmetic: the same inputs still produce the same lead."""
    stub_generator(monkeypatch, None)  # exercise the fallback throughout
    m = manager()
    m.start()
    for text in ["English", "My name is Ravi", "plumber", "Pune", "yes"]:
        result = await m.handle(text)
    assert result.state is ConversationState.COMPLETED
    assert result.lead.user_name == "Ravi"
    assert result.lead.requested_service == "plumber"
    assert result.lead.city_or_area == "Pune"


@pytest.mark.asyncio
async def test_phrasing_is_applied_to_smalltalk_too(natural_on, monkeypatch):
    stub_generator(monkeypatch, "I'm doing well, thanks for asking! Your name, Mama?")
    m = await at_name_step()
    result = await m.handle("how are you?")
    assert result.reply == "I'm doing well, thanks for asking! Your name, Mama?"
    assert result.state is ConversationState.ASK_NAME
    assert m.session.user_name is None

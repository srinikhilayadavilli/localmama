"""Preemptive phrasing: generate the next line while the caller is still talking.

Only possible because progression is deterministic — fill in the field being
asked for, ask the state machine where that lands, and render that prompt. A
free-form agent cannot know its own next line.

The regression that matters: an earlier version reset the cache inside the turn
handler, moments before the lookup, so the hit rate was silently zero.
"""

from __future__ import annotations

import pytest

from backend.app.config import Settings
from backend.app.models import ConversationState, SessionData
from backend.app.prompts.messages import MessageKey, get_message
from backend.app.languages import Language
from backend.app.services import response_generator
from backend.app.services.conversation_manager import ConversationManager


@pytest.fixture
def natural_on(monkeypatch):
    monkeypatch.setattr(Settings, "natural_replies_available", property(lambda self: True))


def manager() -> ConversationManager:
    return ConversationManager(SessionData(session_id="prefetch-test"))


def test_predicts_the_next_line_from_the_state_machine():
    m = manager()
    m.session.state = ConversationState.ASK_SERVICE
    m.session.selected_language = Language.ENGLISH
    m.session.user_name = "Ravi"
    assert m._predicted_next_intent() == get_message(
        MessageKey.ASK_LOCATION, Language.ENGLISH
    )


def test_does_not_predict_a_line_needing_an_uncaptured_value():
    """ASK_SERVICE embeds {name} — the very field about to be collected.
    Predicting it renders an empty name and the cache can never match."""
    m = manager()
    m.session.state = ConversationState.ASK_NAME
    m.session.selected_language = Language.ENGLISH
    assert m.session.user_name is None
    assert m._predicted_next_intent() is None


def test_does_not_predict_the_read_back():
    """REVIEW interpolates every captured value."""
    m = manager()
    m.session.state = ConversationState.ASK_LOCATION
    m.session.selected_language = Language.ENGLISH
    m.session.user_name = "Ravi"
    m.session.requested_service = "plumber"
    assert m._predicted_next_intent() is None


@pytest.mark.asyncio
async def test_cache_survives_a_turn(natural_on, monkeypatch):
    """Regression: the cache was re-initialised inside the turn handler, so a
    prefetched line was always discarded before it could be used."""
    async def fake(**kwargs):
        return "Sure! Which area do you need this in, Mama?"
    monkeypatch.setattr(response_generator, "generate", fake)

    m = manager()
    m.start()
    await m.handle("English")
    m._prefetched["sentinel"] = "still here"
    await m.handle("My name is Ravi")
    assert m._prefetched.get("sentinel") == "still here"


@pytest.mark.asyncio
async def test_a_prefetched_line_is_used_and_still_validated(natural_on, monkeypatch):
    calls = {"n": 0}

    async def fake(**kwargs):
        calls["n"] += 1
        return "Got it! Which city or area should I look in, Mama?"

    monkeypatch.setattr(response_generator, "generate", fake)
    m = manager()
    m.session.state = ConversationState.ASK_SERVICE
    m.session.selected_language = Language.ENGLISH
    m.session.user_name = "Ravi"
    m._prefetched[get_message(MessageKey.ASK_LOCATION, Language.ENGLISH)] = (
        "Got it! Which city or area should I look in, Mama?"
    )
    before = calls["n"]
    result = await m.handle("I need a plumber")
    # Served from cache: no live generation for this turn's reply.
    assert result.reply == "Got it! Which city or area should I look in, Mama?"
    assert calls["n"] == before  # only the follow-up prefetch may fire, not a live call


@pytest.mark.asyncio
async def test_an_unsafe_prefetched_line_is_still_rejected(natural_on, monkeypatch):
    """Cached text goes through the same guard as freshly generated text."""
    async def fake(**kwargs):
        return None
    monkeypatch.setattr(response_generator, "generate", fake)

    m = manager()
    m.session.state = ConversationState.ASK_SERVICE
    m.session.selected_language = Language.ENGLISH
    m.session.user_name = "Ravi"
    m._prefetched[get_message(MessageKey.ASK_LOCATION, Language.ENGLISH)] = (
        "I'll send it to your WhatsApp now. Which area?"      # premature promise
    )
    result = await m.handle("I need a plumber")
    assert result.reply == get_message(MessageKey.ASK_LOCATION, Language.ENGLISH)


@pytest.mark.asyncio
async def test_prefetch_failure_never_breaks_a_call(natural_on, monkeypatch):
    async def boom(**kwargs):
        raise RuntimeError("provider exploded")
    monkeypatch.setattr(response_generator, "generate", boom)

    m = manager()
    m.start()
    for text in ["English", "My name is Ravi", "plumber", "Pune"]:
        result = await m.handle(text)
    assert result.state is ConversationState.REVIEW

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


def test_predicts_a_line_needing_an_uncaptured_value_via_placeholder():
    """ASK_SERVICE embeds {name} — the very field about to be collected.

    This used to return None, which is why the greeting-by-name and the
    read-back were never prefetched and always paid the live call. Now the
    unknown value is rendered as a placeholder and substituted at speak time.
    """
    m = manager()
    m.session.state = ConversationState.ASK_NAME
    m.session.selected_language = Language.ENGLISH
    assert m.session.user_name is None
    intent = m._predicted_next_intent()
    assert intent is not None
    assert response_generator.PLACEHOLDERS["name"] in intent


def test_predicts_the_read_back_with_placeholders():
    """REVIEW interpolates every captured value — including the one still missing."""
    m = manager()
    m.session.state = ConversationState.ASK_LOCATION
    m.session.selected_language = Language.ENGLISH
    m.session.user_name = "Ravi"
    m.session.requested_service = "plumber"
    intent = m._predicted_next_intent()
    assert intent is not None
    # EVERY interpolated field is a placeholder, including the ones already
    # captured. `_placeholder_form` cannot know at lookup time which values were
    # known when the line was phrased, so it swaps all of them; rendering only
    # the unknown ones here makes the two keys differ and the cache never hits.
    for token in response_generator.PLACEHOLDERS.values():
        assert token in intent
    assert "Ravi" not in intent and "plumber" not in intent


def test_does_not_predict_before_a_language_is_chosen():
    """Phrasing in the wrong language is a call spent on text we throw away."""
    m = manager()
    m.session.state = ConversationState.ASK_NAME
    assert m.session.selected_language is None
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


# --- placeholder round trip -------------------------------------------
#
# A prefetched line is phrased before the caller says the value it contains,
# so it comes back holding {{NAME}} / {{SERVICE}} / {{LOCATION}}. Everything
# below is about the one risk that creates: speaking a line with a brace still
# in it, or with the wrong value substituted.


def _ready_manager():
    m = manager()
    m.session.selected_language = Language.ENGLISH
    m.session.user_name = "Ravi"
    m.session.requested_service = "plumber"
    m.session.city_or_area = "Madhapur"
    return m


def test_placeholder_form_is_the_inverse_of_rendering():
    m = _ready_manager()
    rendered = "Got it Ravi, a plumber in Madhapur. Correct?"
    keyed = m._placeholder_form(rendered)
    assert keyed == "Got it {{NAME}}, a {{SERVICE}} in {{LOCATION}}. Correct?"
    assert m._fill_placeholders(keyed) == rendered


def test_a_mangled_placeholder_is_rejected():
    """The model garbled the token. Half a brace must never be spoken."""
    m = _ready_manager()
    assert m._fill_placeholders("Got it {{NAM}}, a plumber in Madhapur.") is None
    assert m._fill_placeholders("Got it {{NAME}, a plumber.") is None


def test_a_placeholder_with_no_value_is_rejected():
    """Nothing to substitute — better a live call than an empty slot."""
    m = manager()
    m.session.selected_language = Language.ENGLISH
    assert m.session.user_name is None
    assert m._fill_placeholders("Nice to meet you, {{NAME}}!") is None


def test_a_dropped_placeholder_still_fills():
    """The model omitted the name entirely. That is merely less personal, not
    wrong, so it is allowed through to the guard like any other phrasing."""
    m = _ready_manager()
    assert m._fill_placeholders("Nice to meet you!") == "Nice to meet you!"


def test_short_values_are_not_substituted_into_the_key():
    """A one-character name would corrupt unrelated text on replacement."""
    m = manager()
    m.session.selected_language = Language.ENGLISH
    m.session.user_name = "A"
    assert m._placeholder_form("A plumber in Madhapur") == "A plumber in Madhapur"


@pytest.mark.asyncio
async def test_prefetched_line_with_placeholders_is_filled_and_spoken(
    natural_on, monkeypatch
):
    """End to end: the cached line holds a placeholder, the caller supplies the
    value, and what is spoken contains the real name and no braces."""
    async def fake(**kwargs):
        return "Lovely to meet you, {{NAME}}! What service do you need today?"
    monkeypatch.setattr(response_generator, "generate", fake)

    m = manager()
    m.start()
    await m.handle("English")
    # Prefetch is fire-and-forget; wait for it so the assertion is about the
    # cache being used, not about a race.
    if m._prefetch_task is not None:
        await m._prefetch_task
    result = await m.handle("My name is Ravi")
    assert "Ravi" in result.reply
    assert "{{" not in result.reply and "}}" not in result.reply


def test_prediction_key_matches_the_lookup_key():
    """The bug that held the read-back at 0% hits: the two keys are built by
    different code paths, so assert directly that they agree."""
    m = manager()
    m.session.selected_language = Language.ENGLISH
    m.session.state = ConversationState.ASK_LOCATION
    m.session.user_name = "Ravi"
    m.session.requested_service = "plumber"

    predicted = m._predicted_next_intent()

    # The caller now answers, and the real line is rendered with real values.
    m.session.city_or_area = "Madhapur"
    actual = get_message(
        MessageKey.REVIEW, m.session.language,
        name="Ravi", service="plumber", location="Madhapur",
    )
    assert m._placeholder_form(actual) == predicted


@pytest.mark.asyncio
async def test_finalising_a_lead_does_not_block_on_the_network(natural_on, monkeypatch):
    """The caller hears silence until the closing line is spoken.

    Both durable steps reach the network — a psycopg round trip to Neon and an
    HTTPS POST that retries three times when the provider is down. Measured at
    15s of dead air on the speech-to-speech path before the same fix landed
    there. The lead is on disk before either runs, so nothing is lost if they
    fail.
    """
    import asyncio
    import time

    from backend.app.services import lead_store, whatsapp

    async def fake_phrasing(**kwargs):
        return None

    monkeypatch.setattr(response_generator, "generate", fake_phrasing)

    slow_calls = {"n": 0}

    def slow_save(lead, caller_phone=""):
        slow_calls["n"] += 1
        time.sleep(0.6)          # stand-in for a slow database
        return True

    monkeypatch.setattr(lead_store, "save", slow_save)
    monkeypatch.setattr(whatsapp, "fire", lambda *a, **k: None)

    m = manager()
    m.start()
    for text in ("English", "my name is Ravi", "plumber", "Hyderabad"):
        await m.handle(text)

    started = time.perf_counter()
    result = await m.handle("yes")
    elapsed = time.perf_counter() - started

    assert result.is_final
    assert elapsed < 0.4, f"the caller waited {elapsed:.2f}s on a slow database"

    await asyncio.gather(*m._background, return_exceptions=True)
    assert slow_calls["n"] == 1, "the write must still happen, just not inline"

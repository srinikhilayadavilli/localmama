"""The workflow must be deterministic and must never skip a mandatory field."""

from __future__ import annotations

import pytest

from backend.app.languages import Language
from backend.app.models import ConversationState, SessionData
from backend.app.services.conversation_manager import ConversationManager
from backend.app.state_machine import missing_fields, next_state


def make_session(**kwargs) -> SessionData:
    return SessionData(session_id="test-session", **kwargs)


def test_happy_path_order():
    session = make_session(state=ConversationState.WELCOME)
    assert next_state(session) is ConversationState.LANGUAGE_SELECTION

    session.selected_language = Language.TELUGU
    session.state = ConversationState.LANGUAGE_SELECTION
    assert next_state(session) is ConversationState.ASK_NAME

    session.user_name = "Ravi"
    session.state = ConversationState.ASK_NAME
    assert next_state(session) is ConversationState.ASK_SERVICE

    session.requested_service = "electrician"
    session.state = ConversationState.ASK_SERVICE
    assert next_state(session) is ConversationState.ASK_LOCATION

    session.city_or_area = "Madhapur Hyderabad"
    session.state = ConversationState.ASK_LOCATION
    # The read-back sits here now: nothing is committed until the caller agrees.
    assert next_state(session) is ConversationState.REVIEW

    session.state = ConversationState.REVIEW
    assert next_state(session) is ConversationState.CONFIRMATION


def test_skips_states_already_satisfied():
    """A caller who answers everything at once should not be re-asked."""
    session = make_session(
        state=ConversationState.LANGUAGE_SELECTION,
        selected_language=Language.ENGLISH,
        user_name="Ravi",
        requested_service="plumber",
    )
    assert next_state(session) is ConversationState.ASK_LOCATION


def test_never_confirms_with_missing_field():
    session = make_session(
        state=ConversationState.ASK_LOCATION,
        selected_language=Language.HINDI,
        requested_service="plumber",
        city_or_area="Pune",
    )
    # user_name is missing, so we must go back and collect it.
    session.state = ConversationState.REVIEW
    assert next_state(session) is ConversationState.ASK_NAME
    assert "user_name" in missing_fields(session)


@pytest.mark.asyncio
async def test_full_conversation_english():
    manager = ConversationManager(make_session())
    opening = manager.start()
    assert opening.state is ConversationState.LANGUAGE_SELECTION
    assert "Local Mama" in opening.reply

    result = await manager.handle("English")
    assert result.state is ConversationState.ASK_NAME

    result = await manager.handle("My name is Ravi")
    assert manager.session.user_name == "Ravi"
    assert result.state is ConversationState.ASK_SERVICE

    result = await manager.handle("I need an electrician")
    assert manager.session.requested_service == "electrician"
    assert result.state is ConversationState.ASK_LOCATION

    result = await manager.handle("Madhapur Hyderabad")
    assert result.state is ConversationState.REVIEW

    result = await manager.handle("yes that's right")
    assert result.state is ConversationState.COMPLETED
    assert result.is_final
    assert result.lead is not None
    assert result.lead.user_name == "Ravi"
    assert result.lead.requested_service == "electrician"
    assert "Hyderabad" in result.lead.city_or_area


@pytest.mark.asyncio
async def test_multi_field_utterance_advances_past_filled_states():
    manager = ConversationManager(make_session())
    manager.start()
    await manager.handle("Telugu")
    result = await manager.handle("I am Ravi and I need an electrician in Hyderabad")

    session = manager.session
    assert session.user_name == "Ravi"
    assert session.requested_service == "electrician"
    assert session.city_or_area is not None
    assert result.state is ConversationState.REVIEW

    result = await manager.handle("avunu")   # "yes" in Telugu
    assert result.state is ConversationState.COMPLETED


@pytest.mark.asyncio
async def test_unclear_answer_reprompts_and_does_not_advance():
    manager = ConversationManager(make_session())
    manager.start()
    await manager.handle("Hindi")
    assert manager.session.state is ConversationState.ASK_NAME

    result = await manager.handle("...")
    assert result.state is ConversationState.ASK_NAME
    assert manager.session.user_name is None
    assert manager.session.retries(ConversationState.ASK_NAME) == 1


@pytest.mark.asyncio
async def test_unrecognised_language_reprompts():
    manager = ConversationManager(make_session())
    manager.start()
    result = await manager.handle("Klingon please")
    assert result.state is ConversationState.LANGUAGE_SELECTION
    assert manager.session.selected_language is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "spoken,expected",
    [
        ("Hindi", Language.HINDI),
        ("हिंदी", Language.HINDI),
        ("తెలుగు", Language.TELUGU),
        ("bangla", Language.BENGALI),
        ("I would like Tamil please", Language.TAMIL),
        ("kannada", Language.KANNADA),
    ],
)
async def test_language_locks_and_replies_in_that_language(spoken, expected):
    manager = ConversationManager(make_session())
    manager.start()
    result = await manager.handle(spoken)
    assert manager.session.selected_language is expected
    assert result.language is expected
    # The follow-up question must be rendered in the chosen language.
    from backend.app.prompts.messages import MessageKey, get_message

    assert result.reply == get_message(MessageKey.ASK_NAME, expected)


@pytest.mark.asyncio
async def test_misheard_service_is_requeried_then_corrected():
    """Regression from a live LiveKit call: STT heard 'puma' for 'plumber'."""
    manager = ConversationManager(make_session())
    manager.start()
    await manager.handle("English")
    await manager.handle("My name is Ravi")

    result = await manager.handle("I need a puma")
    assert result.state is ConversationState.ASK_SERVICE      # asked again
    assert manager.session.requested_service is None          # nothing committed

    result = await manager.handle("I need a plumber")
    assert manager.session.requested_service == "plumber"
    assert result.state is ConversationState.ASK_LOCATION


@pytest.mark.asyncio
async def test_genuinely_novel_service_is_accepted_on_retry():
    """The catalog cannot list every trade, so persistence must win."""
    manager = ConversationManager(make_session())
    manager.start()
    await manager.handle("English")
    await manager.handle("My name is Ravi")

    await manager.handle("I need a chimney sweep")
    result = await manager.handle("I need a chimney sweep")
    assert manager.session.requested_service == "chimney sweep"
    assert result.state is ConversationState.ASK_LOCATION


@pytest.mark.asyncio
async def test_advance_and_commit_never_disagree():
    """State must not move forward while the field stays empty."""
    manager = ConversationManager(make_session())
    manager.start()
    await manager.handle("English")
    await manager.handle("My name is Ravi")
    for utterance in ("I need a puma", "I need a puma"):
        result = await manager.handle(utterance)
        if result.state is not ConversationState.ASK_SERVICE:
            assert manager.session.requested_service is not None

"""Returning-caller memory: shorter calls, and no raw identifiers on disk.

Two things are being asserted. First that memory actually pays for itself — a
known caller skips the questions we already have answers to. Second that it
does so without becoming a privacy liability: the phone number is a lookup key,
never stored, and erasure works.
"""

from __future__ import annotations

import pytest

from backend.app import caller_profiles
from backend.app.languages import Language
from backend.app.models import ConversationState
from backend.app.services.conversation_manager import ConversationManager

PHONE = "+919876543210"
OTHER = "+919000000001"


@pytest.fixture(autouse=True)
def isolated_profiles(tmp_path, monkeypatch):
    """Point profile storage at a temp dir so tests never touch real data."""
    profiles = tmp_path / "profiles"
    monkeypatch.setattr(caller_profiles, "_profiles_dir", lambda: profiles)
    return profiles


async def complete_call(caller_id, turns):
    m = ConversationManager(caller_id=caller_id)
    m.start()
    result = None
    for text in turns:
        result = await m.handle(text)
    return m, result


FIRST_CALL = ["Telugu", "నా పేరు రవి", "electrician", "Madhapur Hyderabad", "avunu"]


# --- the payoff ----------------------------------------------------------


@pytest.mark.asyncio
async def test_returning_caller_skips_language_and_name():
    await complete_call(PHONE, FIRST_CALL)

    m = ConversationManager(caller_id=PHONE)
    result = m.start()

    # Straight to the only thing we cannot know in advance.
    assert result.state is ConversationState.ASK_SERVICE
    assert m.session.selected_language is Language.TELUGU
    assert m.session.user_name == "రవి"


@pytest.mark.asyncio
async def test_return_call_needs_fewer_turns():
    _, first = await complete_call(PHONE, FIRST_CALL)
    assert first.state is ConversationState.COMPLETED

    m, second = await complete_call(PHONE, ["plumber", "Madhapur Hyderabad", "avunu"])
    assert second.state is ConversationState.COMPLETED
    assert second.lead.user_name == "రవి"
    assert second.lead.requested_service == "plumber"
    assert second.lead.selected_language is Language.TELUGU


@pytest.mark.asyncio
async def test_returning_greeting_is_in_the_remembered_language():
    await complete_call(PHONE, FIRST_CALL)
    m = ConversationManager(caller_id=PHONE)
    reply = m.start().reply
    assert "రవి" in reply                      # greeted by name
    assert "మామా" in reply or "మామి" in reply   # and in Telugu, not English


@pytest.mark.asyncio
async def test_returning_caller_is_not_greeted_as_a_stranger():
    """'Welcome back, Ravi! Nice to meet you, Ravi!' reads as amnesia."""
    await complete_call(PHONE, FIRST_CALL)
    m = ConversationManager(caller_id=PHONE)
    reply = m.start().reply
    assert reply.count("రవి") == 1


# --- service and area are never assumed ----------------------------------


@pytest.mark.asyncio
async def test_previous_service_is_never_prefilled():
    """The service is *why* they are calling — assuming it would be wrong."""
    await complete_call(PHONE, FIRST_CALL)
    m = ConversationManager(caller_id=PHONE)
    m.start()
    assert m.session.requested_service is None


@pytest.mark.asyncio
async def test_previous_area_is_never_prefilled():
    """People move, and may want service at a different address."""
    await complete_call(PHONE, FIRST_CALL)
    m = ConversationManager(caller_id=PHONE)
    m.start()
    assert m.session.city_or_area is None


# --- isolation and first contact -----------------------------------------


@pytest.mark.asyncio
async def test_unknown_caller_gets_the_full_flow():
    m = ConversationManager(caller_id=OTHER)
    result = m.start()
    assert result.state is ConversationState.LANGUAGE_SELECTION
    assert m.session.user_name is None


@pytest.mark.asyncio
async def test_no_caller_id_means_no_memory():
    """Anonymous transports (browser demo) behave exactly as before."""
    await complete_call(PHONE, FIRST_CALL)
    m = ConversationManager(caller_id=None)
    assert m.start().state is ConversationState.LANGUAGE_SELECTION


@pytest.mark.asyncio
async def test_one_callers_memory_does_not_leak_to_another():
    await complete_call(PHONE, FIRST_CALL)
    m = ConversationManager(caller_id=OTHER)
    m.start()
    assert m.session.user_name is None
    assert m.session.selected_language is None


# --- privacy -------------------------------------------------------------


@pytest.mark.asyncio
async def test_phone_number_is_never_written_to_disk(isolated_profiles):
    await complete_call(PHONE, FIRST_CALL)
    files = list(isolated_profiles.glob("*.json"))
    assert files
    digits = PHONE.lstrip("+")
    for f in files:
        assert digits not in f.name
        assert digits not in f.read_text(encoding="utf-8")


def test_caller_key_is_stable_and_distinct():
    assert caller_profiles.caller_key(PHONE) == caller_profiles.caller_key(PHONE)
    assert caller_profiles.caller_key(PHONE) != caller_profiles.caller_key(OTHER)


def test_changing_the_salt_orphans_existing_profiles(monkeypatch):
    """Rotating the salt is the bulk-forget lever."""
    before = caller_profiles.caller_key(PHONE)
    monkeypatch.setattr(caller_profiles, "_salt", lambda: "new-salt")
    assert caller_profiles.caller_key(PHONE) != before


@pytest.mark.asyncio
async def test_forget_erases_the_profile():
    await complete_call(PHONE, FIRST_CALL)
    assert caller_profiles.load(PHONE) is not None

    assert caller_profiles.forget(PHONE) is True
    assert caller_profiles.load(PHONE) is None

    m = ConversationManager(caller_id=PHONE)
    assert m.start().state is ConversationState.LANGUAGE_SELECTION


@pytest.mark.asyncio
async def test_expired_profiles_are_dropped_on_read():
    await complete_call(PHONE, FIRST_CALL)
    profile = caller_profiles.load(PHONE)
    assert profile is not None

    from datetime import timedelta

    stale = profile.last_seen - timedelta(days=caller_profiles.PROFILE_RETENTION_DAYS + 1)
    profile.last_seen = stale
    path = caller_profiles._profiles_dir() / f"{profile.caller_key}.json"
    path.write_text(profile.model_dump_json(), encoding="utf-8")

    assert caller_profiles.load(PHONE) is None
    assert not path.exists()


@pytest.mark.asyncio
async def test_memory_can_be_disabled_entirely(monkeypatch):
    monkeypatch.setattr(caller_profiles, "_enabled", lambda: False)
    await complete_call(PHONE, FIRST_CALL)
    assert caller_profiles.count() == 0


@pytest.mark.asyncio
async def test_partial_call_does_not_erase_a_good_profile():
    """An abandoned call must not wipe what we already knew."""
    await complete_call(PHONE, FIRST_CALL)
    m = ConversationManager(caller_id=PHONE)
    m.start()
    await m.handle("...")  # caller says nothing useful, then hangs up

    profile = caller_profiles.load(PHONE)
    assert profile.name == "రవి"
    assert profile.preferred_language is Language.TELUGU

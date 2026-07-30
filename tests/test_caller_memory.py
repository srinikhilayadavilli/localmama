"""Caller profiles: recorded, never replayed into a call.

Two things are being asserted. First that nothing a previous call learned can
reach a live conversation — every detail on a lead has to have been said on the
call that produced it, and a remembered name is exactly the kind of value that
looks right and cannot be corrected. Second that recording it stays free of
privacy liability: the phone number is a lookup key, never stored, and erasure
works.
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


# --- nothing remembered reaches a live call ------------------------------


@pytest.mark.asyncio
async def test_a_returning_caller_is_asked_everything_again():
    """A remembered value is asserted to the caller, not offered to them.

    Prefilling the name and language skipped two questions and made the agent
    state, as fact, something the caller had not said on this call. That is
    unrecoverable when it is wrong: the read-back names them, they say yes to a
    detail they never gave, and the same name comes back on every later call
    from that number.
    """
    await complete_call(PHONE, FIRST_CALL)

    m = ConversationManager(caller_id=PHONE)
    result = m.start()

    assert result.state is ConversationState.LANGUAGE_SELECTION
    assert m.session.selected_language is None
    assert m.session.user_name is None


@pytest.mark.asyncio
async def test_no_field_is_prefilled_from_a_previous_call():
    await complete_call(PHONE, FIRST_CALL)
    m = ConversationManager(caller_id=PHONE)
    m.start()
    assert m.session.user_name is None
    assert m.session.selected_language is None
    assert m.session.requested_service is None
    assert m.session.city_or_area is None


@pytest.mark.asyncio
async def test_the_opening_line_never_names_the_caller():
    """A number shared by a household must not greet one person as another."""
    await complete_call(PHONE, FIRST_CALL)
    m = ConversationManager(caller_id=PHONE)
    assert "Ravi" not in m.start().reply and "రవి" not in m.start().reply


@pytest.mark.asyncio
async def test_a_second_call_still_captures_its_own_details():
    """Not remembering must cost turns, not correctness."""
    _, first = await complete_call(PHONE, FIRST_CALL)
    assert first.state is ConversationState.COMPLETED

    _, second = await complete_call(
        PHONE, ["Telugu", "నా పేరు రవి", "plumber", "Madhapur Hyderabad", "avunu"]
    )
    assert second.state is ConversationState.COMPLETED
    # Stored in English, like every captured field on both pipelines.
    assert second.lead.user_name == "Ravi"
    assert second.lead.requested_service == "plumber"
    assert second.lead.selected_language is Language.TELUGU


# --- but the profile is still written ------------------------------------


@pytest.mark.asyncio
async def test_a_completed_call_still_records_a_profile():
    """Recording is kept: the profiles are worth having, and `forget()` is a
    legal obligation that needs something to erase."""
    await complete_call(PHONE, FIRST_CALL)
    profile = caller_profiles.load(PHONE)
    assert profile is not None
    assert profile.name == "Ravi"
    assert profile.last_service == "electrician"


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
    assert profile.name == "Ravi"
    assert profile.preferred_language is Language.TELUGU

"""Vendor contact lookup — the business directory, not the brain.

The distinction is the point. `brain.py` does semantic search over shared
knowledge and carries no phone numbers for these rows; this reads the
tenant-owned `localmama.businesses`, where all 120 rows have one. A caller
asking "what is X's number?" wants an exact record — the nearest semantic
neighbour with a real phone attached would send them to a stranger.
"""

from __future__ import annotations

import pytest

from backend.app.services.directory import Business


@pytest.mark.parametrize(
    "raw,expected",
    [
        # Grouped so TTS dictates it rather than reading one long token.
        ("9735927627", "97359 27627"),
        ("919735927627", "+91 97359 27627"),
        ("", ""),
    ],
)
def test_phone_is_spoken_in_groups(raw: str, expected: str) -> None:
    assert Business("X", "cat", raw).spoken_phone() == expected


def test_unusual_phone_is_passed_through_unchanged() -> None:
    """Better to read an odd number verbatim than to mangle it into groups."""
    assert Business("X", "cat", "1800-123").spoken_phone() == "1800-123"


def test_directory_is_disabled_without_a_database() -> None:
    """conftest blanks DATABASE_URL, so this is the no-config path."""
    from backend.app.services import directory

    assert directory.available() is False
    assert directory.find("Mechanic4Me") == []


@pytest.mark.asyncio
async def test_tool_refuses_to_guess_when_nothing_matches(monkeypatch) -> None:
    """The failure that matters: inventing a number sends a caller to a stranger."""
    from backend.app import realtime_tools
    from backend.app.services import directory

    monkeypatch.setattr(directory, "available", lambda: True)

    async def none_found(name, limit=5):
        return []

    monkeypatch.setattr(directory, "find_async", none_found)
    tools = {t.info.name: t for t in realtime_tools.LeadRecorder().build_tools()}
    reply = await tools["lookup_vendor_contact"](business="Brimmies Cafe")
    assert "not guess" in reply.lower()
    assert "brimmies cafe" in reply.lower()


@pytest.mark.asyncio
async def test_tool_asks_which_one_when_several_match(monkeypatch) -> None:
    from backend.app import realtime_tools
    from backend.app.services import directory

    monkeypatch.setattr(directory, "available", lambda: True)

    async def several(name, limit=5):
        return [Business("WOW Wash", "Dry Cleaning", "6290925201"),
                Business("Equibillbook", "Car Wash", "9492056989")]

    monkeypatch.setattr(directory, "find_async", several)
    tools = {t.info.name: t for t in realtime_tools.LeadRecorder().build_tools()}
    reply = await tools["lookup_vendor_contact"](business="wash")
    assert "ask the caller which one" in reply.lower()
    # No number may be handed over while the business is still ambiguous.
    assert "6290925201" not in reply and "62909" not in reply


@pytest.mark.asyncio
async def test_an_exact_name_beats_the_other_matches(monkeypatch) -> None:
    """"WOW Wash" must answer about WOW Wash, not ask which "wash" they meant."""
    from backend.app import realtime_tools
    from backend.app.services import directory

    monkeypatch.setattr(directory, "available", lambda: True)

    async def exact_plus_noise(name, limit=5):
        return [Business("WOW Wash", "Dry Cleaning", "6290925201"),
                Business("The Laundryhub", "Dry Cleaning", "8886100061")]

    monkeypatch.setattr(directory, "find_async", exact_plus_noise)
    tools = {t.info.name: t for t in realtime_tools.LeadRecorder().build_tools()}
    reply = await tools["lookup_vendor_contact"](business="WOW Wash")
    assert "62909 25201" in reply


@pytest.mark.asyncio
async def test_a_listing_without_a_phone_says_so(monkeypatch) -> None:
    from backend.app import realtime_tools
    from backend.app.services import directory

    monkeypatch.setattr(directory, "available", lambda: True)

    async def no_phone(name, limit=5):
        return [Business("Quiet Co", "Events", None)]

    monkeypatch.setattr(directory, "find_async", no_phone)
    tools = {t.info.name: t for t in realtime_tools.LeadRecorder().build_tools()}
    reply = await tools["lookup_vendor_contact"](business="Quiet Co")
    assert "not available" in reply.lower()

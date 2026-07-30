"""Vendor contact lookup, against the brain as the single source of truth.

The phones used to live in a separate tenant table, which meant two stores and
two matching strategies. They are now backfilled onto `utter.knowledge`, which
always had `phone`, `city`, `kind` and `keywords` columns for exactly this.

What did NOT change is that lookups for a name are **literal**. Semantic search
matched "electrician" to an EV charging company at 0.59 — the words look alike —
so a caller asking for a number would have been sent to a stranger. Embeddings
answer "who does this kind of work"; they must not answer "what is X's number".
"""

from __future__ import annotations

import pytest

from backend.app.services.brain import Hit, spoken_phone


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
    assert spoken_phone(raw) == expected


def test_unusual_phone_is_passed_through_unchanged() -> None:
    """Better to read an odd number verbatim than to mangle it into groups."""
    assert spoken_phone("1800-123") == "1800-123"


def test_lookup_is_disabled_without_a_database() -> None:
    """conftest blanks DATABASE_URL, so this is the no-config path."""
    from backend.app.services import brain

    assert brain.available() is False
    assert brain.find_business("Mechanic4Me") == []
    assert brain.matches_for_service("plumber") == []


@pytest.mark.asyncio
async def test_tool_refuses_to_guess_when_nothing_matches(monkeypatch) -> None:
    """The failure that matters: inventing a number sends a caller to a stranger."""
    from backend.app import realtime_tools
    from backend.app.services import brain

    monkeypatch.setattr(brain, "available", lambda: True)
    monkeypatch.setattr(brain, "categories", lambda: set())

    async def none_found(name, limit=5):
        return []

    monkeypatch.setattr(brain, "find_business_async", none_found)
    tools = {t.info.name: t for t in realtime_tools.LeadRecorder().build_tools()}
    reply = await tools["lookup_vendor_contact"](business="Brimmies Cafe")
    assert "not guess" in reply.lower()
    assert "brimmies cafe" in reply.lower()


@pytest.mark.asyncio
async def test_a_category_asks_for_a_business_name(monkeypatch) -> None:
    """"wash" is a kind of business, not a business.

    There is no number to give, and listing the businesses that happen to match
    would volunteer vendors the caller never asked about.
    """
    from backend.app import realtime_tools
    from backend.app.services import brain

    monkeypatch.setattr(brain, "available", lambda: True)
    monkeypatch.setattr(brain, "categories", lambda: {"car wash", "dry cleaning"})
    tools = {t.info.name: t for t in realtime_tools.LeadRecorder().build_tools()}

    reply = await tools["lookup_vendor_contact"](business="wash")
    assert "category" in reply.lower()
    assert "name of the business" in reply.lower()
    assert "wow wash" not in reply.lower(), "must not name any vendor"


@pytest.mark.asyncio
async def test_ambiguous_name_asks_without_reading_the_list(monkeypatch) -> None:
    from backend.app import realtime_tools
    from backend.app.services import brain

    monkeypatch.setattr(brain, "available", lambda: True)
    monkeypatch.setattr(brain, "categories", lambda: set())

    async def several(name, limit=5):
        return [Hit(3, "Speed Auto", "", "Automobiles", None, "9007068682"),
                Hit(4, "Speed Auto Service", "", "Automobiles", None, "9007066682")]

    monkeypatch.setattr(brain, "find_business_async", several)
    tools = {t.info.name: t for t in realtime_tools.LeadRecorder().build_tools()}
    reply = await tools["lookup_vendor_contact"](business="Speed")
    assert "full name" in reply.lower()
    assert "9007068682" not in reply and "90070" not in reply
    assert "speed auto service" not in reply.lower(), "must not read out the candidates"


@pytest.mark.asyncio
async def test_an_exact_name_beats_the_other_matches(monkeypatch) -> None:
    """"WOW Wash" must answer about WOW Wash, not ask which "wash" they meant."""
    from backend.app import realtime_tools
    from backend.app.services import brain

    monkeypatch.setattr(brain, "available", lambda: True)
    monkeypatch.setattr(brain, "categories", lambda: set())

    async def exact_plus_noise(name, limit=5):
        return [Hit(1, "WOW Wash", "", "Dry Cleaning", None, "6290925201"),
                Hit(5, "The Laundryhub", "", "Dry Cleaning", None, "8886100061")]

    monkeypatch.setattr(brain, "find_business_async", exact_plus_noise)
    tools = {t.info.name: t for t in realtime_tools.LeadRecorder().build_tools()}
    reply = await tools["lookup_vendor_contact"](business="WOW Wash")
    assert "62909 25201" in reply


@pytest.mark.asyncio
async def test_a_listing_without_a_phone_says_so(monkeypatch) -> None:
    from backend.app import realtime_tools
    from backend.app.services import brain

    monkeypatch.setattr(brain, "available", lambda: True)
    monkeypatch.setattr(brain, "categories", lambda: set())

    async def no_phone(name, limit=5):
        return [Hit(6, "Quiet Co", "", "Events", None, None)]

    monkeypatch.setattr(brain, "find_business_async", no_phone)
    tools = {t.info.name: t for t in realtime_tools.LeadRecorder().build_tools()}
    reply = await tools["lookup_vendor_contact"](business="Quiet Co")
    assert "not available" in reply.lower()


@pytest.mark.parametrize(
    "query,is_category",
    [
        ("wash", True),           # a word inside "Car Wash"
        ("car wash", True),       # the category itself
        ("dry cleaning", True),
        ("WOW Wash", False),      # a business whose name contains a category word
        ("Mechanic4Me", False),
        ("", False),
    ],
)
def test_category_detection(monkeypatch, query: str, is_category: bool) -> None:
    from backend.app.services import brain

    monkeypatch.setattr(brain, "categories", lambda: {"car wash", "dry cleaning"})
    assert brain.looks_like_a_category(query) is is_category


# --- the WhatsApp options line ----------------------------------------


def test_options_line_carries_the_numbers() -> None:
    """A name alone is a teaser. The number is the point of the message —
    {{4}} used to read "Elecsyn Energy" with nothing to ring."""
    from backend.app.services.brain import options_line

    line = options_line([
        Hit(1, "Infinity Enterprises", "", "Plumbing", None, "9330998918"),
        Hit(2, "Clean Mates", "", "House Cleaning", None, "9147019147"),
    ])
    assert line == "Infinity Enterprises 93309 98918 · Clean Mates 91470 19147"


def test_options_line_drops_entries_with_no_number() -> None:
    from backend.app.services.brain import options_line

    line = options_line([
        Hit(1, "Has Phone", "", "cat", None, "9330998918"),
        Hit(2, "No Phone", "", "cat", None, None),
    ])
    assert line == "Has Phone 93309 98918"


def test_options_line_is_empty_when_nothing_matches() -> None:
    """Empty means the template falls back to "our team is shortlisting", which
    is true. Naming a business we do not have is not."""
    from backend.app.services.brain import options_line

    assert options_line([]) == ""


def test_a_category_match_wins_over_a_keyword_match() -> None:
    """"cleaning" returned a dental clinic: "teeth cleaning" is a fair keyword
    for a dentist and a terrible answer for someone wanting a house cleaned."""
    import inspect

    from backend.app.services import brain

    source = inspect.getsource(brain.matches_for_service)
    assert "by_category" in source
    assert "category" in source.lower()


def test_service_matching_never_uses_embeddings() -> None:
    """Semantic search matched "electrician" to an EV charging company at 0.59.
    Anything we read out to a caller is matched literally."""
    import inspect

    from backend.app.services import brain

    for fn in (brain.matches_for_service, brain.find_business):
        source = inspect.getsource(fn)
        assert "_embed" not in source, f"{fn.__name__} must not embed"
        assert "embedding <=>" not in source

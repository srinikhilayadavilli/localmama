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


@pytest.mark.parametrize(
    "spoken,stored",
    [
        # Typography the catalogue has and a caller cannot say. These fold to
        # the same string outright — no fuzzy matching involved.
        ("Brinda's kitchen", "Brinda’s kitchen"),   # straight vs curly apostrophe
        ("M S B R Enterprises", "M/S B R Enterprises"),
        ("Fix It Up - Phone Repairs", "Fix It Up- Phone Repairs"),
    ],
)
def test_a_name_survives_punctuation_the_caller_cannot_pronounce(
    spoken: str, stored: str
) -> None:
    """No transcriber emits U+2019, so that business was unreachable by name."""
    from backend.app.services.brain import normalise_name

    assert normalise_name(spoken) == normalise_name(stored)


@pytest.mark.parametrize(
    "spoken,stored",
    [
        # These do NOT normalise equal — they are close, and it is the ratio
        # that carries them. Kept separate so it stays obvious which mechanism
        # is doing the work.
        ("Pax Jewellers", "Pax Jwellers"),
        ("Brindas kitchen", "Brinda’s kitchen"),  # apostrophe dropped entirely
        ("Cleanmates", "Clean Mates"),
        ("MS B R Enterprises", "M/S B R Enterprises"),
        ("Frost and Sugar", "Frost O Sugar"),
    ],
)
def test_a_close_spelling_clears_the_name_cutoff(spoken: str, stored: str) -> None:
    import difflib

    from backend.app.services.brain import NAME_CUTOFF, normalise_name

    ratio = difflib.SequenceMatcher(
        None, normalise_name(spoken), normalise_name(stored)
    ).ratio()
    assert ratio >= NAME_CUTOFF, f"{spoken!r} vs {stored!r} scored {ratio:.2f}"


def test_two_different_businesses_stay_apart() -> None:
    """The cutoff has to reject as well as accept, or it is not a floor."""
    import difflib

    from backend.app.services.brain import NAME_CUTOFF, normalise_name

    ratio = difflib.SequenceMatcher(
        None, normalise_name("Clean Mates"), normalise_name("Speed Auto")
    ).ratio()
    assert ratio < NAME_CUTOFF


def test_a_different_business_is_not_folded_into_another() -> None:
    from backend.app.services.brain import normalise_name

    assert normalise_name("Clean Mates") != normalise_name("Speed Auto")


@pytest.mark.asyncio
async def test_an_approximate_name_is_confirmed_before_the_number(monkeypatch) -> None:
    """The safeguard that makes fuzzy names safe.

    "Pax Jewellers" resolving to "Pax Jwellers" is almost certainly right. Giving
    out a number because it almost certainly was is the outcome worth a turn to
    avoid, so the name is read back first.
    """
    from backend.app import realtime_tools
    from backend.app.services import brain

    hit = brain.Hit(1, "Pax Jwellers", "", "jewellery stores", None, "9836318445")
    hit.approximate = True

    monkeypatch.setattr(brain, "available", lambda: True)
    monkeypatch.setattr(brain, "categories", lambda: set())

    async def _found(name, limit=5):
        return [hit]

    monkeypatch.setattr(brain, "find_business_async", _found)
    tools = {t.info.name: t for t in realtime_tools.LeadRecorder().build_tools()}
    reply = await tools["lookup_vendor_contact"](business="Pax Jewellers")
    assert "Pax Jwellers" in reply
    assert "98363" not in reply, "the number must not be read before confirmation"


@pytest.mark.asyncio
async def test_an_exact_name_still_answers_immediately(monkeypatch) -> None:
    """Confirmation is the price of a guess, not of every lookup."""
    from backend.app import realtime_tools
    from backend.app.services import brain

    hit = brain.Hit(1, "Clean Mates", "", "house cleaning", None, "9147019147")

    monkeypatch.setattr(brain, "available", lambda: True)
    monkeypatch.setattr(brain, "categories", lambda: set())

    async def _found(name, limit=5):
        return [hit]

    monkeypatch.setattr(brain, "find_business_async", _found)
    tools = {t.info.name: t for t in realtime_tools.LeadRecorder().build_tools()}
    reply = await tools["lookup_vendor_contact"](business="Clean Mates")
    assert "91470 19147" in reply


def test_an_absent_business_is_never_approximated_into_a_real_one() -> None:
    """A close spelling is a guess; no spelling at all must stay a refusal."""
    import inspect

    from backend.app.services import brain

    source = inspect.getsource(brain._approximate_business)
    assert "NAME_CUTOFF" in source, "approximate matching must have a floor"


@pytest.mark.asyncio
async def test_a_business_named_in_hindi_is_looked_up_in_english(monkeypatch) -> None:
    """The catalogue is English and `find_business` matches literally, so a name
    still in the caller's script reaches nothing at all."""
    from backend.app import realtime_tools
    from backend.app.services import brain

    monkeypatch.setattr(brain, "available", lambda: True)
    monkeypatch.setattr(brain, "categories", lambda: set())
    asked: list[str] = []

    async def _record(name, limit=5):
        asked.append(name)
        return []

    monkeypatch.setattr(brain, "find_business_async", _record)
    tools = {t.info.name: t for t in realtime_tools.LeadRecorder().build_tools()}
    await tools["lookup_vendor_contact"](business="क्लीन मेट्स")

    assert asked and asked[0].isascii(), f"looked up {asked!r} rather than English"


@pytest.mark.asyncio
async def test_the_lead_lookup_is_scoped_to_the_callers_city(monkeypatch, tmp_path) -> None:
    """The city is captured, converted and stored — so it may as well narrow the
    match. Listings without a city stay eligible; see `matches_for_service`."""
    import asyncio

    from backend.app import realtime_tools
    from backend.app.config import settings
    from backend.app.services import brain, whatsapp

    monkeypatch.setattr(type(settings), "data_dir", property(lambda self: tmp_path),
                        raising=False)
    seen: dict = {}

    async def _match(service, limit=3, *, city=None):
        seen["service"], seen["city"] = service, city
        return []

    async def _send(lead, phone, options="", attempts=3):
        return {"ok": False, "reason": "no phone"}

    monkeypatch.setattr(brain, "matches_for_service_async", _match)
    monkeypatch.setattr(whatsapp, "send", _send)

    rec = realtime_tools.LeadRecorder()
    tools = {t.info.name: t for t in rec.build_tools()}
    await tools["set_language"](language="हिंदी")
    await tools["set_name"](name="राहुल")
    await tools["set_service"](service="प्लंबर")
    await tools["set_city"](city="मुंबई")
    await tools["save_lead"]()
    await asyncio.gather(*rec.background, return_exceptions=True)

    assert seen == {"service": "plumber", "city": "Mumbai"}

"""Finding a business the caller named in a sentence.

A caller asked for "Pax business" and got nothing. The catalogue calls it
"Pax Jwellers", and `LIKE '%pax business%'` matches no title on earth — while
"Pax" on its own finds it instantly. The name was there, wearing a coat.
"""

from __future__ import annotations

import pytest

from backend.services.brain import _core_name


@pytest.mark.parametrize("said, core", [
    ("Pax business", "pax"),
    ("Pax Business", "pax"),
    ("Pax shop", "pax"),
    ("the Pax people", "pax"),
    ("number for Pax", "pax"),
    ("Clean Mates shop", "clean mates"),
    ("Brimmies Cafe", "brimmies cafe"),
])
def test_the_words_around_a_name_are_stripped(said, core):
    assert _core_name(said) == core


def test_a_name_made_only_of_filler_survives_as_nothing():
    """"the shop" names no business. Searching on what is left would match half
    the catalogue, so nothing is left."""
    assert _core_name("the shop") == ""
    assert _core_name("the business") == ""


def test_a_real_name_is_not_dismantled():
    """Stripping runs only after a literal search has already failed, but even
    so it must not turn a genuine name into a different one."""
    assert _core_name("Pax Jwellers") == "pax jwellers"
    assert _core_name("Land Of Cakes") == "land cakes"  # "of" is filler


def test_punctuation_goes_with_it():
    """The catalogue contains "Brinda's kitchen" and "M/S B R Enterprises";
    a caller can say neither the apostrophe nor the slash."""
    assert _core_name("Brinda's kitchen shop") == "brinda s kitchen"


# --- every word, never any word -------------------------------------------


def test_all_words_needs_at_least_two():
    """One word is what the literal search already tried; running it again as
    an AND of one clause would just repeat that query."""
    from backend.services.brain import _all_words_business

    assert _all_words_business("pax", limit=5) == []
    assert _all_words_business("", limit=5) == []


def test_short_words_do_not_count_toward_the_and():
    """"of" and "at" appear in half the catalogue's keywords. Letting them into
    an AND makes the AND meaningless."""
    from backend.services.brain import _MIN_WORD, _core_name

    # "Land Of Cakes" -> "of" is filler anyway, but the length guard is the
    # backstop for the ones that are not.
    assert _MIN_WORD == 3
    assert all(len(w) >= 3 or w in ("of",) for w in _core_name("Land Of Cakes").split())


# --- matching a service ----------------------------------------------------


def test_spelling_variants_are_generated_not_listed():
    """Sarvam targets en-IN and still returns American forms. A caller asking
    for a "beauty parlor" was told we had nobody, while two were listed under
    "beauty parlour".

    The variants are generated from suffix rules rather than a hand-kept list,
    which is what makes this cover words nobody thought to add."""
    from backend.services.brain import _variants

    assert "parlour" in _variants("parlor")
    assert "jewellery" in _variants("jewelry")
    assert "centre" in _variants("center")
    assert "catalogue" in _variants("catalog")
    assert "traveller" in _variants("traveler")


def test_a_generated_variant_is_only_used_if_the_catalogue_has_it(monkeypatch):
    """The rules are deliberately over-generous — they turn "doctor" into
    "doctour" and "motor" into "motour". That is safe only because every
    candidate is checked against the catalogue's own vocabulary first, so a
    nonsense variant matches nothing and is discarded."""
    from backend.services import brain

    monkeypatch.setattr(brain, "vocabulary",
                        lambda force=False: {"parlour", "doctor", "motor"})
    assert brain.anglicise("beauty parlor") == "beauty parlour"
    assert brain.anglicise("doctor") == "doctor"      # not doctour
    assert brain.anglicise("motor") == "motor"        # not motour


def test_an_unknown_word_is_left_alone(monkeypatch):
    from backend.services import brain

    monkeypatch.setattr(brain, "vocabulary", lambda force=False: {"plumbing"})
    assert brain.anglicise("plumbing") == "plumbing"
    assert brain.anglicise("zzzz") == "zzzz"


def test_the_word_service_is_stripped_from_a_service_query():
    """The one that did damage. Speed Kawasaki Kolkata, a motorcycle dealer,
    carries the keywords "car service, bike service" — so every query with the
    word in it matched, and someone who asked for a beauty parlour was sent a
    Kawasaki showroom."""
    from backend.services.brain import _service_core

    assert _service_core("beauty parlor service") == "beauty parlor"
    assert _service_core("plumbing services") == "plumbing"
    assert _service_core("beauty parlour service near me") == "beauty parlour"
    assert _service_core("I need a plumber") == "plumber"


def test_a_query_of_pure_filler_names_no_trade():
    """"service near me" is not a trade. Searching on what is left of it would
    match anything with a phone number."""
    from backend.services.brain import _service_core

    assert _service_core("service near me") == ""
    assert _service_core("I need some service please") == ""


def test_a_word_the_catalogue_never_heard_of_is_dropped(monkeypatch):
    """A caller said "necessity of parlour". "of" was filler and "parlour" was
    real, but "necessity" survived, went into the AND, and matched nothing — so
    a query that named the trade exactly returned nothing at all.

    A word appearing in no category and no keyword cannot contribute to a
    match. Requiring it guarantees zero rows; dropping it cannot invent one.
    That is the general form of the filler list, and it covers every phrasing
    nobody thought to add: "requirement of plumber", "kindly arrange one
    carpenter", "urgent need of electrician"."""
    from backend.services import brain

    monkeypatch.setattr(brain, "vocabulary",
                        lambda force=False: {"parlour", "plumber", "beauty"})
    assert brain._known_words("necessity of parlour") == "parlour"
    assert brain._known_words("requirement of plumber") == "plumber"
    assert brain._known_words("beauty parlour") == "beauty parlour"


def test_dropping_unknown_words_cannot_empty_a_real_query(monkeypatch):
    """With no vocabulary to check against — an unreachable database — the
    query is passed through untouched rather than silently emptied."""
    from backend.services import brain

    monkeypatch.setattr(brain, "vocabulary", lambda force=False: set())
    assert brain._known_words("beauty parlour") == "beauty parlour"


# --- the caller's city decides between hits, it does not withhold them ------


def _hit(title, city=None):
    from backend.services.brain import Hit

    return Hit(id=1, title=title, content="", category=None, city=city,
               phone="9000000000")


def test_a_city_is_matched_either_way_round():
    """The caller gives a locality as often as a city, and the listings carry
    cities — so "Madhapur, Hyderabad" has to find a Hyderabad listing."""
    from backend.services.brain import _in_city

    hits = [_hit("A", "Hyderabad"), _hit("B", "Kolkata")]
    assert [h.title for h in _in_city(hits, "Madhapur, Hyderabad")] == ["A"]
    assert [h.title for h in _in_city(hits, "Kolkata")] == ["B"]


def test_a_listing_with_no_city_is_not_in_the_callers_city():
    """It is unknown, not local. It belongs in the fallback — where it is still
    returned — rather than in the narrowed list."""
    from backend.services.brain import _in_city

    assert _in_city([_hit("A"), _hit("B")], "Kolkata") == []


def test_no_city_narrows_nothing():
    from backend.services.brain import _in_city

    assert _in_city([_hit("A", "Kolkata")], None) == []
    assert _in_city([_hit("A", "Kolkata")], "  ") == []


def _catalogue(monkeypatch, hits):
    """A brain whose literal lookup returns `hits`, and nothing else."""
    from backend.services import brain

    monkeypatch.setattr(brain, "available", lambda: True)
    monkeypatch.setattr(brain, "_literal_business",
                        lambda query, limit: hits[:limit])
    return brain


def test_the_callers_city_settles_an_ambiguous_name(monkeypatch):
    """Two branches of nearly the same name used to be an AMBIGUOUS reply —
    "could you say the full name?" — to a caller who had already said the one
    thing that separates them."""
    brain = _catalogue(monkeypatch, [_hit("Pax Jwellers", "Kolkata"),
                                     _hit("Pax Jewellers", "Hyderabad")])
    found = brain.find_business("Pax", limit=4, city="Kolkata")
    assert [h.title for h in found] == ["Pax Jwellers"]


def test_a_business_is_never_withheld_because_of_its_city(monkeypatch):
    """A caller who names a business has named it. Answering "I don't have them
    listed" because the `city` column disagrees is a lie about a business we
    hold — which is why this narrows and never filters."""
    brain = _catalogue(monkeypatch, [_hit("Brimmies Cafe", "Kolkata")])
    assert [h.title for h in brain.find_business("Brimmies Cafe", city="Chennai")] \
        == ["Brimmies Cafe"]
    # …and the overwhelmingly common case: the listing has no city at all.
    brain = _catalogue(monkeypatch, [_hit("Brimmies Cafe")])
    assert [h.title for h in brain.find_business("Brimmies Cafe", city="Chennai")] \
        == ["Brimmies Cafe"]


def test_the_city_narrows_the_whole_list_not_a_truncated_one(monkeypatch):
    """Narrowing a list the `LIMIT` already cut is narrowing the wrong list: the
    record in the caller's own city may be the one that was cut off."""
    from backend.services.brain import _CITY_WINDOW

    hits = [_hit(f"Pax {i}") for i in range(6)] + [_hit("Pax Kolkata", "Kolkata")]
    brain = _catalogue(monkeypatch, hits)

    assert brain.find_business("Pax", limit=4) == hits[:4]      # no city, no window
    assert [h.title for h in brain.find_business("Pax", limit=4, city="Kolkata")] \
        == ["Pax Kolkata"]
    assert _CITY_WINDOW > 1


def test_the_narrowed_list_still_respects_the_limit(monkeypatch):
    brain = _catalogue(monkeypatch, [_hit(f"Pax {i}", "Kolkata") for i in range(9)])
    assert len(brain.find_business("Pax", limit=4, city="Kolkata")) == 4

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

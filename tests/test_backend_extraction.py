"""Rule extraction of the requested trade, with no call and no catalogue.

Everything here is about the fuzzy rescue, which is the one part of the
extractor that guesses. A guess that lands puts a trade in the lead the caller
never asked for, and it is read back at 0.85 — above `MIN_CONFIDENCE` — so
nothing re-prompts and nobody finds out until the WhatsApp goes.
"""

from __future__ import annotations

import pytest

from backend.services.entity_extractor import (
    UNKNOWN_SERVICE_CONFIDENCE,
    extract_service,
    fuzzy_match_service,
)


# --- the request frame is not the trade -----------------------------------


@pytest.mark.parametrize(
    "utterance",
    [
        "I am looking for mental health help",
        "looking for mental health",
        "searching for mental health",
        "I want a booking for mental health support",
    ],
)
def test_the_verb_of_a_request_is_never_the_trade(utterance):
    """"looking" is one character from "cooking" and exactly as long, so both
    the similarity and the length guard pass it — and every "I am looking for
    X" whose X the catalog does not list came out as `cook`."""
    service, confidence = extract_service(utterance, expecting=True)
    assert service not in {"cook", "tutor", "painter"}
    assert "mental health" in service
    assert confidence == UNKNOWN_SERVICE_CONFIDENCE


def test_asking_for_a_number_is_not_asking_for_a_plumber():
    """"number"/"plumber" is 0.77 with comparable length. A caller chasing a
    contact detail is not a caller with a leaking tap."""
    assert fuzzy_match_service("can I get the number please") is None


# --- …and the rescue still rescues ----------------------------------------


@pytest.mark.parametrize(
    ("utterance", "expected"),
    [
        ("plummer", "plumber"),
        ("I need an electrican", "electrician"),
        ("carpentar", "carpenter"),
        # The frame is stepped over word by word rather than abandoning the
        # utterance, so a garbled trade behind a frame word is still reached.
        ("I am looking for a plummer", "plumber"),
        ("searching for an electrican", "electrician"),
    ],
)
def test_a_garbled_trade_is_still_rescued(utterance, expected):
    service, confidence = extract_service(utterance, expecting=True)
    assert service == expected
    assert confidence >= 0.85


def test_a_real_cook_is_still_a_cook():
    """The guard removes words nobody names a trade with. "cooking" is not one
    of them."""
    service, _ = extract_service("I need someone for cooking", expecting=True)
    assert service == "cook"

"""Extraction must handle bare answers, patterned answers, code-mixing, and fillers."""

from __future__ import annotations

import pytest

from backend.app.models import ConversationState
from backend.app.services.entity_extractor import (
    extract,
    match_city,
    match_service,
)

NAME = ConversationState.ASK_NAME
SERVICE = ConversationState.ASK_SERVICE
LOCATION = ConversationState.ASK_LOCATION


@pytest.mark.parametrize(
    "text,expected",
    [
        ("My name is Ravi", "Ravi"),
        ("I am Priya Sharma", "Priya Sharma"),
        ("myself Anil", "Anil"),
        ("this is Lakshmi", "Lakshmi"),
        ("mera naam Suresh hai", "Suresh"),
        ("naa peru Ravi", "Ravi"),
        ("nanna hesaru Kiran", "Kiran"),
        ("amar nam Rahul", "Rahul"),
        ("en peyar Karthik", "Karthik"),
    ],
)
def test_name_patterns(text, expected):
    assert extract(text, expecting=NAME).name == expected


def test_bare_name_when_asked():
    assert extract("Ravi", expecting=NAME).name == "Ravi"


def test_bare_name_with_fillers():
    assert extract("umm haan ji Ravi", expecting=NAME).name == "Ravi"


def test_bare_phrase_not_treated_as_name_when_not_asked():
    """Without being asked, an ambiguous bare phrase must not become a name."""
    assert extract("Ravi", expecting=None).name is None


def test_service_word_is_never_a_name():
    result = extract("electrician", expecting=NAME)
    assert result.name is None


@pytest.mark.parametrize(
    "text,expected",
    [
        ("I need an electrician", "electrician"),
        ("plumber chahiye", "plumber"),
        ("looking for a carpenter", "carpenter"),
        ("AC repair", "ac repair"),
        ("washing machine is broken", "appliance repair"),
        ("बिजली", "electrician"),
        ("ప్లంబర్", "plumber"),
        ("I want a tutor", "tutor"),
    ],
)
def test_service_catalog(text, expected):
    assert extract(text, expecting=SERVICE).requested_service == expected


def test_longest_synonym_wins():
    """'ac repair' must beat the shorter 'ac' synonym."""
    assert match_service("I need ac repair") == "ac repair"


def test_unknown_service_still_extracted():
    """A trade absent from the catalog must still be captured, not dropped."""
    result = extract("I need a chimney sweep", expecting=SERVICE)
    assert result.requested_service is not None
    assert "chimney" in result.requested_service


def test_related_word_maps_to_catalog_category():
    """'chimney cleaner' is a cleaning job — catalog matching should catch it."""
    assert match_service("I need a chimney cleaner") == "cleaning"


@pytest.mark.parametrize(
    "text,expected_fragment",
    [
        ("Hyderabad", "Hyderabad"),
        ("in Bangalore", "Bangalore"),
        ("Madhapur Hyderabad", "Hyderabad"),
        ("Pune mein", "Pune"),
        ("Vizag lo", "Vizag"),
    ],
)
def test_location_extraction(text, expected_fragment):
    result = extract(text, expecting=LOCATION)
    assert result.city_or_area is not None
    assert expected_fragment.lower() in result.city_or_area.lower()


def test_unknown_locality_accepted_when_asked():
    result = extract("Indiranagar", expecting=LOCATION)
    assert result.city_or_area == "Indiranagar"


def test_city_seed_matching():
    assert match_city("I live in mumbai") == "Mumbai"
    assert match_city("nothing here") is None


def test_multiple_fields_in_one_utterance():
    result = extract(
        "I am Ravi and I need an electrician in Hyderabad", expecting=NAME
    )
    assert result.name == "Ravi"
    # Explicit high-confidence service/location survive the name-bias filter.
    assert result.requested_service == "electrician"
    assert result.city_or_area is not None


def test_empty_input():
    result = extract("", expecting=NAME)
    assert result.name is None
    assert result.confidence == 0.0
    assert result.source == "none"


def test_confidence_higher_for_explicit_pattern():
    explicit = extract("My name is Ravi", expecting=NAME)
    bare = extract("Ravi", expecting=NAME)
    assert explicit.confidence > bare.confidence


# --- speech-recognition noise (found by a real LiveKit call) --------------


@pytest.mark.parametrize(
    "garbled,expected",
    [
        ("I need a plummer", "plumber"),
        ("I need an electrican", "electrician"),
        ("I need a carpentar", "carpenter"),
    ],
)
def test_misheard_trade_names_are_rescued(garbled, expected):
    """STT mangles trade names constantly; near-misses should still resolve."""
    assert extract(garbled, expecting=SERVICE).requested_service == expected


@pytest.mark.parametrize("not_a_service", ["ravi", "hello", "chimney"])
def test_fuzzy_matching_does_not_invent_services(not_a_service):
    """A looser cutoff mapped 'ravi' -> maid. It must not guess that hard."""
    from backend.app.services.entity_extractor import fuzzy_match_service

    assert fuzzy_match_service(not_a_service) is None


def test_unrecognised_trade_is_below_the_acceptance_bar():
    """A real call produced 'I need a puma' (STT for 'plumber') and 'puma'
    reached the lead. It must now trigger a re-prompt instead."""
    from backend.app.services.entity_extractor import MIN_CONFIDENCE

    result = extract("I need a puma", expecting=SERVICE)
    assert result.requested_service == "puma"
    assert result.confidence < MIN_CONFIDENCE


@pytest.mark.parametrize(
    "fragment",
    ["I need a please", "I need a plumber please", "give me a service", "please help"],
)
def test_sentence_fragments_are_not_locations(fragment):
    """A live call captured city_or_area='I Need A' from STT noise at the
    location step. Fragments must be rejected, not Title-Cased into a city."""
    assert extract(fragment, expecting=LOCATION).city_or_area is None


@pytest.mark.parametrize(
    "place",
    ["Madhapur Hyderabad", "Indiranagar", "Pune mein", "in Bangalore", "माधापुर हैदराबाद"],
)
def test_real_places_survive_the_fragment_filter(place):
    assert extract(place, expecting=LOCATION).city_or_area is not None


@pytest.mark.parametrize(
    "short_garble",
    ["palm", "puma", "oil", "anil", "area", "ravi", "name", "priya"],
)
def test_short_words_are_never_fuzzy_matched(short_garble):
    """A live call transcribed 'plumber' as 'palm oil'; 'palm' then matched
    'salon' at 0.85 and the lead said salon. Short words carry no signal."""
    from backend.app.services.entity_extractor import fuzzy_match_service

    assert fuzzy_match_service(short_garble) is None


def test_palm_oil_does_not_become_a_service():
    from backend.app.services.entity_extractor import MIN_CONFIDENCE

    result = extract("I need a palm oil", expecting=SERVICE)
    assert result.confidence < MIN_CONFIDENCE   # re-prompt, do not guess


# --- from a real Telugu call --------------------------------------------


def test_telugu_sentence_yields_the_trade_not_the_whole_sentence():
    """A real call stored 'నేను ఎలక్ట్రిషన్ కోసం వెతుకుతున్నాను' (I am looking
    for an electrician) verbatim as the requested service."""
    result = extract("నేను ఎలక్ట్రిషన్ కోసం వెతుకుతున్నాను", expecting=SERVICE)
    assert result.requested_service == "electrician"


@pytest.mark.parametrize(
    "utterance,expected",
    [
        ("నాకు ప్లంబర్ కావాలి", "plumber"),
        ("मुझे बिजली वाला चाहिए", "electrician"),
        ("எனக்கு பிளம்பர் வேண்டும்", "plumber"),
        ("ನನಗೆ ಪ್ಲಂಬರ್ ಬೇಕು", "plumber"),
    ],
)
def test_native_script_request_forms(utterance, expected):
    assert extract(utterance, expecting=SERVICE).requested_service == expected


def test_indic_spelling_variants_are_rescued():
    """The caller said ఎలక్ట్రిషన్; the catalog has ఎలక్ట్రీషియన్ (0.83 similar)."""
    from backend.app.services.entity_extractor import fuzzy_match_service

    assert fuzzy_match_service("ఎలక్ట్రిషన్") == "electrician"


@pytest.mark.parametrize(
    "indic_name_or_place",
    ["ఫణి", "కుమార్", "రాజమండ్రి", "మాధాపూర్", "హైదరాబాద్", "కార్తీక్"],
)
def test_indic_names_and_places_are_not_fuzzy_matched_to_services(indic_name_or_place):
    from backend.app.services.entity_extractor import fuzzy_match_service

    assert fuzzy_match_service(indic_name_or_place) is None


def test_a_place_answered_at_the_service_step_is_not_a_service():
    """'Madhapur, Hyderabad' was stored as the service *and* the area."""
    result = extract("Madhapur, Hyderabad.", expecting=SERVICE)
    assert result.requested_service is None
    assert result.city_or_area is not None

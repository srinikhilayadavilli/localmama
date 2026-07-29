"""Language normalisation and message-template coverage."""

from __future__ import annotations

import pytest

from backend.app.languages import (
    ENDONYM,
    LOCALE,
    Language,
    detect_script_language,
    match_language,
    normalize_text,
    resolve_language,
)
from backend.app.prompts.messages import MessageKey, get_message, missing_translations


@pytest.mark.parametrize(
    "text,expected",
    [
        ("English", Language.ENGLISH),
        ("english please", Language.ENGLISH),
        ("Hindi", Language.HINDI),
        ("हिंदी", Language.HINDI),
        ("हिन्दी", Language.HINDI),
        ("Bengali", Language.BENGALI),
        ("bangla", Language.BENGALI),
        ("বাংলা", Language.BENGALI),
        ("Telugu", Language.TELUGU),
        ("telegu", Language.TELUGU),
        ("తెలుగు", Language.TELUGU),
        ("Tamil", Language.TAMIL),
        ("tamizh", Language.TAMIL),
        ("தமிழ்", Language.TAMIL),
        ("Kannada", Language.KANNADA),
        ("ಕನ್ನಡ", Language.KANNADA),
        ("I want to speak in Telugu", Language.TELUGU),
    ],
)
def test_match_language(text, expected):
    assert match_language(text) is expected


def test_unknown_language_returns_none():
    assert match_language("Klingon") is None
    assert match_language("") is None


def test_script_detection():
    assert detect_script_language("मुझे बिजली वाला चाहिए") is Language.HINDI
    assert detect_script_language("నాకు ఎలక్ట్రీషియన్ కావాలి") is Language.TELUGU
    assert detect_script_language("எனக்கு தேவை") is Language.TAMIL
    # Latin script is ambiguous between English and romanised Indic — no guess.
    assert detect_script_language("mujhe chahiye") is None


def test_normalize_preserves_indic_script():
    assert "हिंदी" in normalize_text("हिंदी")
    assert normalize_text("  Hello,   World! ") == "hello world"


def test_every_message_translated_into_every_language():
    """A missing translation would fall back to English mid-call — catch it here."""
    assert missing_translations() == {}


def test_every_language_has_locale_and_endonym():
    for language in Language:
        assert language in LOCALE
        assert language in ENDONYM


@pytest.mark.parametrize("language", list(Language))
def test_templates_render_for_all_languages(language):
    for key in MessageKey:
        rendered = get_message(
            key, language, name="Ravi", service="electrician", location="Hyderabad"
        )
        assert rendered
        # Spoken output: no markdown, no leftover placeholders.
        assert "{" not in rendered and "}" not in rendered
        assert "*" not in rendered and "#" not in rendered


def test_confirmation_includes_captured_values():
    rendered = get_message(
        MessageKey.CONFIRMATION,
        Language.ENGLISH,
        name="Ravi",
        service="electrician",
        location="Madhapur",
    )
    assert "electrician" in rendered and "Madhapur" in rendered


# --- cross-script language names --------------------------------------
#
# Speech recognition with language auto-detection routinely writes Indic audio
# in the wrong script, Devanagari most often. A caller who said "తెలుగు" and was
# transcribed "तेलुगु" matched no alias, so script detection asserted HINDI —
# the call continued in Hindi, STT was pinned to `hi`, and every later Telugu
# turn came back as Devanagari too, stalling the workflow. Both "it answers in
# Hindi" and "it will not move on" came from this one gap.

@pytest.mark.parametrize(
    "text,expected",
    [
        # Telugu named in Devanagari, the observed failure.
        ("तेलुगु", Language.TELUGU),
        ("तेलुगू", Language.TELUGU),
        ("तेलगु", Language.TELUGU),
        # The rest of the six, same trap.
        ("तमिल", Language.TAMIL),
        ("कन्नड़", Language.KANNADA),
        ("बंगाली", Language.BENGALI),
        ("अंग्रेजी", Language.ENGLISH),
        ("इंग्लिश", Language.ENGLISH),
        # And named in Telugu script, for a recogniser biased the other way.
        ("హిందీ", Language.HINDI),
        ("ఇంగ్లీష్", Language.ENGLISH),
    ],
)
def test_language_named_in_another_script(text: str, expected: Language) -> None:
    assert resolve_language(text) is expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("मुझे प्लंबर चाहिए", Language.HINDI),
        ("నాకు ప్లంబర్ కావాలి", Language.TELUGU),
        ("আমার একজন প্লাম্বার দরকার", Language.BENGALI),
    ],
)
def test_script_detection_still_works_for_plain_speech(text: str, expected: Language) -> None:
    """The aliases must not break the caller who just starts talking."""
    assert resolve_language(text) is expected

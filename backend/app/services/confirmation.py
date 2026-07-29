"""Yes/no and correction detection for the read-back step.

Speech recognition is nondeterministic — the same audio produced "plumber" once
and "puma" another time on a live call — so the last defence before a lead is
committed is asking the caller whether we heard them correctly.

This module answers three narrow questions about one utterance:

  * is it agreement?
  * is it disagreement?
  * does it name a field the caller wants changed?

Like `smalltalk`, it is pure classification: it returns answers, never mutates
anything. The conversation manager decides what to do with them.
"""

from __future__ import annotations

from ..languages import normalize_text

# Matching is on the normalised utterance, so entries are in that form.
_AFFIRMATIVE = {
    # English
    "yes", "yeah", "yep", "yup", "correct", "right", "ok", "okay", "sure",
    "perfect", "exactly", "true", "fine", "good", "confirm", "confirmed",
    # Hindi
    "haan", "han", "ji", "sahi", "theek", "thik", "bilkul", "हाँ", "हां",
    "सही", "ठीक", "बिल्कुल",
    # Bengali
    "hyan", "haan", "thik", "thick", "হ্যাঁ", "ঠিক", "সঠিক",
    # Telugu
    "avunu", "avnu", "sare", "sari", "సరే", "అవును", "కరెక్ట్",
    # Tamil — romanisation varies a lot, so cover the common spellings
    "aam", "aama", "aamaam", "aamam", "ama", "amam", "sari", "seri",
    "ஆம்", "ஆமாம்", "சரி",
    # Kannada
    "houdu", "howdu", "haudu", "ಹೌದು", "ಸರಿ",
}

_NEGATIVE = {
    # English
    "no", "nope", "nah", "wrong", "incorrect", "change", "not", "isnt",
    # Hindi
    "nahi", "nahin", "galat", "नहीं", "नही", "ग़लत", "गलत",
    # Bengali
    "na", "bhul", "না", "ভুল",
    # Telugu
    "kadu", "thappu", "కాదు", "తప్పు",
    # Tamil
    "illai", "illa", "thavaru", "இல்லை", "தவறு",
    # Kannada
    "illa", "thappu", "ಇಲ್ಲ", "ತಪ್ಪು",
}

#: Field the caller says is wrong, mapped to the session attribute.
_FIELD_WORDS: dict[str, tuple[str, ...]] = {
    "user_name": (
        "name", "naam", "nam", "peru", "hesaru", "peyar",
        "नाम", "নাম", "పేరు", "பெயர்", "ಹೆಸರು",
    ),
    "requested_service": (
        "service", "seva", "sewa", "work", "job",
        "सेवा", "পরিষেবা", "సేవ", "சேவை", "ಸೇವೆ",
    ),
    "city_or_area": (
        "area", "city", "place", "location", "shahar", "sheher", "jagah",
        "ilaka", "nagar", "ooru",
        "शहर", "इलाका", "जगह", "এলাকা", "শহর",
        "ప్రాంతం", "నగరం", "ஊர்", "பகுதி", "நகரம்", "ಪ್ರದೇಶ", "ನಗರ",
    ),
}


def _tokens(text: str) -> set[str]:
    return set(normalize_text(text).split())


def is_affirmative(text: str) -> bool:
    """Whether the caller agreed with the read-back."""
    if not text:
        return False
    tokens = _tokens(text)
    # A negation anywhere wins: "yes, but no that's wrong" is not agreement.
    if tokens & _NEGATIVE:
        return False
    return bool(tokens & _AFFIRMATIVE)


def is_negative(text: str) -> bool:
    """Whether the caller disagreed with the read-back."""
    if not text:
        return False
    return bool(_tokens(text) & _NEGATIVE)


def field_to_correct(text: str) -> str | None:
    """Which field the caller named as wrong, if any.

    Returns a `SessionData` attribute name, so the manager can clear it and
    re-ask without a second mapping table.
    """
    if not text:
        return None
    tokens = _tokens(text)
    for field, words in _FIELD_WORDS.items():
        for word in words:
            needle = normalize_text(word)
            if needle and needle in tokens:
                return field
    return None

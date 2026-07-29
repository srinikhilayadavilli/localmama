"""Language enum, normalisation, and script detection.

The workflow itself is language-agnostic: it only ever deals with a `Language`
value. Everything user-visible lives in `prompts/messages.py`, keyed by this
enum. Nothing in the state machine branches on language.
"""

from __future__ import annotations

import re
import unicodedata
from enum import Enum


class Language(str, Enum):
    ENGLISH = "english"
    HINDI = "hindi"
    BENGALI = "bengali"
    TELUGU = "telugu"
    TAMIL = "tamil"
    KANNADA = "kannada"


#: BCP-47 tags. Used for STT/TTS provider hints and the browser Web Speech API.
LOCALE: dict[Language, str] = {
    Language.ENGLISH: "en-IN",
    Language.HINDI: "hi-IN",
    Language.BENGALI: "bn-IN",
    Language.TELUGU: "te-IN",
    Language.TAMIL: "ta-IN",
    Language.KANNADA: "kn-IN",
}

def iso639_1(language: Language) -> str:
    """Bare two-letter language code, e.g. "bn" for Bengali.

    Derived from LOCALE rather than the enum name: `Language.BENGALI.value[:2]`
    is "be" and `KANNADA` is "ka", both of which are wrong and both of which a
    provider will either reject or silently transcribe as another language.
    """
    return LOCALE[language].split("-", 1)[0]


#: Human-readable name in the language itself, for UI display.
ENDONYM: dict[Language, str] = {
    Language.ENGLISH: "English",
    Language.HINDI: "हिंदी",
    Language.BENGALI: "বাংলা",
    Language.TELUGU: "తెలుగు",
    Language.TAMIL: "தமிழ்",
    Language.KANNADA: "ಕನ್ನಡ",
}

# Aliases the user may speak or type. Matching is done on a normalised
# (casefolded, accent-stripped, punctuation-free) form of the utterance, so
# entries here should also be written in that normalised form.
# Each language's name is listed in Latin, its own script, AND the other Indian
# scripts a recogniser is likely to write it in.
#
# That last group is not padding. Speech recognition running with language
# auto-detection routinely writes Indic audio in the wrong script — Devanagari
# most of all, because it is the best-represented Indian script in these models.
# A caller who says "తెలుగు" and gets transcribed "तेलुगु" used to match no
# alias at all, so `detect_script_language` saw Devanagari and asserted HINDI.
# The call then continued in Hindi, STT was pinned to `hi`, every later Telugu
# turn came back as Devanagari too, and the workflow stalled. One missing alias
# produced both "it answers in Hindi" and "it will not move on".
_ALIASES: dict[Language, tuple[str, ...]] = {
    Language.ENGLISH: (
        "english", "englis", "inglish", "angrezi", "angreji", "ingles", "eng",
        "अंग्रेजी", "अंग्रेज़ी", "इंग्लिश", "इंग्लिश",
        "ఇంగ్లీష్", "ఆంగ్లం", "ইংরেজি", "ஆங்கிலம்", "ಇಂಗ್ಲಿಷ್",
    ),
    Language.HINDI: (
        "hindi", "hindhi", "हिंदी", "हिन्दी", "हिंदि",
        "హిందీ", "হিন্দি", "இந்தி", "ಹಿಂದಿ",
    ),
    Language.BENGALI: (
        "bengali", "bangla", "bengoli", "bangali", "বাংলা", "বাঙলা",
        "बंगाली", "बांग्ला", "बांगला", "బెంగాలీ", "ಬಂಗಾಳಿ",
    ),
    Language.TELUGU: (
        "telugu", "telegu", "thelugu", "tel", "తెలుగు",
        "तेलुगु", "तेलुगू", "तेलगु", "तेलेगु",
        "তেলুগু", "தெலுங்கு", "ತೆಲುಗು",
    ),
    Language.TAMIL: (
        "tamil", "thamil", "tamizh", "thamizh", "தமிழ்",
        "तमिल", "तमिळ", "तामिल", "తమిళ్", "ತಮಿಳು", "তামিল",
    ),
    Language.KANNADA: (
        "kannada", "kannad", "canada", "kanada", "kannadam", "ಕನ್ನಡ",
        "कन्नड़", "कन्नड", "कन्नडा", "కన్నడ", "কন্নড়",
    ),
}

# Unicode block ranges per script. Used for detection when the user simply
# starts speaking a language rather than naming it.
_SCRIPT_RANGES: tuple[tuple[Language, int, int], ...] = (
    (Language.HINDI, 0x0900, 0x097F),   # Devanagari
    (Language.BENGALI, 0x0980, 0x09FF),
    (Language.TAMIL, 0x0B80, 0x0BFF),
    (Language.TELUGU, 0x0C00, 0x0C7F),
    (Language.KANNADA, 0x0C80, 0x0CFF),
)


def normalize_text(text: str) -> str:
    """Casefold, strip accents, and collapse punctuation/whitespace.

    Indic scripts are left intact — NFKD on Devanagari would split matras from
    their base consonants, so we only drop combining marks for Latin text.
    """
    text = text.strip().casefold()
    if text.isascii():
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^\w\sऀ-෿]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def match_language(text: str) -> Language | None:
    """Map a spoken/typed language name to a supported `Language`.

    Handles "Hindi", "हिंदी", "I want Telugu please", "let's do bangla", etc.
    Returns None when nothing matches, so the caller can re-prompt rather than
    guessing.
    """
    if not text:
        return None
    normalized = normalize_text(text)
    if not normalized:
        return None

    # Prefer a whole-word hit so "english" inside a longer sentence still wins
    # but "can" never matches "kannada".
    tokens = set(normalized.split())
    for language, aliases in _ALIASES.items():
        for alias in aliases:
            if alias in tokens:
                return language

    # Fall back to substring matching for aliases written in Indic script,
    # which may not be whitespace-separated from surrounding text.
    for language, aliases in _ALIASES.items():
        for alias in aliases:
            if len(alias) >= 4 and alias in normalized:
                return language

    return None


def detect_script_language(text: str) -> Language | None:
    """Guess the language from the Unicode script the user is writing in.

    Only meaningful for Indic scripts — Latin text is ambiguous between English
    and romanised Indian languages, so this returns None for it rather than
    asserting English.
    """
    if not text:
        return None
    counts: dict[Language, int] = {}
    for ch in text:
        code = ord(ch)
        for language, low, high in _SCRIPT_RANGES:
            if low <= code <= high:
                counts[language] = counts.get(language, 0) + 1
                break
    if not counts:
        return None
    return max(counts, key=lambda lang: counts[lang])


def resolve_language(text: str) -> Language | None:
    """Best-effort language resolution: explicit name first, then script."""
    return match_language(text) or detect_script_language(text)

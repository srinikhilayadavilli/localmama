"""Indic script to Latin letters, offline and deterministic.

Two jobs, both of which must work on a host with no Sarvam key and inside a
live call with no time to spare:

  * **Storage.** A lead's name, city and service are stored in English. Sarvam's
    `/transliterate` is better at this than any table (see `translate.py`), but
    it is a network call that can be slow, disabled, or down, and a name is not
    allowed to fall back to Devanagari — the lead, the WhatsApp message and the
    Postgres row would then hold a value nobody downstream can read.
  * **Grounding.** `grounding.py` compares what a tool was given against what
    the caller was transcribed saying. Those two arrive in different scripts all
    the time — the realtime model hands over "Rahul" while the transcript reads
    "राहुल" — and a comparison that cannot see through the script would reject
    the caller's own name as a hallucination.

Neither job needs scholarly accuracy. Grounding compares fuzzily, and a name is
read back to the caller before it is saved, so "haidarabad" for "हैदराबाद" is a
success: it is Latin, it is recognisable, and it is 0.86 similar to the spelling
a matcher wants.

**Why one table covers five scripts.** Devanagari, Bengali, Tamil, Telugu and
Kannada are all encoded on the ISCII layout: the same letter sits at the same
offset from the start of each block, so ka is +0x15 in every one of them. The
table is therefore keyed by offset, not by code point, and Tamil's missing
letters are simply unassigned holes we never look up.

Abugida, not alphabet: a bare consonant carries an inherent "a" which a vowel
sign replaces and a virama cancels. That is the whole algorithm.
"""

from __future__ import annotations

#: Start of each supported block. Order matters only for lookup speed.
_BLOCKS: tuple[tuple[int, str], ...] = (
    (0x0900, "devanagari"),
    (0x0980, "bengali"),
    (0x0B80, "tamil"),
    (0x0C00, "telugu"),
    (0x0C80, "kannada"),
)

#: Scripts that drop a word-final inherent vowel. "राहुल" is Rahul, not Rahula
#: — and a Telugu name really does keep it, which is why this is per script and
#: not a blanket rule: "రవి" must stay Ravi.
_DROPS_FINAL_SCHWA = {"devanagari", "bengali"}

_INHERENT = "a"

#: Independent vowels, at their ISCII offsets.
_VOWELS: dict[int, str] = {
    0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u",
    0x0B: "ri", 0x0C: "li", 0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai",
    0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au",
    0x60: "ri", 0x61: "li",
}

#: Consonants. Deliberately the popular spellings Indians use for their own
#: names rather than IAST: "ch" not "c", and no diacritics anywhere, because the
#: output is read by people and matched against an ASCII catalogue.
_CONSONANTS: dict[int, str] = {
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "ng",
    0x1A: "ch", 0x1B: "chh", 0x1C: "j", 0x1D: "jh", 0x1E: "ny",
    0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n",
    0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n",
    0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m",
    0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l",
    0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
    # Nukta forms: borrowed sounds that Urdu and English loanwords rely on.
    0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "r", 0x5D: "rh",
    0x5E: "f", 0x5F: "y",
}

#: Dependent vowel signs. These REPLACE the inherent vowel.
_MATRAS: dict[int, str] = {
    0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u",
    0x43: "ri", 0x44: "ri", 0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai",
    0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au", 0x62: "ri", 0x63: "li",
}

_VIRAMA = 0x4D
#: Candrabindu and anusvara: a nasal whose place of articulation is borrowed
#: from the consonant after it. Defaulting to "n" gives "munbai" and "planbar";
#: taking the labial when a labial follows gives Mumbai and plambar, which is
#: what a caller means and what a matcher can use.
_NASALS = {0x01, 0x02}
_LABIALS = {0x2A, 0x2B, 0x2C, 0x2D, 0x2E}
_VISARGA = 0x03
#: Nukta modifies the previous letter and has no sound of its own; avagraha is
#: a lengthening mark. Both are silent here.
_SILENT = {0x3C, 0x3D, 0x4E, 0x4F}
_DIGITS = {0x66 + n: str(n) for n in range(10)}


def _block(ch: str) -> tuple[int, str] | None:
    """(offset, script) for an Indic character, or None if it is not one."""
    code = ord(ch)
    for start, script in _BLOCKS:
        if start <= code <= start + 0x7F:
            return code - start, script
    return None


#: Zero-width joiner and non-joiner. Indic typography, not sound: they tell a
#: renderer whether to form a conjunct and belong to no script block, so the
#: "non-Indic characters pass through untouched" rule carried them into the
#: output. "కోల్‌కతా" romanised to "kol‌kata" — indistinguishable from
#: "kolkata" to a reader and unequal to it everywhere it matters.
_ZERO_WIDTH = {ord("‌"): None, ord("‍"): None}


def _visible(text: str) -> str:
    return (text or "").translate(_ZERO_WIDTH)


def is_indic(text: str) -> bool:
    """Whether any character belongs to a script this can romanise."""
    return any(_block(ch) is not None for ch in text or "")


def romanise_word(word: str) -> str:
    """One whitespace-free token in Latin letters.

    Non-Indic characters pass through untouched, so a mixed token like
    "AC सर्विस" survives and "plumber" is returned as it arrived. Zero-width
    joiners are the exception: they are typography rather than sound, and
    letting them through leaves an invisible character in the output.
    """
    word = _visible(word)
    out: list[str] = []
    #: True when a consonant has been emitted whose inherent vowel has not yet
    #: been decided — the next character either replaces it, cancels it, or
    #: lets it stand.
    pending = False
    script = ""

    def flush() -> None:
        nonlocal pending
        if pending:
            out.append(_INHERENT)
            pending = False

    for position, ch in enumerate(word):
        found = _block(ch)
        if found is None:
            flush()
            out.append(ch)
            continue
        offset, script = found
        if offset in _CONSONANTS:
            flush()
            out.append(_CONSONANTS[offset])
            pending = True
        elif offset in _MATRAS:
            # Replaces the inherent vowel rather than following it.
            pending = False
            out.append(_MATRAS[offset])
        elif offset == _VIRAMA:
            pending = False
        elif offset in _VOWELS:
            flush()
            out.append(_VOWELS[offset])
        elif offset in _NASALS:
            flush()
            following = _block(word[position + 1]) if position + 1 < len(word) else None
            out.append("m" if following and following[0] in _LABIALS else "n")
        elif offset == _VISARGA:
            flush()
            out.append("h")
        elif offset in _DIGITS:
            flush()
            out.append(_DIGITS[offset])
        elif offset in _SILENT:
            continue
        else:
            # Unassigned in this block, or punctuation like danda. Dropping it
            # is right for both: it carries no sound.
            flush()

    if pending and script not in _DROPS_FINAL_SCHWA:
        out.append(_INHERENT)
    return "".join(out)


def romanise(text: str) -> str:
    """`text` in Latin letters, word by word. Never raises, never returns None."""
    if not text:
        return ""
    # Before the Indic test, not after: a value that is already Latin apart from
    # a stray joiner is exactly the one that returns early.
    text = _visible(text)
    if not is_indic(text):
        return text
    return " ".join(romanise_word(word) for word in text.split())

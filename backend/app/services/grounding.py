"""Did the caller actually say this?

On the speech-to-speech path the model is the transcriber, and nothing between
it and a saved lead used to check its work: `set_name` took whatever string it
passed. A realtime model under a noisy phone line does not fail loudly — it
produces a fluent, plausible Indian name that the caller never said, and that
name reaches the lead, the WhatsApp message and Postgres looking exactly like a
real one.

So every value a tool is handed is checked against the caller's own audio,
using the transcript LiveKit emits for logging (`user_input_transcribed`). That
transcript comes from a *different* model than the one hearing the call, which
is precisely what makes it useful evidence: two independent decodes of the same
audio agree on a name the caller said and disagree on one that was invented.

Comparison is script-blind. The realtime model routinely hands over "Rahul"
while the transcript reads "राहुल", so both sides are romanised
(`translit.py`) before anything is compared — otherwise the caller's own name
is rejected as a fabrication on every Hindi call.

Comparison is also fuzzy, because two decoders of the same phone audio spell
Indian names slightly differently ("Rahul"/"Rahool"). The cutoff sits where
those variants pass and unrelated names do not: "rahul"/"rahool" is 0.80 while
"rahul"/"suresh" — a fluent hallucination — is 0.18.

Strictness differs by field, and the difference is deliberate:

  * **Names and cities are proper nouns.** They exist only because the caller
    uttered them, so every word must be traceable to the audio.
  * **A service is a description.** A caller says "मुझे एसी ठीक करवाना है" and
    the model reasonably hands over "AC repair" — "repair" was never spoken.
    Requiring every word there would reject good calls, so one grounded content
    word is enough, and a value the rule extractor independently derives from
    the transcript is accepted outright.
"""

from __future__ import annotations

import difflib
from enum import Enum

from ..languages import normalize_text
from . import translit

#: How similar two words must sound to count as the same one. Applied to the
#: sound key below, not to raw spelling: on raw spelling the two populations
#: overlap — "lakshmi"/"laxmi" is 0.67 and "phani"/"fani" is 0.67, while
#: "asha"/"usha" is 0.75 — and no single cutoff separates them. Keyed, the same
#: pairs are 1.00 and the unrelated ones stay where they were ("suresh"/"rahul"
#: 0.18, "delhi"/"mumbai" 0.18).
#:
#: What is left above the line is one-letter pairs of short names —
#: "neha"/"meha", "pune"/"puna". Those are accepted deliberately: two decoders
#: differing by a letter is far likelier than a hallucination landing that close,
#: and the read-back is what settles which of the two the caller said.
CUTOFF = 0.75

#: Words too short to carry evidence. "ac", "ki", "hai" match something in
#: almost any sentence, so grounding a service on one of them grounds nothing.
MIN_CONTENT_CHARS = 4


class Verdict(str, Enum):
    HEARD = "heard"
    NOT_HEARD = "not_heard"
    #: No caller audio was transcribed at all — transcription is off, or failed.
    #: Not evidence of anything, so a tool must not reject on it.
    NO_TRANSCRIPT = "no_transcript"


def fold(text: str) -> str:
    """A string reduced to comparable form: romanised, casefolded, depunctuated."""
    return normalize_text(translit.romanise(text or ""))


def _tokens(text: str) -> list[str]:
    return [t for t in fold(text).split() if t]


#: Placeholder for a digraph being protected from a single-letter rule below.
_CH = "\x00"

#: The ways two systems spell the same Indian name differently. Every pair here
#: came out of comparing real romanisations against each other: Sarvam's, the
#: offline table's, and what a transcription model writes when it hears the same
#: audio. Applied to both sides, so they meet in the middle rather than either
#: being declared correct.
_SOUND_RULES: tuple[tuple[str, str], ...] = (
    # "c" is "k" except in "ch", so "ch" is parked on a character that cannot
    # occur here and put back afterwards — otherwise "car"/"kar" needs a fuzzy
    # match and "chai" becomes "khai". (`fold` has already removed anything that
    # is not a word character, so the marker cannot collide with real input.)
    ("ch", _CH), ("c", "k"), (_CH, "ch"),
    # Consonant clusters written two ways: Lakshmi/Laxmi, Phani/Fani.
    ("x", "ks"), ("ph", "f"), ("chh", "ch"), ("sh", "s"), ("th", "t"),
    ("gh", "g"), ("bh", "b"), ("dh", "d"), ("kh", "k"),
    # Long vowels written doubled or singly: Ruchee/Ruchi, Rahool/Rahul.
    ("aa", "a"), ("ee", "i"), ("oo", "u"), ("ii", "i"), ("uu", "u"),
    # y/i and v/b/w vary by region and by transcriber — Ravi is Rabi in Bengali
    # romanisation, Sanjay is Sanjai.
    ("y", "i"), ("v", "b"), ("w", "b"), ("z", "j"),
)


def sound_key(word: str) -> str:
    """A word reduced to roughly how it sounds.

    Not a phonetic algorithm; a spelling-variance eraser for one specific
    problem — two romanisations of one Indian name. Soundex and Metaphone are
    both tuned for English and both collapse far more aggressively than this,
    which matters when the thing being decided is whether a caller said a name.
    """
    key = word
    for old, new in _SOUND_RULES:
        key = key.replace(old, new)
    # Doubled anything is a single sound: Krishnaa, Siddharth.
    collapsed: list[str] = []
    for ch in key:
        if not collapsed or collapsed[-1] != ch:
            collapsed.append(ch)
    key = "".join(collapsed)
    # A trailing inherent vowel or breath is exactly what the scripts disagree
    # about: Ram/Rama, Ruchi/Ruchih. One character, not a run — stripping a run
    # takes "shah" down to "s", which then matches far too much.
    if len(key) > 1 and key[-1] in "ah":
        key = key[:-1]
    return key


#: Cities with two current names. Not spelling variants — "bengaluru" and
#: "bangalore" sound nothing alike and score 0.67 — but the caller says one and
#: the model hands over the other, and refusing that is refusing a correct
#: answer about the largest cities we serve.
_SAME_PLACE: tuple[frozenset[str], ...] = tuple(
    frozenset(names) for names in (
        ("bengaluru", "bangalore"),
        ("mumbai", "bombay"),
        ("kolkata", "calcutta"),
        ("chennai", "madras"),
        ("mysuru", "mysore"),
        ("visakhapatnam", "vizag"),
        ("gurugram", "gurgaon"),
        ("puducherry", "pondicherry"),
        ("thiruvananthapuram", "trivandrum"),
        ("kochi", "cochin"),
        ("varanasi", "banaras", "kashi"),
        ("prayagraj", "allahabad"),
    )
)


def _same_place(word: str, heard: list[str]) -> bool:
    for names in _SAME_PLACE:
        if word in names and any(h in names for h in heard):
            return True
    return False


#: Below this a word is too short for its consonant pattern to identify it:
#: "ram" and "rum" would be one word. At four and above the pattern is
#: distinctive — "plumber" and "painter" stay apart.
_VOWEL_BLIND_MIN = 4


def _vowel_blind(key: str) -> str:
    """A sound key with every vowel reduced to "a vowel was here".

    Loanwords are where vowels stop being evidence. An English word written back
    out of an Indic script comes back vowel-shifted, because the script writes an
    inherent vowel the English spelling does not have: "प्लंबर" romanises to
    "plambar" and "वॉश" to "vosh". The consonants and the shape survive intact.

    Positions are kept rather than the vowels dropped entirely, so this cannot
    quietly merge words of different lengths.
    """
    return "".join("V" if ch in "aeiou" else ch for ch in key)


def _word_is_heard(word: str, heard: list[str]) -> bool:
    if not word:
        return False
    if any(word == h for h in heard):
        return True
    # Contained in a longer transcribed word, or containing one: agglutinative
    # transcripts glue particles on ("ravigaru", "hyderabadlo").
    if any(len(h) >= 4 and (word in h or h in word) for h in heard):
        return True
    if _same_place(word, heard):
        return True
    key = sound_key(word)
    keys = [sound_key(h) for h in heard]
    if key in keys:
        return True
    if len(word) >= _VOWEL_BLIND_MIN:
        shape = _vowel_blind(key)
        if any(
            len(h) >= _VOWEL_BLIND_MIN and _vowel_blind(sound_key(h)) == shape
            for h in heard
        ):
            return True
    return bool(difflib.get_close_matches(key, keys, n=1, cutoff=CUTOFF))


def check(value: str, turns: list[str], *, every_word: bool = True) -> Verdict:
    """Whether `value` is traceable to anything in `turns`.

    `every_word` is the proper-noun rule: with it, each word of the value must
    be found; without it, one content word is enough. See the module docstring
    for why a service is judged more loosely than a name.
    """
    spoken = [t for t in turns if (t or "").strip()]
    if not spoken:
        return Verdict.NO_TRANSCRIPT

    heard = _tokens(" ".join(spoken))
    if not heard:
        return Verdict.NO_TRANSCRIPT

    words = _tokens(value)
    if not words:
        return Verdict.NOT_HEARD

    # Whole phrase said verbatim — the common case, and cheap to settle first.
    folded = fold(value)
    if folded and folded in " ".join(heard):
        return Verdict.HEARD

    if every_word:
        ok = all(_word_is_heard(word, heard) for word in words)
    else:
        content = [w for w in words if len(w) >= MIN_CONTENT_CHARS] or words
        ok = any(_word_is_heard(word, heard) for word in content)
    return Verdict.HEARD if ok else Verdict.NOT_HEARD

"""How much of this did the caller actually say?

The model is the transcriber on a speech-to-speech call, and nothing between it
and a saved lead used to check its work. A realtime model under a noisy phone
line does not fail loudly — it produces a fluent, plausible Indian name the
caller never said, and that name reaches the lead and the WhatsApp message
looking exactly like a real one.

So every captured value is scored against the caller's own audio, using the
transcript the agent ships with `call.ended`. That transcript comes from a
*different* model than the one hearing the call, which is precisely what makes
it useful evidence: two independent decodes of the same audio agree on a name
the caller said and disagree on one that was invented.

This is a score, not a gate. It ran as a refusal inside the agent and sent
callers back to repeat their name at the moment they expected to hang up. The
read-back corrects a bad value while they are still on the line; this catches
the one that got past it.

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

from ..text import normalize_text
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


def score(value: str, turns: list[str], *, every_word: bool = True) -> float:
    """How much of `value` is traceable to the caller's own audio, 0..1.

    A score rather than the verdict above, because this no longer gates
    anything. Inside the agent this check was a refusal: a value it could not
    corroborate was rejected and the caller was asked again — at save time,
    which is the moment they expect to hang up. Here it is evidence for a
    review queue, so a *partial* match is worth distinguishing from no match at
    all. "Ravi Kumar" where only "Ravi" was transcribed is a much better lead
    than one where neither word was.

    The read-back is what corrects a wrong value while the caller is still on
    the line. This is what catches the one that got past it.
    """
    spoken = [t for t in turns if (t or "").strip()]
    if not spoken:
        return 0.0

    heard = _tokens(" ".join(spoken))
    words = _tokens(value)
    if not heard or not words:
        return 0.0

    folded = fold(value)
    if folded and folded in " ".join(heard):
        return 1.0

    if not every_word:
        # A description only has to be anchored somewhere. Scoring it by the
        # fraction of words found would punish exactly the rephrasing that is
        # correct — "AC repair" for "मुझे एसी ठीक करवाना है" says a word the
        # caller never did, and is still the right answer.
        content = [w for w in words if len(w) >= MIN_CONTENT_CHARS] or words
        return 1.0 if any(_word_is_heard(w, heard) for w in content) else 0.0

    found = sum(1 for word in words if _word_is_heard(word, heard))
    return found / len(words)

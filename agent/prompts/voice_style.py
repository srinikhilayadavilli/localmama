"""Accent and delivery steering for the OpenAI voice models.

OpenAI ships no Indian-accented voice. `gpt-4o-mini-tts` instead takes a free-
text `instructions` field that steers delivery — accent, rhythm, warmth, pace —
independently of the words being spoken. That field is the only lever that
makes `coral` or `sage` sound like someone from Hyderabad rather than someone
from California, so it is treated here as real configuration, not decoration.

The same idea applies on the way in: the transcription models take a `prompt`
that biases decoding. Feeding it Indian names, city names, and the service
vocabulary callers actually use is what stops "Lakshmi" becoming "Lucky" and
"Kukatpally" becoming "cook at Bally".

Both strings are per-language, and both are rebuilt when the caller switches
language mid-call (see `agent.py`).
"""

from __future__ import annotations

from ..languages import ENDONYM, Language

#: Accent and delivery notes shared by every language. Written as prose because
#: that is what the model responds to — a terse list of tags steers much less.
_BASE_STYLE = """\
Voice: a warm, friendly Indian woman in her late thirties — the helpful \
neighbourhood aunty everyone in the building calls when something breaks. \
Familiar and reassuring, never a scripted call-centre agent.

Accent: native Indian. This is the single most important instruction. Use the \
syllable-timed rhythm of Indian speech, where each syllable gets close to equal \
weight, rather than the stress-timed rhythm of American or British English. Use \
retroflex t and d, a clear trilled r, unaspirated p/t/k, a dental th, and full \
unreduced vowels — do not swallow unstressed syllables into a schwa. Never drift \
into an American, British, or generic "neutral" accent at any point in the turn.

Delivery: unhurried and clear, slightly slower than a native English newsreader, \
with a gentle rising intonation on questions. Warm and patient throughout — the \
caller may be on a bad line, in traffic, or hesitant about what they need.

Details: say money the Indian way ("five hundred rupees", "one and a half lakh"). \
Read phone numbers digit by digit with a small pause after every group. Keep \
Indian names and place names in their natural Indian pronunciation — Lakshmi, \
Gachibowli, Kukatpally, Koramangala — and never anglicise them."""

#: Per-language additions: the regional flavour, plus the code-switching rule
#: that matters most on a real call in India.
_LANGUAGE_STYLE: dict[Language, str] = {
    Language.ENGLISH: """\
Language: Indian English, as spoken in a metro like Hyderabad or Bangalore. \
Educated and fluent, but unmistakably Indian — not an Indian speaker imitating \
an American one. Sprinkle in the natural discourse habits of Indian English \
where they fit ("only", "itself", "please tell me") without laying it on thick.
This is absolute and has no exceptions. Do not drift into an American, British \
or Australian accent at any point, however long the call runs. If the caller \
speaks American-accented English, do NOT mirror them. Re-anchor to Indian \
English after every interruption, every correction, and every language switch \
— those are exactly the moments the accent slips.""",
    Language.HINDI: """\
Language: Hindi, as a native speaker from north India. Everyday spoken \
Hindustani, not literary Sanskritised Hindi. Pronounce the English words that \
Indians normally keep in Hindi — plumber, service, area, booking, OK — with an \
Indian accent, exactly as a Hindi speaker would, never switching to an American \
accent for them.""",
    Language.BENGALI: """\
Language: Bengali, as a native speaker from Kolkata. Warm, conversational \
Bengali with its characteristic soft intonation. Keep everyday English loanwords \
in Indian pronunciation rather than switching accent for them.""",
    Language.TELUGU: """\
Language: Telugu, as a native speaker from Hyderabad. Everyday spoken Telugu, \
the way people actually talk at home, not formal news-reader Telugu. Keep \
everyday English loanwords — plumber, current, service, area — in Indian \
pronunciation rather than switching accent for them.""",
    Language.TAMIL: """\
Language: Tamil, as a native speaker from Chennai. Everyday spoken Tamil rather \
than formal literary Tamil. Keep everyday English loanwords in Indian \
pronunciation rather than switching accent for them.""",
    Language.KANNADA: """\
Language: Kannada, as a native speaker from Bangalore. Everyday spoken Kannada \
rather than formal literary Kannada. Keep everyday English loanwords in Indian \
pronunciation rather than switching accent for them.""",
}

#: Context for the transcription model — and deliberately NOT a word list.
#:
#: This used to enumerate Indian names, neighbourhoods and services to bias
#: decoding toward them. That is a documented way to break Whisper-family
#: models, `gpt-4o-transcribe` included: on silence or line noise they emit the
#: prompt's own content as if it had been spoken. Callers saw phantom turns
#: reading "Suresh" and "Lakshmi" — names straight out of the list — and a
#: phantom name is not a cosmetic glitch, because the workflow captures it as
#: the caller's name and a hallucinated "plumber" as their service.
#:
#: So the prompt now describes the *situation* only. Every word here is one the
#: model might hallucinate, so it stays short, and nothing in it is shaped like
#: a value the tools capture. Indian names and places are recovered by the
#: backend's rules instead, which run on real transcripts.
_STT_VOCABULARY = "They are asking for help with a household service."


def realtime_accent_instructions() -> str:
    """Accent block for the speech-to-speech model (`worker.py`).

    A realtime model owns all six languages inside one session and switches
    whenever the caller does, so every language's note goes in up front — there
    is no per-turn `instructions` field to swap, only the system prompt.
    """
    notes = "\n".join(
        f"- {language.value.capitalize()} ({ENDONYM[language]}): "
        f"{_LANGUAGE_STYLE[language].removeprefix('Language: ')}"
        for language in Language
    )
    return (
        f"{_BASE_STYLE}\n\n"
        "Languages: the caller picks one at the start of the call. Speak that "
        "one, as a native speaker, for the rest of the call — and keep the "
        "Indian accent above in every one of them.\n"
        f"{notes}"
    )


#: Before the caller has picked a language: no language is asserted, so the
#: model is free to auto-detect, but the vocabulary bias still applies.
GREETING_STT_PROMPT = (
    "An Indian caller with an Indian accent, on a phone line, speaking English, "
    "Hindi, Bengali, Telugu, Tamil, or Kannada, and possibly mixing English "
    f"words into the sentence. {_STT_VOCABULARY}"
)

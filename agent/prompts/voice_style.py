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
#: Deliberately empty.
#:
#: It used to read "They are asking for help with a household service." — one
#: short sentence, already trimmed once for this exact reason. It was still
#: enough. Whisper-family models emit prompt content as transcription when fed
#: noise or silence, and on a real call the transcript filled with invented
#: service requests: "I want a new geyser installed" in Hindi, on a Tamil call,
#: from a caller who said nothing at all.
#:
#: Those phantom turns are not harmless. They are the evidence the backend
#: audits captured values against, so noise became corroboration.
_STT_VOCABULARY = ""


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


#: What the greeting is generated from.
#:
#: `generate_reply` takes its own instruction, and that instruction is what the
#: model is most immediately steered by — the system prompt's accent block is
#: two thousand tokens away and the conversation has no history yet to set a
#: voice from. Asking only for "greet the caller" got a greeting in the model's
#: default American, on the one turn that sets the caller's expectation for the
#: whole call.
#:
#: So the accent travels with the request. Short, because it competes with
#: nothing else here, and blunt for the same reason.
GREETING_INSTRUCTION = (
    "Greet the caller and begin. Speak in a NATIVE INDIAN ENGLISH accent from "
    "the very first syllable — the warm, clear English of a Hyderabad or "
    "Bangalore call centre. Syllable-timed rhythm with equal weight on each "
    "syllable, retroflex t and d, unaspirated p/t/k, full unreduced vowels. "
    "NOT American, NOT British, NOT neutral. This first sentence sets the "
    "accent for the entire call, so it is the one that matters most."
)

#: What a tool result says the moment the caller picks their language.
#:
#: Same reasoning as `GREETING_INSTRUCTION`, at the second turn that sets the
#: accent for a whole call. The system prompt's accent block is thousands of
#: tokens back; the tool result is the most recent thing in context and the
#: model is steered by it far more strongly.
#:
#: **English is the case that needs this most, not least.** Picking Telugu or
#: Hindi forces the model off its default voice whether it wants to or not —
#: there is no American Telugu to fall back into. Picking *English* changes
#: nothing about the words it was already producing, so it simply carries on in
#: the accent it defaults to, which is American. The one language where the
#: caller's choice does not itself disturb the voice is the one where the voice
#: has to be named explicitly.
_ACCENT_REANCHOR: dict[Language, str] = {
    Language.ENGLISH: (
        " Speak INDIAN ENGLISH from the very next syllable — the warm, clear "
        "English of Hyderabad or Bangalore. Syllable-timed rhythm, retroflex t "
        "and d, unaspirated p/t/k, full unreduced vowels. NOT American, NOT "
        "British, NOT neutral. Choosing English does NOT mean American English; "
        "it changes the language, never the accent. Hold this for the whole "
        "call, including after every interruption and correction."
    ),
}

#: Every other language gets the shorter form: the switch itself does most of
#: the work, and what is left to guard is the English words inside the sentence.
_ACCENT_REANCHOR_DEFAULT = (
    " Speak {language} as a native speaker, in a native Indian accent. Keep the "
    "English words you mix in — plumber, service, area, booking — in Indian "
    "pronunciation too; never switch to an American accent for them."
)


def accent_reminder(language: Language) -> str:
    """The accent instruction that rides along with a recorded language."""
    return _ACCENT_REANCHOR.get(
        language, _ACCENT_REANCHOR_DEFAULT.format(language=language.value)
    )


#: Before the caller has picked a language: no language is asserted, so the
#: model is free to auto-detect, but the vocabulary bias still applies.
GREETING_STT_PROMPT = (
    "An Indian caller with an Indian accent, on a phone line, speaking English, "
    "Hindi, Bengali, Telugu, Tamil, or Kannada, and possibly mixing English "
    "words into the sentence."
)

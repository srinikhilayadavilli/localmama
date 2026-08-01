"""Resolving which language the caller asked for.

The only interpretation the agent still does, because it has to: it needs to
know what to speak next before the turn ends.
"""

from __future__ import annotations

import pytest

from agent.languages import Language, resolve_language
from agent.prompts.voice_style import realtime_accent_instructions


@pytest.mark.parametrize("said, expected", [
    ("Hindi", Language.HINDI),
    ("I would like Telugu please", Language.TELUGU),
    ("let's do bangla", Language.BENGALI),
    ("kannad", Language.KANNADA),
    ("tamizh", Language.TAMIL),
])
def test_a_language_named_in_any_form_resolves(said, expected):
    assert resolve_language(said) is expected


@pytest.mark.parametrize("said, expected", [
    ("తెలుగు", Language.TELUGU),
    ("हिंदी", Language.HINDI),
    ("বাংলা", Language.BENGALI),
])
def test_a_language_named_in_its_own_script_resolves(said, expected):
    assert resolve_language(said) is expected


def test_a_language_named_in_the_wrong_script_still_resolves():
    """Auto-detecting recognisers routinely write Indic audio in Devanagari,
    because it is the best-represented Indian script in these models. A caller
    who said "తెలుగు" and got transcribed "तेलुगु" used to match no alias, so
    the script check asserted Hindi — and the whole call continued in Hindi."""
    assert resolve_language("तेलुगु") is Language.TELUGU


def test_speaking_a_language_resolves_it_without_naming_it():
    """A caller who just starts talking has still told us."""
    assert resolve_language("నాకు ప్లంబర్ కావాలి") is Language.TELUGU


def test_an_unsupported_language_resolves_to_nothing():
    """None means re-prompt, rather than guessing at a language we cannot speak."""
    assert resolve_language("French") is None
    assert resolve_language("") is None


def test_latin_text_is_not_asserted_to_be_english():
    """Romanised Indian languages are Latin too, so script alone proves nothing."""
    assert resolve_language("naaku plumber kaavali") is None


def test_the_accent_block_covers_every_language():
    """One system prompt owns all six: a speech-to-speech model has no TTS
    object to retune when the caller switches."""
    block = realtime_accent_instructions()
    for language in Language:
        assert language.value.capitalize() in block


# --- the accent, which is prompt-only on a speech-to-speech model ----------


def test_the_greeting_carries_the_accent_itself():
    """`generate_reply` takes its own instruction, and that is what the model
    is most immediately steered by — the system prompt's accent block is two
    thousand tokens away and there is no conversation yet to set a voice from.

    Asking only for "greet the caller" produced a greeting in the model's
    default American, on the one turn that sets the caller's expectation for
    the whole call."""
    from agent.prompts.voice_style import GREETING_INSTRUCTION

    lowered = GREETING_INSTRUCTION.lower()
    assert "indian english" in lowered
    for excluded in ("american", "british", "neutral"):
        assert excluded in lowered, f"{excluded} must be named to be excluded"


def test_the_accent_is_anchored_before_the_workflow_not_only_after_it():
    """It used to appear first at 67% through the prompt. The model's opening
    tokens saw nothing about how to sound."""
    from agent.prompts.flow import instructions

    text = instructions()
    first = text.lower().find("indian english")
    assert 0 <= first < 400, f"accent first mentioned at char {first}"


def test_every_language_still_carries_the_indian_accent():
    """Switching to Telugu must not switch to a Telugu-accented American."""
    from agent.prompts.voice_style import realtime_accent_instructions

    block = realtime_accent_instructions().lower()
    assert "native indian" in block
    assert "never drift" in block


def test_the_stt_prompt_seeds_nothing_a_caller_could_be_credited_with():
    """Whisper-family models emit prompt content as transcription when fed
    noise. On a real call the transcript filled with invented service requests
    — "I want a new geyser installed", in Hindi, on a Tamil call, from someone
    who had said nothing — because the prompt told it callers ask about
    household services.

    Those phantom turns are the evidence the backend audits captured values
    against, so noise was being counted as corroboration."""
    from agent.prompts.voice_style import GREETING_STT_PROMPT

    lowered = GREETING_STT_PROMPT.lower()
    for seed in ("service", "plumber", "electrician", "geyser", "household",
                 "repair", "cleaning", "asking for help"):
        assert seed not in lowered, f"{seed!r} is a word the model can hallucinate"


# --- the accent travels with the language choice ---------------------------


def test_choosing_english_names_the_accent_explicitly():
    """Picking Telugu or Hindi forces the model off its default voice whether
    it wants to or not — there is no American Telugu to fall back into. Picking
    English changes nothing about the words it was already producing, so it
    carries on in the accent it defaults to. That is the one language whose
    accent has to be said out loud."""
    from agent.prompts.voice_style import accent_reminder
    from agent.languages import Language

    said = accent_reminder(Language.ENGLISH).lower()
    assert "indian english" in said
    assert "not american" in said
    for drift in ("american", "british", "neutral"):
        assert drift in said


def test_every_language_gets_an_accent_reminder():
    from agent.prompts.voice_style import accent_reminder
    from agent.languages import Language

    for language in Language:
        said = accent_reminder(language).lower()
        assert "indian" in said, language
        assert "american" in said, language

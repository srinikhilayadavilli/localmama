"""OpenAI voice wiring: accent instructions, ISO codes, and plugin arguments.

The accent is the product here — a voice that slips into an American default is
a bug, not a cosmetic issue — so the pieces that carry it are pinned by tests
even though they are "just strings".
"""

from __future__ import annotations

import pytest

from backend.app.config import settings
from backend.app.languages import ENDONYM, LOCALE, Language, iso639_1
from backend.app.prompts.voice_style import (
    GREETING_STT_PROMPT,
    realtime_accent_instructions,
    stt_prompt,
    tts_instructions,
)


@pytest.fixture
def configure():
    """Temporarily override `settings`, which is a frozen dataclass.

    `monkeypatch.setattr` cannot touch it (FrozenInstanceError) and patching the
    class does not help either, because dataclass fields live on the instance.
    """
    original: dict = {}

    def _set(**values) -> None:
        for key, value in values.items():
            original.setdefault(key, getattr(settings, key))
            object.__setattr__(settings, key, value)

    yield _set

    for key, value in original.items():
        object.__setattr__(settings, key, value)


@pytest.mark.parametrize(
    "language,expected",
    [
        (Language.ENGLISH, "en"),
        (Language.HINDI, "hi"),
        # These two are the reason the helper exists: the enum name would give
        # "be" and "ka", which are Belarusian and Georgian.
        (Language.BENGALI, "bn"),
        (Language.KANNADA, "kn"),
        (Language.TELUGU, "te"),
        (Language.TAMIL, "ta"),
    ],
)
def test_iso639_1(language: Language, expected: str) -> None:
    assert iso639_1(language) == expected
    assert LOCALE[language].startswith(f"{expected}-")


@pytest.mark.parametrize("language", list(Language))
def test_every_language_has_accent_instructions(language: Language) -> None:
    text = tts_instructions(language)
    assert "native Indian" in text
    assert "Never drift" in text
    # The per-language block must actually be present, not just the base style.
    assert "Language:" in text


@pytest.mark.parametrize("language", list(Language))
def test_instructions_name_the_language(language: Language) -> None:
    assert language.value.capitalize() in stt_prompt(language)


def test_greeting_prompt_asserts_no_single_language() -> None:
    """The opening turn may arrive in any of the six, so none may be pinned."""
    assert "Telugu" in GREETING_STT_PROMPT and "Hindi" in GREETING_STT_PROMPT


#: Words that must never appear in an STT prompt. Whisper-family models emit
#: prompt content as transcription on silence, so anything here would arrive as
#: a phantom turn — and these are shaped exactly like values the state machine
#: captures, so a phantom would be committed as a real name or service.
_FORBIDDEN_IN_STT_PROMPTS = (
    "Suresh", "Lakshmi", "Ramesh", "Priya", "Kavitha",     # names
    "plumber", "electrician", "carpenter", "maid",          # services
    "Gachibowli", "Koramangala", "Madhapur", "Hyderabad",   # places
)


@pytest.mark.parametrize("prompt_text", [
    GREETING_STT_PROMPT, *[stt_prompt(lang) for lang in Language],
])
def test_stt_prompts_contain_no_capturable_vocabulary(prompt_text: str) -> None:
    """Regression: a vocabulary list here was read aloud by the model on silence.

    Callers saw phantom turns containing "Suresh" and "Lakshmi" — both straight
    out of the list this prompt used to carry — and the workflow captured them
    as the caller's name.
    """
    for word in _FORBIDDEN_IN_STT_PROMPTS:
        assert word.lower() not in prompt_text.lower(), (
            f"{word!r} in an STT prompt: the model will eventually speak it "
            f"on silence and the workflow will capture it as a real answer"
        )


def test_build_stt_passes_openai_arguments(monkeypatch, configure) -> None:
    openai = pytest.importorskip("livekit.plugins.openai")
    from backend.app.providers import livekit_plugins

    configure(stt_provider="openai", openai_stt_detect_language=True)
    captured: dict = {}
    monkeypatch.setattr(
        openai, "STT", lambda **kwargs: captured.update(kwargs) or "stt"
    )

    assert livekit_plugins.build_stt(Language.ENGLISH) == "stt"
    assert captured["detect_language"] is True
    assert "language" not in captured  # mutually exclusive with detection
    assert captured["prompt"] == GREETING_STT_PROMPT


def test_build_stt_pins_language_when_detection_is_off(monkeypatch, configure) -> None:
    openai = pytest.importorskip("livekit.plugins.openai")
    from backend.app.providers import livekit_plugins

    configure(stt_provider="openai", openai_stt_detect_language=False)
    captured: dict = {}
    monkeypatch.setattr(openai, "STT", lambda **kwargs: captured.update(kwargs))

    livekit_plugins.build_stt(Language.BENGALI)
    assert captured["language"] == "bn"
    assert "detect_language" not in captured


def test_build_tts_carries_the_accent(monkeypatch, configure) -> None:
    openai = pytest.importorskip("livekit.plugins.openai")
    from backend.app.providers import livekit_plugins

    configure(tts_provider="openai", openai_tts_instructions="")
    captured: dict = {}
    monkeypatch.setattr(openai, "TTS", lambda **kwargs: captured.update(kwargs))

    livekit_plugins.build_tts(Language.TELUGU)
    assert captured["instructions"] == tts_instructions(Language.TELUGU)
    assert captured["model"] == "gpt-4o-mini-tts", "only this model takes instructions"


def test_tts_language_switch_uses_instructions(configure) -> None:
    """A mid-call switch must retune the accent, not just the words."""
    from backend.app import agent as agent_module

    configure(tts_provider="openai", openai_tts_instructions="")
    calls: list[dict] = []

    class FakeTTS:
        def update_options(self, **kwargs):
            calls.append(kwargs)

    agent_module._apply_language_to_tts(FakeTTS(), Language.TAMIL)
    assert calls == [{"instructions": tts_instructions(Language.TAMIL)}]


def test_stt_language_switch_pins_language() -> None:
    from backend.app import agent as agent_module

    calls: list[dict] = []

    class FakeSTT:
        def update_options(self, **kwargs):
            calls.append(kwargs)

    agent_module._apply_language_to_stt(FakeSTT(), Language.KANNADA)
    assert calls == [
        {"language": "kn", "prompt": stt_prompt(Language.KANNADA)},
    ]


def test_realtime_instructions_cover_every_language() -> None:
    """A speech-to-speech model switches language itself, so all six must be
    described up front — there is no per-turn instructions field to swap."""
    text = realtime_accent_instructions()
    assert "native Indian" in text
    for language in Language:
        assert language.value.capitalize() in text
        assert ENDONYM[language] in text
    # Derived from the TTS blocks, so the label must not be left dangling.
    assert "Language: " not in text


def test_realtime_agent_instructions_carry_workflow_and_accent() -> None:
    from backend.app import agent_realtime

    text = agent_realtime.instructions()
    assert "call set_language" in text, "workflow rules must survive"
    assert "native Indian" in text, "accent must ride along in the same prompt"


@pytest.mark.parametrize(
    "provider,expected",
    [("openai", "openai"), ("gemini", "gemini"), ("nonsense", "openai")],
)
def test_realtime_provider_selection(monkeypatch, configure, provider, expected) -> None:
    """An unknown provider falls back to OpenAI rather than failing the call."""
    from backend.app import agent_realtime

    configure(realtime_provider=provider)
    monkeypatch.setattr(agent_realtime, "_build_openai_model", lambda: "openai")
    monkeypatch.setattr(agent_realtime, "_build_gemini_model", lambda: "gemini")
    assert agent_realtime.build_realtime_model() == expected


def test_realtime_openai_needs_a_key(configure) -> None:
    from backend.app import agent_realtime

    configure(realtime_provider="openai", openai_api_key="")
    assert agent_realtime.build_realtime_model() is None


def test_stt_switch_survives_a_plugin_without_that_option() -> None:
    """Sarvam's STT has no update_options; that must not end the call."""
    from backend.app import agent as agent_module

    class FixedSTT:
        pass

    agent_module._apply_language_to_stt(FixedSTT(), Language.HINDI)  # no raise
    agent_module._apply_language_to_stt(None, Language.HINDI)


# --- Indian English, always ------------------------------------------
#
# The accent is the product for an Indian audience, and the reported failure
# was drift: it slips after an interruption, a correction, or a language
# switch. These pin the instruction rather than the audio, which is the only
# part we control deterministically.


def test_english_forbids_every_other_accent() -> None:
    text = tts_instructions(Language.ENGLISH)
    lowered = text.lower()
    for accent in ("american", "british", "australian"):
        assert accent in lowered, f"{accent} must be named so it can be excluded"
    assert "no exceptions" in lowered
    assert "do not mirror them" in lowered, "mirroring the caller is how it drifts"


def test_english_re_anchors_at_the_moments_it_slips() -> None:
    """Interruptions, corrections and language switches are the reported triggers."""
    text = tts_instructions(Language.ENGLISH).lower()
    for trigger in ("interruption", "correction", "language switch"):
        assert trigger in text


def test_the_realtime_prompt_carries_the_same_rule() -> None:
    """The deployed worker is speech-to-speech: the system prompt is the only
    place an accent can be specified, so the rule must survive into it."""
    from backend.app import agent_realtime

    text = agent_realtime.instructions().lower()
    assert "indian english and never changes" in text
    assert "re-anchor" in text


@pytest.mark.parametrize("language", list(Language))
def test_no_language_permits_an_american_accent(language: Language) -> None:
    assert "native indian" in tts_instructions(language).lower()


# --- latency: keep replies short -------------------------------------
#
# Measured on real calls: model ttft is 0.34s p50 / 0.60s p95, while the gap a
# caller experiences is 4.03s p50 / 16.07s p95. The model is not the wait — the
# reply being spoken is. So brevity is a latency setting, not a style note.


def test_the_prompt_treats_reply_length_as_latency() -> None:
    from backend.app import agent_realtime

    text = agent_realtime.instructions().lower()
    assert "one or two short sentences" in text
    assert "dead air" in text, "the reason must be stated, not just the rule"


def test_eagerness_defaults_to_the_low_latency_setting(monkeypatch) -> None:
    """`high` makes semantic VAD commit sooner to "the caller has finished".

    The env var is cleared first: a developer's own .env must not decide whether
    the shipped default is the low-latency one.
    """
    monkeypatch.delenv("OPENAI_REALTIME_EAGERNESS", raising=False)
    from backend.app.config import Settings

    assert Settings().openai_realtime_eagerness == "high"

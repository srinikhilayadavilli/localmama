"""LiveKit plugin wiring for the realtime voice path.

`AgentSession` takes LiveKit's own STT/TTS/VAD objects, so this module builds
those rather than the protocol adapters in `base.py`. The two layers coexist
on purpose: the protocols keep the *core* vendor-free and testable, while this
module handles the fact that LiveKit owns the realtime audio pipeline.

Every import is lazy and every failure is non-fatal, so `pip install
livekit-agents` alone is enough to run the agent — you only need a given
plugin package if you actually select that provider.

Indic-language notes (verify current coverage against each vendor's docs
before committing to one):
  - openai    — the default. gpt-4o-transcribe handles Indic audio and Indian
                code-switching well, and gpt-4o-mini-tts is the only hosted
                option here whose accent is steerable in free text, which is
                what gives Mami an Indian voice rather than an American one.
  - deepgram  — good English/Hindi; other Indic languages vary by model.
  - google    — broadest Indic STT/TTS coverage of the hosted options.
  - sarvam    — India-specific vendor; strongest Indic coverage if available.
Open-source targets to swap in later: AI4Bharat IndicWhisper (STT) and
Indic-Parler-TTS / AI4Bharat IndicTTS (TTS), fronted by your own HTTP service.
"""

from __future__ import annotations

from ..config import settings
from ..languages import LOCALE, Language, iso639_1
from ..logger import get_logger
from ..prompts.voice_style import GREETING_STT_PROMPT, tts_instructions

logger = get_logger("localmama.livekit.plugins")


def build_stt(language: Language):
    """Return a LiveKit STT instance for the selected provider, or None.

    None means "let AgentSession fall back to its own default", which keeps a
    misconfigured provider from taking the whole worker down.
    """
    provider = settings.stt_provider
    locale = LOCALE[language]

    try:
        if provider == "livekit":
            # LiveKit Cloud Inference: STT brokered through your LiveKit
            # account, so no separate vendor key is required. "multi" selects
            # the multilingual model, which is what lets a caller switch
            # between English and an Indian language mid-sentence.
            from livekit.agents import inference

            return inference.STT(model=settings.livekit_stt_model, language="multi")

        if provider == "deepgram":
            from livekit.plugins import deepgram

            # "multi" enables Deepgram's multilingual model rather than pinning
            # a single locale, which suits a call that may switch languages.
            return deepgram.STT(model="nova-3", language="multi")

        if provider == "google":
            from livekit.plugins import google

            return google.STT(languages=[locale])

        if provider == "openai":
            from livekit.plugins import openai

            # `detect_language` and `language` are mutually exclusive in the
            # plugin (detection blanks the language), so this starts in
            # whichever mode is configured and `agent.py` pins the language once
            # the caller has chosen one.
            kwargs: dict = {
                "model": settings.openai_stt_model,
                "use_realtime": settings.openai_stt_realtime,
                "prompt": GREETING_STT_PROMPT,
            }
            if settings.openai_stt_detect_language:
                kwargs["detect_language"] = True
            else:
                kwargs["language"] = iso639_1(language)
            return openai.STT(**kwargs)

        if provider == "sarvam":
            from livekit.plugins import sarvam

            # Sarvam's STT has no `update_options`, so its language is fixed for
            # the whole call and cannot follow a mid-call switch. It is taken
            # from STT_LANGUAGE rather than the session's starting language, so
            # you can point it at the language your callers actually speak.
            return sarvam.STT(language=settings.stt_language or locale)

    except ImportError as exc:
        logger.error(
            "STT provider %r selected but its plugin is not installed (%s). "
            "Install it, e.g. `pip install livekit-plugins-%s`.",
            provider,
            exc,
            provider,
        )
        return None
    except Exception as exc:  # noqa: BLE001 - never let provider setup kill the worker
        logger.error("failed to build STT provider %r: %s", provider, exc)
        return None

    if provider != "mock":
        logger.warning("no LiveKit STT adapter for provider %r", provider)
    return None


def build_tts(language: Language):
    """Return a LiveKit TTS instance for the selected provider, or None."""
    provider = settings.tts_provider
    locale = LOCALE[language]

    try:
        if provider == "livekit":
            # LiveKit Cloud Inference TTS. Voice is optional — omitting it uses
            # the model default, which avoids passing an id the model rejects.
            from livekit.agents import inference

            kwargs = {"model": settings.livekit_tts_model}
            if settings.livekit_tts_voice:
                kwargs["voice"] = settings.livekit_tts_voice
            if settings.livekit_tts_language:
                kwargs["language"] = settings.livekit_tts_language
            return inference.TTS(**kwargs)

        if provider == "cartesia":
            from livekit.plugins import cartesia

            return cartesia.TTS(language=iso639_1(language))

        if provider == "google":
            from livekit.plugins import google

            return google.TTS(language=locale)

        if provider == "openai":
            from livekit.plugins import openai

            # No `language` argument exists: the model infers the language from
            # the text it is given, and the accent comes entirely from
            # `instructions`. See prompts/voice_style.py.
            return openai.TTS(
                model=settings.openai_tts_model,
                voice=settings.openai_tts_voice,
                speed=settings.openai_tts_speed,
                instructions=(
                    settings.openai_tts_instructions or tts_instructions(language)
                ),
            )

        if provider == "elevenlabs":
            from livekit.plugins import elevenlabs

            return elevenlabs.TTS()

        if provider == "sarvam":
            from livekit.plugins import sarvam

            # Natively recorded Indic voices, and streaming over a websocket —
            # which is why a switch here fixes both the anglicised Telugu and a
            # chunk of the gap between turns. `speaker` is optional; blank uses
            # the model default rather than passing None through as a value.
            kwargs = {
                "target_language_code": locale,
                "model": settings.sarvam_tts_model,
                "pace": settings.sarvam_pace,
                "temperature": settings.sarvam_temperature,
            }
            if settings.sarvam_speaker:
                kwargs["speaker"] = settings.sarvam_speaker
            return sarvam.TTS(**kwargs)

    except ImportError as exc:
        logger.error(
            "TTS provider %r selected but its plugin is not installed (%s). "
            "Install it, e.g. `pip install livekit-plugins-%s`.",
            provider,
            exc,
            provider,
        )
        return None
    except Exception as exc:  # noqa: BLE001
        logger.error("failed to build TTS provider %r: %s", provider, exc)
        return None

    if provider != "mock":
        logger.warning("no LiveKit TTS adapter for provider %r", provider)
    return None


def build_vad():
    """Silero VAD for turn detection. Optional — None disables VAD."""
    try:
        from livekit.plugins import silero

        vad = silero.VAD.load()
        # Logged on success too: this used to be silent, so "is VAD even
        # running?" could only be answered by reading the source.
        logger.info("Silero VAD loaded")
        return vad
    except ImportError:
        logger.warning(
            "livekit-plugins-silero not installed; turn detection will rely on "
            "the STT provider's endpointing"
        )
        return None
    except Exception as exc:  # noqa: BLE001
        logger.error("failed to load Silero VAD: %s", exc)
        return None

"""Provider factory — the single place to swap STT/TTS vendors.

Selection is by name via `STT_PROVIDER` / `TTS_PROVIDER` in `.env`.
To add a provider: write an adapter satisfying the protocol in `base.py`,
register it in the dict below, done. Nothing else in the codebase changes.

The LiveKit-plugin factories are imported lazily so that the default `mock`
path never requires the vendor SDKs to be installed.
"""

from __future__ import annotations

from typing import Callable

from ..config import settings
from ..languages import Language
from ..logger import get_logger
from .mock import MockSTT, MockTTS, RuleLanguageDetector

logger = get_logger("localmama.providers")


def _mock_stt():
    return MockSTT()


def _mock_tts():
    return MockTTS()


STT_FACTORIES: dict[str, Callable[[], object]] = {"mock": _mock_stt}
TTS_FACTORIES: dict[str, Callable[[], object]] = {"mock": _mock_tts}


#: Providers that exist only inside the LiveKit AgentSession (built in
#: providers/livekit_plugins.py). They have no protocol-level adapter here, so
#: the browser/CLI paths fall back to mock — which is correct, not an error.
LIVEKIT_ONLY = {"livekit", "deepgram", "google", "openai", "sarvam",
                "cartesia", "elevenlabs"}


def get_stt():
    if settings.stt_provider in LIVEKIT_ONLY:
        return _mock_stt()
    factory = STT_FACTORIES.get(settings.stt_provider)
    if factory is None:
        logger.warning(
            "unknown STT_PROVIDER=%r, falling back to mock (available: %s)",
            settings.stt_provider,
            ", ".join(STT_FACTORIES),
        )
        factory = _mock_stt
    return factory()


def get_tts():
    if settings.tts_provider in LIVEKIT_ONLY:
        return _mock_tts()
    factory = TTS_FACTORIES.get(settings.tts_provider)
    if factory is None:
        logger.warning(
            "unknown TTS_PROVIDER=%r, falling back to mock (available: %s)",
            settings.tts_provider,
            ", ".join(TTS_FACTORIES),
        )
        factory = _mock_tts
    return factory()


def get_language_detector():
    return RuleLanguageDetector()


def describe() -> dict[str, str | list[str]]:
    """Current provider wiring, surfaced on the debug endpoint."""
    return {
        "stt": settings.stt_provider,
        "tts": settings.tts_provider,
        "language_detector": get_language_detector().name,
        "available_stt": sorted(STT_FACTORIES),
        "available_tts": sorted(TTS_FACTORIES),
        "supported_languages": [lang.value for lang in Language],
    }

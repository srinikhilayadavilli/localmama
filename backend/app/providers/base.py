"""Provider interfaces for speech-to-text, text-to-speech, and language ID.

Everything provider-specific lives behind these three protocols. The
conversation controller never imports a vendor SDK, so swapping Deepgram for a
self-hosted IndicWhisper is a change to `registry.py` plus one new adapter
file.

Protocols (rather than ABCs) mean an adapter does not need to inherit from
anything — it just needs the right shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Protocol, runtime_checkable

from ..languages import Language


@dataclass(frozen=True)
class TranscriptChunk:
    text: str
    is_final: bool
    language: Language | None = None
    confidence: float = 0.0


@dataclass(frozen=True)
class SynthesizedAudio:
    audio: bytes
    sample_rate: int
    mime_type: str = "audio/wav"


@runtime_checkable
class SpeechToTextProvider(Protocol):
    """Streaming or batch STT."""

    name: str

    def supports(self, language: Language) -> bool:
        """Whether this provider can transcribe the given language."""

    async def transcribe(
        self, audio: bytes, language: Language
    ) -> TranscriptChunk:
        """Transcribe a complete utterance."""

    async def stream(
        self, audio_stream: AsyncIterator[bytes], language: Language
    ) -> AsyncIterator[TranscriptChunk]:
        """Transcribe incrementally, yielding interim and final chunks."""


@runtime_checkable
class TextToSpeechProvider(Protocol):
    """Speech synthesis."""

    name: str

    def supports(self, language: Language) -> bool: ...

    async def synthesize(self, text: str, language: Language) -> SynthesizedAudio: ...


@runtime_checkable
class LanguageDetector(Protocol):
    """Identify which supported language an utterance is in."""

    name: str

    async def detect(self, text: str) -> Language | None: ...

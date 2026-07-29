"""Mock STT/TTS providers — the default, so the MVP runs with zero API keys.

These are explicitly *not* real speech engines. They exist so that:
  - the browser test UI can drive the full workflow using the browser's own
    Web Speech API for audio (see `frontend/app.js`), and
  - the test suite and text-chat debug mode can run the whole call flow with
    no network access.

Real audio paths use the LiveKit adapters in `livekit_plugins.py`.
"""

from __future__ import annotations

import wave
from io import BytesIO
from typing import AsyncIterator

from ..languages import Language, resolve_language
from ..providers.base import SynthesizedAudio, TranscriptChunk


class MockSTT:
    """Echoes back whatever text is handed to it as UTF-8 'audio'.

    Lets the workflow be driven end-to-end from text fixtures.
    """

    name = "mock-stt"

    def supports(self, language: Language) -> bool:  # noqa: ARG002
        return True

    async def transcribe(self, audio: bytes, language: Language) -> TranscriptChunk:
        text = audio.decode("utf-8", errors="ignore")
        return TranscriptChunk(text=text, is_final=True, language=language, confidence=1.0)

    async def stream(
        self, audio_stream: AsyncIterator[bytes], language: Language
    ) -> AsyncIterator[TranscriptChunk]:
        async for chunk in audio_stream:
            yield await self.transcribe(chunk, language)


class MockTTS:
    """Returns a short silent WAV so callers get real, playable bytes.

    Silence rather than a stub object means audio plumbing (content types,
    buffering, players) can be exercised without a TTS vendor.
    """

    name = "mock-tts"
    sample_rate = 16_000

    def supports(self, language: Language) -> bool:  # noqa: ARG002
        return True

    async def synthesize(self, text: str, language: Language) -> SynthesizedAudio:  # noqa: ARG002
        # ~60ms of silence per word, so timing roughly tracks utterance length.
        frames = int(self.sample_rate * 0.06 * max(len(text.split()), 1))
        buffer = BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(b"\x00\x00" * frames)
        return SynthesizedAudio(
            audio=buffer.getvalue(), sample_rate=self.sample_rate, mime_type="audio/wav"
        )


class RuleLanguageDetector:
    """Script- and keyword-based detection. No model, no network.

    Good enough for the MVP's "say your language" step, which is an explicit
    choice rather than a true language-ID problem.
    """

    name = "rule-language-detector"

    async def detect(self, text: str) -> Language | None:
        return resolve_language(text)

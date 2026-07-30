"""Agent configuration. Voice, turn-taking, and where the backend lives.

Nothing about leads, vendors, WhatsApp or Postgres appears here any more —
those are the backend's, and this process no longer has credentials for any of
them. What is left is what a phone call needs.

Environment variables, read once. There was a per-call config store behind
this for a while — a table, a history table, a CLI and an endpoint — and it
existed because the agent's image took minutes to build, so changing a voice
meant planning a deploy. That image is now four dependencies and no model
weights. Changing a voice is an environment variable and a restart, which is
faster than the machinery that existed to avoid it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


@dataclass(frozen=True)
class Settings:
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))

    # --- the lead backend ---
    #: e.g. https://localmama-backend.onrender.com. Without it the agent still
    #: answers calls and still talks — it simply has nowhere to send the lead,
    #: which is reported loudly at startup rather than discovered per call.
    backend_url: str = field(default_factory=lambda: _env("BACKEND_URL").rstrip("/"))
    backend_token: str = field(default_factory=lambda: _env("BACKEND_TOKEN"))
    #: How long shutdown waits for queued events to reach the backend. Anything
    #: still queued after this is genuinely lost, so it is generous — the call
    #: is over and nobody is listening.
    drain_seconds: float = field(
        default_factory=lambda: float(_env("DRAIN_SECONDS", "10"))
    )

    # --- LiveKit ---
    livekit_url: str = field(default_factory=lambda: _env("LIVEKIT_URL"))
    livekit_api_key: str = field(default_factory=lambda: _env("LIVEKIT_API_KEY"))
    livekit_api_secret: str = field(default_factory=lambda: _env("LIVEKIT_API_SECRET"))
    #: Registering under a name means the worker only takes jobs explicitly
    #: dispatched to it. There is deliberately no default: a dev run that
    #: inherits production's name becomes eligible for production dispatch, and
    #: real callers land on a laptop.
    livekit_agent_name: str = field(default_factory=lambda: _env("LIVEKIT_AGENT_NAME"))

    # --- the speech-to-speech model ---
    #: "openai" | "gemini".
    realtime_provider: str = field(
        default_factory=lambda: _env("REALTIME_PROVIDER", "openai").lower()
    )
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    openai_realtime_model: str = field(
        default_factory=lambda: _env("OPENAI_REALTIME_MODEL", "gpt-realtime")
    )
    #: "marin" and "cedar" are the newer, most natural gpt-realtime voices.
    #: None of them is Indian — the accent comes from the prompt.
    openai_realtime_voice: str = field(
        default_factory=lambda: _env("OPENAI_REALTIME_VOICE", "marin")
    )
    #: Semantic VAD eagerness: "low" waits longer before deciding the caller is
    #: done, "high" answers sooner. Semantic rather than plain server VAD is the
    #: point on Indian calls, where speakers pause between clauses.
    #:
    #: "high" is the setting for minimum dead air, and it is a real trade: a
    #: caller who pauses mid-thought is more likely to be answered over. Drop
    #: back to "auto" if people start getting cut off.
    openai_realtime_eagerness: str = field(
        default_factory=lambda: _env("OPENAI_REALTIME_EAGERNESS", "high").lower()
    )
    #: "near_field" for a phone handset, "far_field" for a laptop mic.
    openai_realtime_noise_reduction: str = field(
        default_factory=lambda: _env("OPENAI_REALTIME_NOISE_REDUCTION", "near_field")
    )
    openai_realtime_speed: float = field(
        default_factory=lambda: float(_env("OPENAI_REALTIME_SPEED", "1.0"))
    )
    #: Transcribes the caller's audio alongside the realtime model. Not for the
    #: logs: it is a second, independent decode, and the only evidence the
    #: backend's audit has that a captured value came from the caller.
    openai_stt_model: str = field(
        default_factory=lambda: _env("OPENAI_STT_MODEL", "gpt-4o-transcribe")
    )

    # --- Gemini Live (REALTIME_PROVIDER=gemini) ---
    google_api_key: str = field(default_factory=lambda: _env("GOOGLE_API_KEY"))
    #: Blank uses the plugin's default, which is a *native-audio* model: best
    #: voice, worst latency (measured ~6s from stop-speaking to reply). The
    #: latency-optimised variant measured ~0.5-1s on the same call — but it
    #: refuses generate_reply, so the caller has to speak first.
    gemini_realtime_model: str = field(
        default_factory=lambda: _env("GEMINI_REALTIME_MODEL")
    )
    gemini_voice: str = field(default_factory=lambda: _env("GEMINI_VOICE", "Puck"))
    gemini_language: str = field(default_factory=lambda: _env("GEMINI_LANGUAGE"))
    #: Gemini runs its OWN turn detection; the OpenAI settings do not apply to
    #: it. Silence before it decides the caller has finished is the single
    #: biggest cause of a long gap between turns.
    gemini_silence_ms: int = field(
        default_factory=lambda: int(_env("GEMINI_SILENCE_MS", "500"))
    )
    gemini_prefix_padding_ms: int = field(
        default_factory=lambda: int(_env("GEMINI_PREFIX_PADDING_MS", "200"))
    )
    gemini_end_sensitivity: str = field(
        default_factory=lambda: _env("GEMINI_END_SENSITIVITY", "high").lower()
    )

    # --- call lifecycle ---
    #: Tail after the closing line has finished playing, so the last audio
    #: frames reach the caller before the room disappears — ending on the final
    #: syllable sounds like a cut-off.
    hangup_grace_seconds: float = field(
        default_factory=lambda: float(_env("HANGUP_GRACE_SECONDS", "1.5"))
    )
    #: Backstop for the case where the closing line never comes. Without it a
    #: model that says nothing would leave the caller holding a live, metered
    #: line indefinitely.
    hangup_max_wait_seconds: float = field(
        default_factory=lambda: float(_env("HANGUP_MAX_WAIT_SECONDS", "25"))
    )
    #: Hard ceiling on a call that never completes. Nothing else ends one: the
    #: model can loop and the caller can go silent without hanging up, while
    #: both the SIP leg and the realtime session bill by the minute.
    max_call_seconds: float = field(
        default_factory=lambda: float(_env("MAX_CALL_SECONDS", "600"))
    )

    @property
    def livekit_configured(self) -> bool:
        return bool(self.livekit_url and self.livekit_api_key and self.livekit_api_secret)


settings = Settings()


def startup_problems() -> list[str]:
    """Misconfiguration that otherwise fails quietly at the point of use."""
    problems = []
    if not settings.livekit_configured:
        problems.append(
            "LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET are not all set — "
            "the worker cannot connect"
        )
    if not settings.livekit_agent_name:
        problems.append(
            "LIVEKIT_AGENT_NAME is not set — the worker would register unnamed and "
            "auto-join every room in the project"
        )
    if not (settings.backend_url and settings.backend_token):
        problems.append(
            "BACKEND_URL / BACKEND_TOKEN are not both set — calls will be answered "
            "and no lead will be recorded"
        )
    if settings.realtime_provider == "gemini" and not settings.google_api_key:
        problems.append("GOOGLE_API_KEY is not set; Gemini Live will not start")
    if settings.realtime_provider != "gemini" and not settings.openai_api_key:
        problems.append("OPENAI_API_KEY is not set; the OpenAI Realtime API will not start")
    return problems

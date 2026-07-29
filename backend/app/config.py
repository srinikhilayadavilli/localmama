"""Environment-driven configuration.

Plain dataclass over `os.getenv` — no settings framework, so there is nothing
to learn and nothing to break on a dependency bump.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Repo root: backend/app/config.py -> backend/app -> backend -> <root>
ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _env_bool(key: str, default: bool = False) -> bool:
    raw = _env(key)
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # --- storage ---
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR") or ROOT_DIR / "data"))

    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))

    # --- LLM backend (see services/llm_client.py) ---
    #: "anthropic" | "openai". The openai backend also speaks to anything
    #: OpenAI-compatible via LLM_BASE_URL (Sarvam, Groq, vLLM, Ollama).
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "anthropic"))
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    #: Override for OpenAI-compatible endpoints. Blank = OpenAI itself.
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL"))

    # --- LLM fallback for entity extraction ---
    anthropic_api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "claude-opus-5"))
    llm_extraction_enabled: bool = field(
        default_factory=lambda: _env_bool("LLM_EXTRACTION_ENABLED", True)
    )
    #: Let an LLM rephrase each turn naturally. Every generated line is still
    #: validated against the workflow state and falls back to the template.
    natural_replies_enabled: bool = field(
        default_factory=lambda: _env_bool("NATURAL_REPLIES", False)
    )
    #: Phrasing sits in the live audio path, so it is latency-bound rather than
    #: reasoning-bound. Measured medians for this task: Haiku 4.5 1.44s,
    #: Sonnet 5 2.42s, Opus 5 4.04s — and a caller hears every one of those
    #: seconds as silence. Haiku is the default; extraction keeps LLM_MODEL.
    phrasing_model: str = field(
        default_factory=lambda: _env("PHRASING_MODEL", "claude-haiku-4-5")
    )

    # --- returning-caller memory (see caller_profiles.py) ---
    caller_memory_enabled: bool = field(
        default_factory=lambda: _env_bool("CALLER_MEMORY", True)
    )
    #: Salt for hashing caller identifiers. Change this and all existing
    #: profiles become unreachable — which is also how you bulk-forget them.
    profile_salt: str = field(
        default_factory=lambda: _env("PROFILE_SALT", "local-mama-dev-salt")
    )

    # --- provider selection (see providers/livekit_plugins.py) ---
    stt_provider: str = field(default_factory=lambda: _env("STT_PROVIDER", "openai"))
    tts_provider: str = field(default_factory=lambda: _env("TTS_PROVIDER", "openai"))
    #: Fixed STT language for providers that cannot switch at runtime (Sarvam).
    stt_language: str = field(default_factory=lambda: _env("STT_LANGUAGE", "en-IN"))

    #: Seconds between the lead being saved and the agent ending the call.
    #: Long enough for the closing line to finish playing — cutting the room
    #: mid-sentence is worse than a moment of silence — short enough that the
    #: caller is not left holding a live, metered line.
    #: Tail after the closing line has finished playing, so the last audio
    #: frames reach the caller before the room disappears. This used to be the
    #: whole wait — a fixed 6s counted from save_lead, which fires before the
    #: goodbye is spoken and so cut it off mid-sentence.
    hangup_grace_seconds: float = field(
        default_factory=lambda: float(_env("HANGUP_GRACE_SECONDS", "1.5"))
    )
    #: Backstop for the case where the closing line never comes. Without it a
    #: model that says nothing would leave the caller holding a live, metered
    #: line indefinitely.
    hangup_max_wait_seconds: float = field(
        default_factory=lambda: float(_env("HANGUP_MAX_WAIT_SECONDS", "25"))
    )

    # --- Knowledge base / "brain" (services/brain.py) ---
    #: NeonDB DSN, shared with the Vaani bridge. Blank disables every lookup
    #: cleanly rather than erroring per call.
    database_url: str = field(default_factory=lambda: _env("DATABASE_URL"))
    #: Scope within the shared table. The two products live in one schema and
    #: are kept apart by these, so getting them wrong reads someone else's data
    #: rather than failing loudly — which is why they are explicit settings.
    brain_owner_id: str = field(default_factory=lambda: _env("BRAIN_OWNER_ID", "localmama"))
    brain_agent_id: str = field(default_factory=lambda: _env("BRAIN_AGENT_ID", "localmama"))
    #: Minimum blended score to count as a real match. Hybrid retrieval always
    #: returns its best rows, however bad — "fix my geyser" surfaced a tutoring
    #: service at 0.22 because the catalogue has no plumber. Measured on this
    #: data: a genuine match scores ~0.4+, coincidence sits ~0.2-0.3.
    brain_min_score: float = field(
        default_factory=lambda: float(_env("BRAIN_MIN_SCORE", "0.35"))
    )

    # --- WhatsApp lead handoff (services/whatsapp.py) ---
    #: Off until configured: a half-configured sender that 401s on every lead is
    #: worse than an honest "not sent" line in the log.
    whatsapp_enabled: bool = field(
        default_factory=lambda: _env_bool("WHATSAPP_ENABLED", False)
    )
    whatsapp_api_key: str = field(default_factory=lambda: _env("WHATSAPP_API_KEY"))
    whatsapp_template_name: str = field(
        default_factory=lambda: _env("WHATSAPP_TEMPLATE_NAME")
    )
    whatsapp_lang_code: str = field(
        default_factory=lambda: _env("WHATSAPP_LANG_CODE", "en_US")
    )
    whatsapp_api_url: str = field(
        default_factory=lambda: _env(
            "WHATSAPP_API_URL",
            "https://service.api.campaignbot.online/v1/whatsapp/message/send",
        )
    )
    #: Text for template param {{4}} before real vendor matches exist.
    whatsapp_param4: str = field(
        default_factory=lambda: _env(
            "WHATSAPP_PARAM4",
            "Our team is shortlisting the best options for you and will share "
            "them here shortly",
        )
    )

    # --- Sarvam (TTS_PROVIDER=sarvam) ---
    #: The Indic-native alternative. OpenAI's TTS is English-phonetic: it will
    #: read Telugu or Tamil in an English speaker's mouth no matter what the
    #: instructions say, because the accent lever steers delivery, not
    #: phonemes. Sarvam's bulbul voices are recorded natively per language, and
    #: its TTS streams over a websocket instead of OpenAI's ~1.3s
    #: time-to-first-audio. Blank speaker uses the model default.
    sarvam_speaker: str = field(default_factory=lambda: _env("SARVAM_SPEAKER"))
    sarvam_tts_model: str = field(
        default_factory=lambda: _env("SARVAM_TTS_MODEL", "bulbul:v3")
    )
    #: Speaking rate. 1.0 is what the Vaani deployment settled on against real
    #: phone calls; 0.9 read as dragging.
    sarvam_pace: float = field(default_factory=lambda: float(_env("SARVAM_PACE", "1.0")))
    #: Prosody variation. The plugin default of 0.6 is flat and monotone, which
    #: over a phone line is heard as both "robotic" and "unclear" — and pushing
    #: the pace up to compensate only trades clarity for speed. 0.85 is the
    #: value Vaani arrived at on the same voice and the same provider.
    sarvam_temperature: float = field(
        default_factory=lambda: float(_env("SARVAM_TEMPERATURE", "0.85"))
    )

    # --- OpenAI voice models (STT/TTS_PROVIDER=openai) ---
    #: "gpt-4o-transcribe" is the accurate one; "gpt-4o-mini-transcribe" is
    #: cheaper and a little faster but noticeably weaker on Indic audio over a
    #: phone line, which is exactly the case this product lives in.
    openai_stt_model: str = field(
        default_factory=lambda: _env("OPENAI_STT_MODEL", "gpt-4o-transcribe")
    )
    #: Stream transcripts over the realtime API instead of posting each VAD
    #: segment as a file. Partials arrive while the caller is still talking, so
    #: the first word of the reply comes noticeably sooner.
    openai_stt_realtime: bool = field(
        default_factory=lambda: _env_bool("OPENAI_STT_REALTIME", True)
    )
    #: Let the model detect the spoken language until the caller has actually
    #: picked one. Once they do, the agent pins the language for the rest of the
    #: call, which is more accurate than detecting on every turn.
    openai_stt_detect_language: bool = field(
        default_factory=lambda: _env_bool("OPENAI_STT_DETECT_LANGUAGE", True)
    )
    #: Only "gpt-4o-mini-tts" accepts `instructions`, and `instructions` is the
    #: only thing that produces an Indian accent — the older tts-1 voices are
    #: fixed American. Changing this model away from it loses the accent.
    openai_tts_model: str = field(
        default_factory=lambda: _env("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
    )
    #: OpenAI has no Indian voice; the accent comes from the instructions. The
    #: choice here is timbre only. "coral" is warm and female, which fits Mami;
    #: "sage" and "shimmer" are the other two worth auditioning.
    openai_tts_voice: str = field(
        default_factory=lambda: _env("OPENAI_TTS_VOICE", "coral")
    )
    #: Overrides the per-language accent instructions in prompts/voice_style.py.
    #: Set it to hand-tune the accent; leaving it blank keeps the per-language
    #: text, which is what follows a mid-call language switch.
    openai_tts_instructions: str = field(
        default_factory=lambda: _env("OPENAI_TTS_INSTRUCTIONS")
    )
    #: 1.0 is the model default. Indian English carries fine slightly slower.
    openai_tts_speed: float = field(
        default_factory=lambda: float(_env("OPENAI_TTS_SPEED", "1.0"))
    )

    # --- LiveKit ---
    livekit_url: str = field(default_factory=lambda: _env("LIVEKIT_URL"))
    livekit_api_key: str = field(default_factory=lambda: _env("LIVEKIT_API_KEY"))
    livekit_api_secret: str = field(default_factory=lambda: _env("LIVEKIT_API_SECRET"))
    #: Registering under a name means the worker only takes jobs explicitly
    #: dispatched to it. Unnamed workers auto-join every room in the project,
    #: which would hijack any other pipeline running on the same LiveKit
    #: account. Named + explicit dispatch keeps pipelines isolated.
    livekit_agent_name: str = field(
        default_factory=lambda: _env("LIVEKIT_AGENT_NAME", "local-mama")
    )
    #: The speech-to-speech worker, deployed alongside the deterministic one so
    #: both can be called and compared. Only used by the console to offer it as
    #: a second dispatch target; the worker itself reads LIVEKIT_AGENT_NAME.
    livekit_realtime_agent_name: str = field(
        default_factory=lambda: _env("LIVEKIT_REALTIME_AGENT_NAME", "local-mama-realtime")
    )
    #: LiveKit Cloud Inference model ids, used when STT/TTS_PROVIDER=livekit.
    #: These are brokered and billed through LiveKit, so no vendor keys needed.
    livekit_stt_model: str = field(
        default_factory=lambda: _env("LIVEKIT_STT_MODEL", "deepgram/nova-3")
    )
    livekit_tts_model: str = field(
        default_factory=lambda: _env("LIVEKIT_TTS_MODEL", "elevenlabs/eleven_multilingual_v2")
    )
    livekit_tts_voice: str = field(default_factory=lambda: _env("LIVEKIT_TTS_VOICE"))
    #: Language hint passed to the TTS model. For an Indian audience this is
    #: what nudges a multilingual model towards an Indian accent rather than
    #: the model's default (usually American or British).
    livekit_tts_language: str = field(
        default_factory=lambda: _env("LIVEKIT_TTS_LANGUAGE", "en")
    )

    # --- speech-to-speech realtime path (experimental, agent_realtime.py) ---
    #: "openai" | "gemini". Which vendor's realtime model runs that worker.
    realtime_provider: str = field(
        default_factory=lambda: _env("REALTIME_PROVIDER", "openai").lower()
    )
    openai_realtime_model: str = field(
        default_factory=lambda: _env("OPENAI_REALTIME_MODEL", "gpt-realtime")
    )
    #: "marin" and "cedar" are the newer, most natural gpt-realtime voices.
    #: As with TTS, none of them is Indian — the accent comes from the prompt.
    openai_realtime_voice: str = field(
        default_factory=lambda: _env("OPENAI_REALTIME_VOICE", "marin")
    )
    #: Semantic VAD eagerness: "low" waits longer before deciding the caller is
    #: done, "high" answers sooner. Semantic (rather than plain server VAD) is
    #: the point on Indian calls, where speakers pause between clauses.
    openai_realtime_eagerness: str = field(
        default_factory=lambda: _env("OPENAI_REALTIME_EAGERNESS", "auto").lower()
    )
    #: "near_field" for a phone handset, "far_field" for a laptop mic.
    openai_realtime_noise_reduction: str = field(
        default_factory=lambda: _env("OPENAI_REALTIME_NOISE_REDUCTION", "near_field")
    )
    openai_realtime_speed: float = field(
        default_factory=lambda: float(_env("OPENAI_REALTIME_SPEED", "1.0"))
    )

    # --- Gemini Live, speech-to-speech (REALTIME_PROVIDER=gemini) ---
    google_api_key: str = field(default_factory=lambda: _env("GOOGLE_API_KEY"))
    #: Blank uses the plugin's default, which is a *native-audio* model:
    #: best voice, worst latency (measured ~6s from stop-speaking to reply).
    #: "gemini-3.1-flash-live-preview" is the latency-optimised variant and
    #: measured ~0.5-1s on the same call — but it refuses generate_reply, so
    #: the agent cannot open the conversation and the caller must speak first.
    gemini_realtime_model: str = field(
        default_factory=lambda: _env("GEMINI_REALTIME_MODEL")
    )
    gemini_voice: str = field(default_factory=lambda: _env("GEMINI_VOICE", "Puck"))
    #: BCP-47 hint, e.g. "te-IN". Blank lets Gemini pick from the audio.
    gemini_language: str = field(default_factory=lambda: _env("GEMINI_LANGUAGE"))

    # --- turn-taking feel (tune these by ear on a real call) ---
    #: How long to wait after the caller stops before treating the turn as
    #: over. Too low cuts people off mid-sentence; too high feels dead.
    endpoint_min_delay: float = field(
        default_factory=lambda: float(_env("ENDPOINT_MIN_DELAY", "0.25"))
    )
    #: Hard ceiling on that wait. This was 5.0s, which is an eternity on a
    #: phone call — whenever the turn detector was unsure the caller simply sat
    #: in silence. 1.5s keeps a natural pause without abandoning them.
    endpoint_max_delay: float = field(
        default_factory=lambda: float(_env("ENDPOINT_MAX_DELAY", "1.5"))
    )
    #: How much speech is needed to interrupt Mami.
    interrupt_min_duration: float = field(
        default_factory=lambda: float(_env("INTERRUPT_MIN_DURATION", "0.4"))
    )
    #: How many recognised WORDS are needed to interrupt her. LiveKit defaults
    #: this to 0, which means duration alone decides — so a cough, a breath,
    #: traffic, or her own voice coming back through the caller's speakers all
    #: cut her off mid-sentence. 1 is the useful floor: noise produces no words
    #: and is ignored, while a caller cutting in with a single "Telugu" still
    #: gets through. Raising it to 2+ would block those one-word answers, which
    #: on this agent are most of what a caller actually says.
    interrupt_min_words: int = field(
        default_factory=lambda: int(_env("INTERRUPT_MIN_WORDS", "1"))
    )
    #: Seconds at the start and end of Mami's turn where a short "haan", "sari"
    #: or "ok" is treated as listening-along rather than an interruption. Indian
    #: callers backchannel constantly; without this they interrupt themselves.
    #: Widen it if she still stops when someone is only agreeing with her.
    backchannel_boundary: float = field(
        default_factory=lambda: float(_env("BACKCHANNEL_BOUNDARY", "1.0"))
    )
    #: "fixed" waits the same time every turn; "dynamic" lets the semantic
    #: detector shorten the wait when it is confident the caller has finished,
    #: within the min/max bounds above.
    endpointing_mode: str = field(
        default_factory=lambda: _env("ENDPOINTING_MODE", "dynamic").lower()
    )

    # --- Gemini Live turn taking (its OWN detector; the settings above do not
    # apply to it, which is why tuning them changed nothing on that path) ---
    #: Silence Gemini waits for before deciding the caller has finished. This
    #: is the single biggest cause of a long gap between turns.
    gemini_silence_ms: int = field(
        default_factory=lambda: int(_env("GEMINI_SILENCE_MS", "500"))
    )
    #: Audio kept from before speech onset, so the first word is not clipped.
    gemini_prefix_padding_ms: int = field(
        default_factory=lambda: int(_env("GEMINI_PREFIX_PADDING_MS", "200"))
    )
    #: "high" ends the turn more eagerly (snappier); "low" waits longer.
    gemini_end_sensitivity: str = field(
        default_factory=lambda: _env("GEMINI_END_SENSITIVITY", "high").lower()
    )

    @property
    def leads_dir(self) -> Path:
        return self.data_dir / "leads"

    @property
    def transcripts_dir(self) -> Path:
        return self.data_dir / "transcripts"

    @property
    def profiles_dir(self) -> Path:
        return self.data_dir / "profiles"

    @property
    def _has_llm_key(self) -> bool:
        if self.llm_provider == "openai":
            return bool(self.openai_api_key)
        return bool(self.anthropic_api_key)

    @property
    def llm_available(self) -> bool:
        return self._has_llm_key and self.llm_extraction_enabled

    @property
    def natural_replies_available(self) -> bool:
        return self._has_llm_key and self.natural_replies_enabled

    @property
    def whatsapp_available(self) -> bool:
        """All three are required: the flag alone sends nothing useful."""
        return bool(
            self.whatsapp_enabled and self.whatsapp_api_key and self.whatsapp_template_name
        )

    @property
    def livekit_configured(self) -> bool:
        return bool(self.livekit_url and self.livekit_api_key and self.livekit_api_secret)

    def ensure_dirs(self) -> None:
        self.leads_dir.mkdir(parents=True, exist_ok=True)
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()

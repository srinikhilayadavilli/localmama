"""Backend configuration. Environment only — this service has no callers to
protect from a redeploy, so there is no per-request config store.

Everything here is a secret or a deployment fact. The knobs that used to live
in Postgres because changing them meant a redeploy are all on the *agent* side:
voice, model, turn-taking. Nothing about how a lead is processed is worth
changing without a review.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _env_bool(key: str, default: bool = False) -> bool:
    raw = _env(key)
    return raw.lower() in {"1", "true", "yes", "on"} if raw else default


@dataclass(frozen=True)
class Settings:
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))
    port: int = field(default_factory=lambda: int(_env("PORT", "8000")))

    # --- the shared token the agent presents ---
    #: There is exactly one client. A bearer token checked in constant time is
    #: proportionate; mTLS would be better and is more moving parts than this
    #: deployment currently earns.
    agent_token: str = field(default_factory=lambda: _env("AGENT_TOKEN"))

    # --- Postgres (Neon) ---
    database_url: str = field(default_factory=lambda: _env("DATABASE_URL"))
    #: Render runs the web service and the outbox worker as separate processes,
    #: each with its own pool. Neon's connection ceiling is the constraint, not
    #: this service's concurrency.
    db_pool_min: int = field(default_factory=lambda: int(_env("DB_POOL_MIN", "1")))
    db_pool_max: int = field(default_factory=lambda: int(_env("DB_POOL_MAX", "8")))

    # --- knowledge base scope ---
    brain_owner_id: str = field(default_factory=lambda: _env("BRAIN_OWNER_ID", "localmama"))
    brain_agent_id: str = field(default_factory=lambda: _env("BRAIN_AGENT_ID", "localmama"))

    # --- Sarvam, for normalising captured values into English ---
    sarvam_api_key: str = field(default_factory=lambda: _env("SARVAM_API_KEY"))
    translate_enabled: bool = field(
        default_factory=lambda: _env_bool("TRANSLATE_ENABLED", True)
    )
    #: Nothing is waiting on the line any more, so this is generous where the
    #: agent's had to be 2s. A better romanisation is worth a slower one.
    translate_timeout: float = field(
        default_factory=lambda: float(_env("TRANSLATE_TIMEOUT", "8.0"))
    )

    # --- WhatsApp handoff ---
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
    whatsapp_param4: str = field(
        default_factory=lambda: _env(
            "WHATSAPP_PARAM4",
            "Our team is shortlisting the best options for you and will share "
            "them here shortly",
        )
    )

    # --- accuracy ---
    #: Below this a captured field is flagged for human review rather than
    #: trusted. Not a rejection: the lead is still actionable and still sent,
    #: it just carries a note. See `pipeline.audit`.
    review_threshold: float = field(
        default_factory=lambda: float(_env("REVIEW_THRESHOLD", "0.75"))
    )
    #: How long a transcript is kept. It contains everything the caller said,
    #: which is personal data under the DPDP Act, and it is only needed for the
    #: confidence audit and for debugging a bad lead. 0 keeps them forever.
    transcript_retention_days: int = field(
        default_factory=lambda: int(_env("TRANSCRIPT_RETENTION_DAYS", "30"))
    )

    # --- observability ---
    #: Gates /metrics and /v1/ops/*. Separate from the agent's token because
    #: they are different principals with different lifetimes: the agent holds
    #: one secret to file leads, a scraper holds another to read aggregates,
    #: and rotating either should not silence the other. Unset refuses the ops
    #: surface entirely rather than opening it — these endpoints carry the
    #: business's unit economics.
    ops_token: str = field(default_factory=lambda: _env("OPS_TOKEN"))
    #: For display only. Every stored price is USD, because that is what every
    #: provider bills in; this converts for the people reading the dashboard,
    #: who think in rupees. Deliberately not applied to anything stored — a
    #: rate that moves would otherwise silently restate history.
    inr_per_usd: float = field(
        default_factory=lambda: float(_env("INR_PER_USD", "88.0"))
    )
    #: How long the per-response turn series is kept. It is the largest thing
    #: this schema writes — tens of rows per call against a handful for the
    #: ledger — and its value is diagnostic, which decays in days. The usage
    #: ledger itself is never swept: it is the billing record.
    turn_retention_days: int = field(
        default_factory=lambda: int(_env("TURN_RETENTION_DAYS", "90"))
    )
    #: Cost per call above which a call is worth a human look. Not an alert on
    #: its own — a long, successful call is allowed to be expensive — but the
    #: dashboard ranks by it, because a runaway context looks exactly like an
    #: ordinary call until you sort by price.
    cost_alert_usd: float = field(
        default_factory=lambda: float(_env("COST_ALERT_USD", "0.75"))
    )

    # --- outbox ---
    outbox_sweep_seconds: float = field(
        default_factory=lambda: float(_env("OUTBOX_SWEEP_SECONDS", "300"))
    )
    outbox_max_attempts: int = field(
        default_factory=lambda: int(_env("OUTBOX_MAX_ATTEMPTS", "25"))
    )

    @property
    def whatsapp_available(self) -> bool:
        """All three are required: the flag alone sends nothing useful."""
        return bool(
            self.whatsapp_enabled and self.whatsapp_api_key and self.whatsapp_template_name
        )


settings = Settings()


def missing_required() -> list[str]:
    """Configuration this service cannot run without.

    Checked at startup and surfaced on `/healthz`, because every one of these
    fails silently at the point of use: no token means the API is open, no DSN
    means leads are accepted and dropped.
    """
    problems = []
    if not settings.database_url:
        problems.append("DATABASE_URL is not set — leads cannot be stored")
    if not settings.agent_token:
        problems.append("AGENT_TOKEN is not set — the API would accept any caller")
    return problems

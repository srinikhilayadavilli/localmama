"""Durable lead storage in Postgres.

`persistence.py` writes leads as JSON under `DATA_DIR`, which is right for the
laptop and wrong everywhere else: on LiveKit Cloud and Render free that path is
container-local and ephemeral, so real leads were captured, written, and then
lost on the next deploy — and the `/admin` page, running in a different
container, read its own empty directory and showed nothing.

This writes the same lead to the shared Neon database instead, in its own
`localmama` schema (the knowledge base lives in `utter`, and mixing a
write-heavy operational table into it would couple two products' migrations).

Additive: the JSON write still happens. If Postgres is unreachable the lead is
still on disk and in the logs, so a database problem degrades to the old
behaviour rather than dropping a customer's request.
"""

from __future__ import annotations

import json

from ..config import settings
from ..logger import get_logger
from ..models import Lead

logger = get_logger("localmama.leads")

_DDL = """
CREATE SCHEMA IF NOT EXISTS localmama;
CREATE TABLE IF NOT EXISTS localmama.leads (
    session_id   TEXT PRIMARY KEY,
    agent_id     TEXT NOT NULL,
    caller_phone TEXT,
    language     TEXT,
    name         TEXT,
    service      TEXT,
    city         TEXT,
    status       TEXT NOT NULL,
    transcript   JSONB NOT NULL DEFAULT '[]'::jsonb,
    started_at   TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_leads_agent   ON localmama.leads(agent_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_leads_phone   ON localmama.leads(caller_phone);
-- The outbox. A lead whose WhatsApp handoff has not succeeded is work still
-- owed to a customer, so the state lives next to the lead rather than in a
-- process that dies when the call ends.
ALTER TABLE localmama.leads ADD COLUMN IF NOT EXISTS whatsapp_status TEXT NOT NULL DEFAULT 'pending';
ALTER TABLE localmama.leads ADD COLUMN IF NOT EXISTS whatsapp_attempts INTEGER NOT NULL DEFAULT 0;
ALTER TABLE localmama.leads ADD COLUMN IF NOT EXISTS whatsapp_error TEXT;
ALTER TABLE localmama.leads ADD COLUMN IF NOT EXISTS whatsapp_at TIMESTAMPTZ;
CREATE INDEX IF NOT EXISTS idx_leads_outbox ON localmama.leads(agent_id, whatsapp_status)
    WHERE whatsapp_status = 'pending';
"""

_schema_ready = False


def available() -> bool:
    return bool(settings.database_url)


def _connect():
    import psycopg

    return psycopg.connect(settings.database_url, connect_timeout=5)


def _ensure_schema(conn) -> None:
    global _schema_ready
    if _schema_ready:
        return
    with conn.cursor() as cur:
        cur.execute(_DDL)
    conn.commit()
    _schema_ready = True


def save(lead: Lead, caller_phone: str = "") -> bool:
    """Upsert a lead. Returns whether it reached Postgres.

    Never raises. Keyed by session_id so a repeated save is an update, not a
    duplicate row — the same property `save_lead` relies on for idempotency.
    """
    if not available():
        return False
    try:
        with _connect() as conn:
            _ensure_schema(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO localmama.leads (session_id, agent_id, caller_phone,"
                    " language, name, service, city, status, transcript, started_at,"
                    " completed_at)"
                    " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
                    " ON CONFLICT (session_id) DO UPDATE SET"
                    "   caller_phone = EXCLUDED.caller_phone,"
                    "   language = EXCLUDED.language, name = EXCLUDED.name,"
                    "   service = EXCLUDED.service, city = EXCLUDED.city,"
                    "   status = EXCLUDED.status, transcript = EXCLUDED.transcript,"
                    "   completed_at = EXCLUDED.completed_at",
                    (
                        lead.session_id,
                        settings.brain_agent_id,
                        caller_phone or None,
                        lead.selected_language.value if lead.selected_language else None,
                        lead.user_name,
                        lead.requested_service,
                        lead.city_or_area,
                        lead.conversation_status.value,
                        json.dumps(
                            [t.model_dump(mode="json") for t in lead.transcript],
                            ensure_ascii=False,
                        ),
                        lead.started_at,
                        lead.completed_at,
                    ),
                )
            conn.commit()
        logger.info("lead %s stored in postgres", lead.session_id[:8])
        return True
    except Exception as exc:  # noqa: BLE001 - the lead is already on disk
        logger.warning(
            "could not store lead %s in postgres (%s); it is still on disk",
            lead.session_id[:8], exc,
        )
        return False


def mark_whatsapp(session_id: str, ok: bool, error: str = "") -> None:
    """Record the outcome of a handoff attempt. Never raises.

    `sent` is terminal. `pending` means it is still owed and will be retried;
    `skipped` means there was nothing to send to (no phone number), which is not
    a failure and must not be retried forever.
    """
    if not available():
        return
    status = "sent" if ok else ("skipped" if error == "no phone" else "pending")
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE localmama.leads SET whatsapp_status = %s,"
                " whatsapp_attempts = whatsapp_attempts + 1,"
                " whatsapp_error = %s, whatsapp_at = now()"
                " WHERE session_id = %s",
                (status, (error or None), session_id),
            )
            conn.commit()
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not record whatsapp status: %s", exc)


def pending_whatsapp(limit: int = 50, max_attempts: int = 25) -> list[dict]:
    """Leads still owed a WhatsApp message, oldest first.

    `max_attempts` stops a permanently broken lead being retried forever — a
    bad phone number would otherwise be attempted on every worker start for
    the rest of the deployment's life.
    """
    if not available():
        return []
    try:
        with _connect() as conn:
            _ensure_schema(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT session_id, caller_phone, name, service, city, language,"
                    " whatsapp_attempts FROM localmama.leads"
                    " WHERE agent_id = %s AND whatsapp_status = 'pending'"
                    "   AND caller_phone IS NOT NULL"
                    "   AND whatsapp_attempts < %s"
                    " ORDER BY created_at LIMIT %s",
                    (settings.brain_agent_id, max_attempts, limit),
                )
                cols = ["session_id", "caller_phone", "name", "service", "city",
                        "language", "attempts"]
                return [dict(zip(cols, row)) for row in cur.fetchall()]
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not read the outbox: %s", exc)
        return []

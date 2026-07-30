"""Persistence for events and leads.

Two tables, one idea: the event log is what the agent said happened, the lead
row is what we made of it. Keeping both means a lead that comes out wrong can
be diagnosed, and re-processed, without another phone call.

Nothing here raises on a database problem *except* `record_events` — the API
must be able to tell the agent an event was not stored, or the agent will drop
it and the lead is gone. Everywhere else a failure degrades to a warning,
because by then the lead is already durable.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from contract import CapturedField, Event
from psycopg.types.json import Jsonb

from . import db
from .config import settings
from .logger import get_logger

logger = get_logger("localmama.store")


def record_events(events: list[Event]) -> tuple[list[str], list[str]]:
    """Store events. Returns (accepted_ids, duplicate_ids).

    `ON CONFLICT DO NOTHING` on the primary key is the whole deduplication
    strategy: the agent may retry any event any number of times, and Postgres
    decides which attempt was first. `RETURNING` tells us which rows were
    genuinely new, so the acknowledgement is accurate rather than optimistic.
    """
    if not events:
        return [], []

    # One statement, so the answer is exact: `RETURNING` after
    # `ON CONFLICT DO NOTHING` yields precisely the rows this call inserted.
    # Anything in the batch and not in that set was already there — a retry.
    tuples = ", ".join(["(%s, %s, %s, %s, %s)"] * len(events))
    params: list[Any] = []
    for e in events:
        params += [e.event_id, e.call_id, e.seq, e.type, Jsonb(e.model_dump(mode="json"))]

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO localmama.call_events"
                " (event_id, call_id, seq, type, payload)"
                f" VALUES {tuples}"
                " ON CONFLICT (event_id) DO NOTHING"
                " RETURNING event_id",
                params,
            )
            inserted = {r[0] for r in cur.fetchall()}
        conn.commit()

    accepted = [e.event_id for e in events if e.event_id in inserted]
    duplicates = [e.event_id for e in events if e.event_id not in inserted]
    return accepted, duplicates


def upsert_call(call_id: str, **fields: Any) -> None:
    """Merge fields onto a lead row, creating it if this is the first event.

    Events may arrive in any order — a `call.captured` can beat the
    `call.started` that names the caller — so every write is a merge and the
    row is created by whichever event turns up first.
    """
    if not fields:
        return
    fields["updated_at"] = datetime.now().astimezone()
    columns = ["call_id", "agent_id", *fields.keys()]
    values = [call_id, settings.brain_agent_id, *fields.values()]
    placeholders = ", ".join(["%s"] * len(columns))
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in fields)
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO localmama.leads ({', '.join(columns)})"
                f" VALUES ({placeholders})"
                f" ON CONFLICT (call_id) DO UPDATE SET {updates}",
                values,
            )
        conn.commit()


def merge_raw(call_id: str, field: CapturedField, value: str) -> None:
    """Record one captured value, in the caller's own words.

    `jsonb_set`-style merge rather than read-modify-write: two captures for the
    same call can be in flight at once (the agent fires them without awaiting),
    and a read-modify-write would lose one.
    """
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO localmama.leads (call_id, agent_id, raw)"
                " VALUES (%s, %s, %s)"
                " ON CONFLICT (call_id) DO UPDATE SET"
                "   raw = localmama.leads.raw || EXCLUDED.raw,"
                "   updated_at = now()",
                (call_id, settings.brain_agent_id, Jsonb({field.value: value})),
            )
        conn.commit()


def get_lead(call_id: str) -> dict | None:
    with db.cursor() as cur:
        cur.execute(
            "SELECT call_id, caller_phone, dialled, language, raw, name, service,"
            " city, status, confirmed, confidence, needs_review, vendors,"
            " whatsapp_status, transcript, started_at, ended_at"
            " FROM localmama.leads WHERE call_id = %s",
            (call_id,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    cols = ["call_id", "caller_phone", "dialled", "language", "raw", "name",
            "service", "city", "status", "confirmed", "confidence", "needs_review",
            "vendors", "whatsapp_status", "transcript", "started_at", "ended_at"]
    return dict(zip(cols, row))


def mark_whatsapp(call_id: str, ok: bool, error: str = "") -> None:
    """Record the outcome of a handoff attempt. Never raises.

    `sent` is terminal. `pending` means it is still owed and will be retried;
    `skipped` means there was nothing to send to, or nothing to send with —
    neither is a failure and neither improves by being retried forever.
    """
    status = (
        "sent" if ok
        else ("skipped" if error in {"no phone", "not configured"} else "pending")
    )
    try:
        with db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE localmama.leads SET whatsapp_status = %s,"
                    " whatsapp_attempts = whatsapp_attempts + 1,"
                    " whatsapp_error = %s, whatsapp_at = now(), updated_at = now()"
                    " WHERE call_id = %s",
                    (status, error or None, call_id),
                )
            conn.commit()
    except Exception as exc:  # noqa: BLE001 - the lead is already stored
        logger.warning("could not record whatsapp status for %s: %s", call_id[:8], exc)


def claim_owed_handoffs(limit: int = 50, stale_after_minutes: int = 10) -> list[dict]:
    """Take ownership of owed handoffs, atomically, and return them.

    The sweep runs unattended, and a second worker sweeping at the same moment
    would send one customer the same message twice. So rows are claimed rather
    than merely read: one `UPDATE ... RETURNING` moves them from `pending` to
    `sending`, and Postgres decides who wins.

    A claim that is never resolved — the process died mid-send — would strand
    the lead in `sending` forever, so anything stuck there longer than
    `stale_after_minutes` is treated as owed again. That risks a duplicate in
    the narrow window where a send succeeded but the result was never recorded;
    a duplicate is recoverable, a lead silently never sent is not.
    """
    try:
        with db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE localmama.leads SET whatsapp_status = 'sending',"
                    " whatsapp_at = now()"
                    " WHERE call_id IN ("
                    "   SELECT call_id FROM localmama.leads"
                    "   WHERE agent_id = %s AND caller_phone IS NOT NULL"
                    "     AND whatsapp_attempts < %s"
                    "     AND (whatsapp_status = 'pending'"
                    "          OR (whatsapp_status = 'sending'"
                    "              AND whatsapp_at < now() - make_interval(mins => %s)))"
                    "   ORDER BY created_at"
                    "   FOR UPDATE SKIP LOCKED"
                    "   LIMIT %s)"
                    " RETURNING call_id, caller_phone, name, service, city, language,"
                    "           vendors, whatsapp_attempts",
                    (settings.brain_agent_id, settings.outbox_max_attempts,
                     stale_after_minutes, limit),
                )
                cols = ["call_id", "caller_phone", "name", "service", "city",
                        "language", "vendors", "attempts"]
                claimed = [dict(zip(cols, r)) for r in cur.fetchall()]
            conn.commit()
        if claimed:
            logger.info("outbox: claimed %d owed handoff(s)", len(claimed))
        return claimed
    except Exception as exc:  # noqa: BLE001 - a sweep must never kill the worker
        logger.warning("could not claim from the outbox: %s", exc)
        return []


def pending_handoffs(limit: int = 50) -> list[dict]:
    """Leads still owed a WhatsApp message, oldest first. Read-only."""
    try:
        with db.cursor() as cur:
            cur.execute(
                "SELECT call_id, caller_phone, name, service, city,"
                " whatsapp_status, whatsapp_attempts, whatsapp_error"
                " FROM localmama.leads"
                " WHERE agent_id = %s AND whatsapp_status IN ('pending', 'sending')"
                "   AND caller_phone IS NOT NULL AND whatsapp_attempts < %s"
                " ORDER BY created_at LIMIT %s",
                (settings.brain_agent_id, settings.outbox_max_attempts, limit),
            )
            cols = ["call_id", "caller_phone", "name", "service", "city",
                    "status", "attempts", "error"]
            return [dict(zip(cols, r)) for r in cur.fetchall()]
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not read the outbox: %s", exc)
        return []


def expire_transcripts() -> int:
    """Clear transcripts past the retention window. Returns rows cleared.

    A transcript is everything the caller said, which is personal data under
    the DPDP Act. It is needed for the confidence audit and for diagnosing a
    bad lead, and after that it is a liability — so it is cleared while the
    lead itself, which is the business record, is kept.
    """
    days = settings.transcript_retention_days
    if days <= 0:
        return 0
    try:
        with db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE localmama.leads SET transcript = '[]'::jsonb"
                    " WHERE transcript <> '[]'::jsonb"
                    "   AND ended_at < now() - make_interval(days => %s)",
                    (days,),
                )
                cleared = cur.rowcount
            conn.commit()
        if cleared:
            logger.info("cleared %d transcript(s) past %d-day retention", cleared, days)
        return cleared
    except Exception as exc:  # noqa: BLE001
        logger.warning("transcript expiry failed: %s", exc)
        return 0


def transcript_for(call_id: str) -> list[str]:
    """Just the caller's turns, oldest first — the evidence the audit needs."""
    with db.cursor() as cur:
        cur.execute("SELECT transcript FROM localmama.leads WHERE call_id = %s", (call_id,))
        row = cur.fetchone()
    if not row or not row[0]:
        return []
    turns = row[0] if isinstance(row[0], list) else json.loads(row[0])
    return [t["text"] for t in turns if t.get("role") == "caller" and t.get("text")]

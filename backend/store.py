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

#: Outcomes that will never improve by trying again. Everything else stays
#: `pending` for the outbox to sweep.
_NOT_WORTH_RETRYING = frozenset({
    "no phone",            # anonymous caller — there is nowhere to send
    "incomplete lead",     # no service; the message would have nothing to say
    "service unverified",  # unconfirmed, and we cannot vouch for the trade
})

#: Deliberately NOT in the set above. An unconfigured webhook is a state of
#: this deployment, not of the lead — it changes the moment someone sets the
#: URL and secret, and every lead that arrived in the meantime is still owed a
#: delivery. Marking those terminal loses them permanently for a reason that
#: has since gone away.
_CONFIG_WILL_CHANGE = "not configured"


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


def _digits(raw: str | None) -> str:
    """A phone number reduced to what actually identifies it.

    `dialled` is whatever the trunk put in `sip.trunkPhoneNumber`, only
    stripped — Indian SIP providers variously present `+918071581496`,
    `918071581496`, `08071581496` and `sip:+918071581496@host`. Matching those
    as strings against a stored E.164 form misses, and a miss is silent: the
    call is filed under the default tenant with one INFO line.
    """
    digits = "".join(ch for ch in str(raw or "") if ch.isdigit()).lstrip("0")
    # The same rule the senders apply to a caller's number: a bare ten-digit
    # Indian number is missing its country code, and "08071581496" is the
    # national form of "+918071581496". Without this the national form matches
    # nothing and every call presented that way is filed under the default.
    if len(digits) == 10:
        digits = "91" + digits
    return digits


def agent_for_did(dialled: str | None) -> str | None:
    """Which tenant answers this number. Never raises.

    The number the caller rang is the routing key — a business owns its DID,
    and that ownership is the only thing about a call that identifies whose
    lead it is before a word is spoken.

    Returns None when the answer is *unknown* rather than "nobody owns this".
    Those are different, and conflating them was a bug: a Neon blip during
    `call.started` returned the default tenant, `upsert_call` wrote it, and
    nothing ever re-resolved — `call.started` is deduplicated by `event_id` and
    no other event carries the number. One transient error misfiled a lead
    under the wrong business permanently. None lets the caller leave the column
    alone and resolve again later, which `pipeline` does.

    A number genuinely mapped to nobody still returns the default: a call on an
    unclaimed number is a lead worth keeping.
    """
    if not dialled:
        return settings.brain_agent_id
    if not db.available():
        return None
    try:
        with db.cursor() as cur:
            cur.execute(
                "SELECT n.agent_id FROM localmama.tenant_numbers n"
                "  JOIN localmama.tenants t ON t.agent_id = n.agent_id"
                " WHERE regexp_replace(n.did, '\\D', '', 'g') "
                "       = regexp_replace(%s, '^0+', '')"
                "   AND n.active AND t.active",
                (_digits(dialled),),
            )
            row = cur.fetchone()
        if row:
            return row[0]
        logger.info("no tenant owns %s; filing under %s",
                    dialled, settings.brain_agent_id)
        return settings.brain_agent_id
    except Exception as exc:  # noqa: BLE001 - unknown, not "the default"
        logger.warning("could not resolve the tenant for %s: %s", dialled, exc)
        return None


def active_agents() -> list[str]:
    """Tenants with somewhere to deliver a webhook. Never raises.

    Used by the sweep to decide which leads are worth claiming. Claiming one
    for a tenant with no endpoint still increments its attempt count, and
    `OUTBOX_MAX_ATTEMPTS` is permanent.
    """
    if not db.available():
        return []
    try:
        with db.cursor() as cur:
            cur.execute(
                "SELECT s.agent_id FROM localmama.webhook_subscriptions s"
                "  JOIN localmama.tenants t ON t.agent_id = s.agent_id"
                " WHERE s.active AND t.active"
            )
            return [r[0] for r in cur.fetchall()]
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not list tenants with a webhook: %s", exc)
        return []


def upsert_call(call_id: str, *, agent_id: str | None = None, **fields: Any) -> None:
    """Merge fields onto a lead row, creating it if this is the first event.

    Events may arrive in any order — a `call.captured` can beat the
    `call.started` that names the caller — so every write is a merge and the
    row is created by whichever event turns up first.

    `agent_id` is written only when given, which in practice means only by
    `call.started`, the one event carrying the dialled number. A row created
    early by a capture gets the default and is corrected when `call.started`
    lands; passing it on every write would let a later event silently reassign
    a lead to another tenant.
    """
    if not fields:
        return
    fields["updated_at"] = datetime.now().astimezone()
    columns = ["call_id", "agent_id", *fields.keys()]
    values = [call_id, agent_id or settings.brain_agent_id, *fields.values()]
    placeholders = ", ".join(["%s"] * len(columns))
    # `agent_id` is always inserted and only *updated* when explicitly given,
    # which is why it is not simply another entry in `fields`.
    on_conflict = [*fields.keys()] + (["agent_id"] if agent_id else [])
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in on_conflict)
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


def record_usage(call_id: str, source: str, records: list) -> int:
    """Store what a call consumed. Returns rows written. Never raises.

    Upsert rather than insert, because the agent's records are whole-call
    *totals* keyed on ref='session'. A retried `call.ended` must restate them,
    not add a second call's worth — so quantity is replaced, not summed. The
    backend's own rows carry a unique ref each and so never collide.

    Best-effort by construction. This runs after the lead is already durable,
    and a cost number is never worth failing an event that carries a lead.
    """
    if not records:
        return 0
    tuples = ", ".join(["(%s, %s, %s, %s, %s, %s, %s, %s, %s)"] * len(records))
    params: list[Any] = []
    for r in records:
        params += [call_id, r.ref, source, r.provider, r.model, r.operation,
                   r.unit.value if hasattr(r.unit, "value") else str(r.unit),
                   float(r.quantity), bool(getattr(r, "estimated", False))]
    try:
        with db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO localmama.usage"
                    " (call_id, ref, source, provider, model, operation, unit,"
                    "  quantity, estimated)"
                    f" VALUES {tuples}"
                    " ON CONFLICT (call_id, ref, provider, model, operation, unit)"
                    " DO UPDATE SET quantity = EXCLUDED.quantity,"
                    "               estimated = EXCLUDED.estimated",
                    params,
                )
            conn.commit()
        return len(records)
    except Exception as exc:  # noqa: BLE001 - the lead is already stored
        logger.warning("could not record usage for %s: %s", call_id[:8], exc)
        return 0


def record_turns(call_id: str, turns: list) -> int:
    """Store the per-response series. Returns rows written. Never raises.

    `DO NOTHING` rather than an update: a turn is immutable once it happened,
    and the only way the same (call_id, ref) arrives twice is a retried
    `call.ended` carrying the identical row.
    """
    if not turns:
        return 0
    tuples = ", ".join(["(%s, %s, %s, %s, %s, %s, %s, %s, %s)"] * len(turns))
    params: list[Any] = []
    for t in turns:
        params += [call_id, t.ref, float(t.at), float(t.ttft), float(t.duration),
                   int(t.input_tokens), int(t.cached_tokens), int(t.output_tokens),
                   bool(t.cancelled)]
    try:
        with db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO localmama.call_turns"
                    " (call_id, ref, at, ttft, duration, input_tokens,"
                    "  cached_tokens, output_tokens, cancelled)"
                    f" VALUES {tuples}"
                    " ON CONFLICT (call_id, ref) DO NOTHING",
                    params,
                )
            conn.commit()
        return len(turns)
    except Exception as exc:  # noqa: BLE001 - diagnostics, not the lead
        logger.warning("could not record turns for %s: %s", call_id[:8], exc)
        return 0


def record_service_call(
    call_id: str | None, ref: str, provider: str, unit: str, quantity: float,
    *, model: str = "", operation: str = "", ok: bool = True,
    latency_ms: int = 0, error: str = "",
) -> None:
    """One provider call the backend made. Never raises.

    Failures are recorded too, and with their own quantity. A translator that
    times out still consumed nothing and cost nothing — but the *attempt* is
    the thing that shows a provider degrading, and a table that only holds
    successes is a table where an outage looks like a quiet afternoon.

    `call_id` may be absent for work not tied to a call, in which case there is
    nothing to attribute and nothing is written; the caller's own logs still
    have it.
    """
    if not call_id:
        return
    try:
        with db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO localmama.usage"
                    " (call_id, ref, source, provider, model, operation, unit,"
                    "  quantity, ok, latency_ms, error)"
                    " VALUES (%s, %s, 'backend', %s, %s, %s, %s, %s, %s, %s, %s)"
                    " ON CONFLICT (call_id, ref, provider, model, operation, unit)"
                    " DO UPDATE SET quantity = EXCLUDED.quantity, ok = EXCLUDED.ok,"
                    "               latency_ms = EXCLUDED.latency_ms,"
                    "               error = EXCLUDED.error",
                    (call_id, ref, provider, model, operation, unit,
                     max(0.0, float(quantity)), ok, int(latency_ms), error[:300] or None),
                )
            conn.commit()
    except Exception as exc:  # noqa: BLE001 - metering never breaks the pipeline
        logger.warning("could not record a %s call for %s: %s",
                       provider, call_id[:8], exc)


def get_lead(call_id: str) -> dict | None:
    with db.cursor() as cur:
        cur.execute(
            "SELECT call_id, agent_id, caller_phone, dialled, language, raw, name, service,"
            " service_said, city, status, confirmed, confidence, needs_review,"
            " review_reason, vendors, asked_vendors, service_inferred,"
            " handoff_status, handoff_subject, transcript, started_at, ended_at"
            " FROM localmama.leads WHERE call_id = %s",
            (call_id,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    cols = ["call_id", "agent_id", "caller_phone", "dialled", "language", "raw", "name",
            "service", "service_said", "city", "status", "confirmed", "confidence",
            "needs_review", "review_reason", "vendors", "asked_vendors",
            "service_inferred", "handoff_status", "handoff_subject",
            "transcript", "started_at", "ended_at"]
    return dict(zip(cols, row))


def record_asked_vendor(call_id: str, title: str, phone: str,
                        category: str | None = None) -> None:
    """Remember a business the caller asked about by name. Never raises.

    Appended rather than overwritten — a caller may ask about several — and
    deduplicated on title, because asking twice is one business, not two.

    Creates the lead row if the lookup somehow beats `call.started`. The write
    happens after the response has gone back to the agent, so it costs the
    caller nothing: they are waiting on the number, not on us filing it.
    """
    try:
        with db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO localmama.leads (call_id, agent_id, asked_vendors)"
                    " VALUES (%s, %s, %s)"
                    " ON CONFLICT (call_id) DO UPDATE SET"
                    "   asked_vendors = ("
                    "     SELECT jsonb_agg(DISTINCT v) FROM jsonb_array_elements("
                    "       localmama.leads.asked_vendors || EXCLUDED.asked_vendors"
                    "     ) AS v"
                    "   ),"
                    "   updated_at = now()",
                    (call_id, settings.brain_agent_id,
                     Jsonb([{"title": title, "phone": phone, "category": category}])),
                )
            conn.commit()
        logger.info("call %s asked about %r", call_id[:8], title)
    except Exception as exc:  # noqa: BLE001 - the caller already has the number
        logger.warning("could not record the business %s asked about: %s",
                       call_id[:8], exc)


#: The two handoff channels, and the columns each one owns.
#:
#: They are tracked separately and swept separately on purpose. A lead whose
#: WhatsApp went out but whose webhook 500'd is owed the webhook and nothing
#: else — share one status column and the next sweep messages that customer a
#: second time to fix a problem they never had.
#:
#: WhatsApp is the channel the caller actually hears from and stays until the
#: webhook is proven against a real receiver. Retiring it is then this dict
#: entry, its columns, and `services/whatsapp.py`.
_CHANNELS = {
    "whatsapp": {
        "status": "whatsapp_status", "attempts": "whatsapp_attempts",
        "error": "whatsapp_error", "at": "whatsapp_at",
        "detail": "whatsapp_message_id",
    },
    "webhook": {
        "status": "handoff_status", "attempts": "handoff_attempts",
        "error": "handoff_error", "at": "handoff_at",
        "detail": "handoff_response",
    },
}


def _cols(channel: str) -> dict:
    """Column names for a channel. Raises on an unknown one rather than
    interpolating it — these go into SQL, so the whitelist is the guard."""
    try:
        return _CHANNELS[channel]
    except KeyError:
        raise ValueError(f"unknown handoff channel {channel!r}") from None


def mark_handoff(call_id: str, channel: str, ok: bool, error: str = "",
                 detail: object = None, terminal: bool = False) -> None:
    """Record the outcome of one channel's attempt. Never raises.

    `sent` is terminal. `pending` means it is still owed and will be retried;
    `skipped` means there was nothing to deliver to, or nothing to deliver
    with — neither is a failure and neither improves by being retried forever.

    `terminal` is for a *sender* that knows its failure will not improve, and
    exists because the alternative was matching on error text. A receiver that
    rejects the body with 400 is refusing this lead permanently — a rotated
    secret, a schema it does not accept — and without a way to say so it was
    recorded as `pending`, re-POSTed every five minutes for 25 attempts, and
    then dropped past `OUTBOX_MAX_ATTEMPTS` with nothing but repeated warnings
    to show for it.

    `detail` is whatever that channel has worth keeping: a provider message id
    for WhatsApp, the receiver's HTTP status for the webhook. Coerced by the
    caller — these are two differently typed columns behind one parameter.
    """
    c = _cols(channel)
    status = (
        "sent" if ok
        else ("skipped" if terminal or error in _NOT_WORTH_RETRYING else "pending")
    )
    try:
        with db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE localmama.leads SET {c['status']} = %s,"
                    f" {c['attempts']} = {c['attempts']} + 1,"
                    f" {c['error']} = %s, {c['detail']} = %s,"
                    f" {c['at']} = now(), updated_at = now()"
                    " WHERE call_id = %s",
                    (status, error or None, detail if detail not in ("", None) else None, call_id),
                )
            conn.commit()
    except Exception as exc:  # noqa: BLE001 - the lead is already stored
        logger.warning("could not record %s status for %s: %s",
                       channel, call_id[:8], exc)


def claim_owed_handoffs(channel: str, limit: int = 50,
                        stale_after_minutes: int = 10,
                        agents: list[str] | None = None) -> list[dict]:
    """Take ownership of the handoffs owed on one channel, and return them.

    The sweep runs unattended, and a second worker sweeping at the same moment
    would deliver one customer's lead twice. So rows are claimed rather than
    merely read: one `UPDATE ... RETURNING` moves them from `pending` to
    `sending`, and Postgres decides who wins.

    Only *processed* leads are owed anything. `pending` is also the column
    default, so a lead the pipeline never reached looks identical to one whose
    delivery failed — and the sweep took a call that captured nothing at all
    and handed it on, bypassing every guard in the pipeline because the
    pipeline had not run.

    A claim that is never resolved — the process died mid-delivery — would
    strand the lead in `sending` forever, so anything stuck there longer than
    `stale_after_minutes` is treated as owed again. That risks a duplicate in
    the narrow window where a delivery succeeded but the result was never
    recorded; a duplicate is recoverable, a lead silently never sent is not.

    Claimed **per channel**, so a lead whose WhatsApp landed and whose webhook
    failed is claimed for the webhook alone. Sharing one claim across both
    would re-send the message that already worked.

    Returns the whole lead, not the four fields a template needed. The webhook
    body carries everything the pipeline produced, so the sweep must be able to
    build the same payload the live path does rather than a thinner one — a
    lead delivered late should not also be a lead delivered poorer.
    """
    c = _cols(channel)
    try:
        with db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE localmama.leads SET {c['status']} = 'sending',"
                    f" {c['at']} = now()"
                    " WHERE call_id IN ("
                    "   SELECT call_id FROM localmama.leads"
                    "   WHERE (%s::text[] IS NULL OR agent_id = ANY(%s::text[]))"
                    "     AND caller_phone IS NOT NULL"
                    "     AND processed_at IS NOT NULL AND service IS NOT NULL"
                    f"     AND {c['attempts']} < %s"
                    f"     AND ({c['status']} = 'pending'"
                    f"          OR ({c['status']} = 'sending'"
                    f"              AND {c['at']} < now() - make_interval(mins => %s)))"
                    "   ORDER BY created_at"
                    "   FOR UPDATE SKIP LOCKED"
                    "   LIMIT %s)"
                    " RETURNING call_id, agent_id, caller_phone, dialled, language, name,"
                    "           service, service_said, service_inferred, city,"
                    "           status, confirmed, confidence, needs_review,"
                    "           review_reason, vendors, handoff_subject,"
                    f"           started_at, ended_at, {c['attempts']}",
                    (agents, agents, settings.outbox_max_attempts,
                     stale_after_minutes, limit),
                )
                cols = ["call_id", "agent_id", "caller_phone", "dialled", "language", "name",
                        "service", "service_said", "service_inferred", "city",
                        "status", "confirmed", "confidence", "needs_review",
                        "review_reason", "vendors", "handoff_subject",
                        "started_at", "ended_at", "attempts"]
                claimed = [dict(zip(cols, r)) for r in cur.fetchall()]
            conn.commit()
        if claimed:
            logger.info("outbox: claimed %d lead(s) owed a %s", len(claimed), channel)
        return claimed
    except Exception as exc:  # noqa: BLE001 - a sweep must never kill the worker
        logger.warning("could not claim %s from the outbox: %s", channel, exc)
        return []


def unprocessed_leads(limit: int = 50) -> list[str]:
    """Calls that ended but were never turned into a lead.

    The pipeline runs in a background task after the API has answered, so
    anything that kills the process in between — a deploy, a restart, an
    exception nobody caught — leaves a call with every captured value stored
    and nothing done with it. It has no vendors, no confidence, no message, and
    since `processed_at` is null the outbox will not touch it either.

    Those are the leads that go missing without anybody noticing, so the sweep
    re-runs the pipeline on them rather than trying to send from a half-built
    row. Processing is idempotent, so a lead that was mid-flight when we looked
    simply gets done twice.
    """
    try:
        with db.cursor() as cur:
            cur.execute(
                "SELECT call_id FROM localmama.leads"
                " WHERE ended_at IS NOT NULL AND processed_at IS NULL"
                " ORDER BY created_at LIMIT %s",
                (limit,),
            )
            return [r[0] for r in cur.fetchall()]
    except Exception as exc:  # noqa: BLE001 - a sweep must never kill the worker
        logger.warning("could not look for unprocessed leads: %s", exc)
        return []


def active_webhook(agent_id: str | None = None) -> dict | None:
    """The customer's configured delivery endpoint, or None. Never raises.

    Read per delivery rather than cached, because the point of moving this out
    of the environment is that a customer can change it without waiting for a
    deploy — and a cache measured in minutes reintroduces exactly the delay we
    removed. It is one indexed row on a connection the caller already holds.

    A database that is unreachable returns None, which falls back to the
    environment rather than failing the handoff. The lead is already stored;
    losing the endpoint for one pass costs a retry, and the sweep is the retry.

    `db.available()` first, and not merely as an optimisation: without a DSN
    the pool still tries to open, and this is called on every delivery. It cost
    90 seconds on a test suite that runs in two.
    """
    if not db.available():
        return None
    try:
        with db.cursor() as cur:
            cur.execute(
                "SELECT s.id, s.url, s.secret"
                "  FROM localmama.webhook_subscriptions s"
                "  JOIN localmama.tenants t ON t.agent_id = s.agent_id"
                " WHERE s.agent_id = %s AND s.active AND t.active"
                " LIMIT 1",
                (agent_id or settings.brain_agent_id,),
            )
            row = cur.fetchone()
        return {"id": row[0], "url": row[1], "secret": row[2]} if row else None
    except Exception as exc:  # noqa: BLE001 - fall back to the environment
        logger.warning("could not read the webhook subscription: %s", exc)
        return None


def pending_handoffs(channel: str, limit: int = 50) -> list[dict]:
    """Leads still owed a delivery on one channel, oldest first. Read-only."""
    c = _cols(channel)
    try:
        with db.cursor() as cur:
            cur.execute(
                "SELECT call_id, agent_id, caller_phone, name, service, city,"
                f" {c['status']}, {c['attempts']}, {c['error']}"
                " FROM localmama.leads"
                f" WHERE {c['status']} IN ('pending', 'sending')"
                f"   AND caller_phone IS NOT NULL AND {c['attempts']} < %s"
                " ORDER BY created_at LIMIT %s",
                (settings.outbox_max_attempts, limit),
            )
            cols = ["call_id", "agent_id", "caller_phone", "name", "service",
                    "city", "status", "attempts", "error"]
            return [dict(zip(cols, r)) for r in cur.fetchall()]
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not read the outbox: %s", exc)
        return []


def expire_transcripts() -> int:
    """Clear transcripts past the retention window. Returns rows redacted.

    A transcript is everything the caller said, which is personal data under
    the DPDP Act. It is needed for the confidence audit and for diagnosing a
    bad lead, and after that it is a liability — so it is cleared while the
    lead itself, which is the business record, is kept.

    **Both copies.** The transcript is stored twice: assembled onto the lead
    row, and again inside the `call.ended` payload in the raw event log. For a
    long time only the first was swept, so this function reported success, the
    lead row genuinely lost its transcript, and a complete copy of every word
    every caller had said stayed in `call_events` forever. A deletion that
    leaves the data in the next table over is worse than no deletion, because
    it stops anyone from looking again.

    The event rows are redacted, not deleted: `event_id` is the deduplication
    key, and an event whose row has gone would read as new if it were ever
    replayed. Nothing reads these payloads — they exist for deduplication and
    for forensics on a lead that came out wrong — so removing one key from them
    costs nothing that is used.

    Both statements run in one transaction. Half a redaction is the state this
    whole function exists to get out of.
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
                # Keyed on `received_at`, which is on this table, rather than
                # joining the lead's `ended_at`. Simpler, indexable — and it
                # also catches the events of a call that never closed, which an
                # `ended_at` join would leave behind forever precisely because
                # that column is null.
                cur.execute(
                    "UPDATE localmama.call_events"
                    " SET payload = jsonb_set(payload, '{transcript}', '[]'::jsonb)"
                    " WHERE jsonb_exists(payload, 'transcript')"
                    "   AND payload->'transcript' <> '[]'::jsonb"
                    "   AND received_at < now() - make_interval(days => %s)",
                    (days,),
                )
                redacted = cur.rowcount
            conn.commit()
        if cleared or redacted:
            logger.info(
                "cleared %d transcript(s) and redacted %d event payload(s) "
                "past %d-day retention", cleared, redacted, days,
            )
        return cleared + redacted
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

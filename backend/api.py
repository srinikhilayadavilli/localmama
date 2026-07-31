"""The agent-facing HTTP surface.

    POST /v1/events    capture events from a call        -> 202
    GET  /v1/vendors   look a business up, during a call -> 200
    GET  /healthz      readiness

Two rules shape everything here.

**The agent must never be made to wait.** `/v1/events` stores and acknowledges;
processing a finished call — normalising, matching, sending WhatsApp — happens
after the response, in a background task. The one exception is `/v1/vendors`,
which a caller is genuinely waiting on, and which is therefore the only
endpoint that touches the catalogue synchronously.

**The agent must never be made to fail.** An event this service cannot store is
reported as rejected so the agent stops retrying it; anything else is a 5xx the
agent will retry. What must not happen is a 200 for an event that was dropped —
the agent would believe the lead had landed.
"""

from __future__ import annotations

import hmac
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from pydantic import ValidationError

from contract import (
    CallCaptured,
    CallEnded,
    CallStarted,
    EventAck,
    EventBatch,
    Language,
    MatchKind,
    VendorReply,
)

from . import db, pipeline, store
from .config import missing_required, settings
from .logger import get_logger, setup_logging
from .services import brain, translate

logger = get_logger("localmama.api")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    setup_logging()
    for problem in missing_required():
        logger.error("STARTUP: %s", problem)
    yield
    db.close()


app = FastAPI(title="Local Mama lead backend", version="1.0", lifespan=lifespan)


async def authorise(authorization: str = Header(default="")) -> None:
    """One client, one shared token, compared in constant time.

    An unset `AGENT_TOKEN` refuses everything rather than allowing everything:
    a misconfigured deployment that accepts anonymous leads is worse than one
    that accepts none, because nothing about it looks wrong.
    """
    if not settings.agent_token:
        raise HTTPException(503, "AGENT_TOKEN is not configured")
    presented = authorization.removeprefix("Bearer ").strip()
    if not hmac.compare_digest(presented, settings.agent_token):
        raise HTTPException(401, "bad token")


@app.get("/healthz")
async def healthz() -> dict:
    """Readiness. Render polls this; a human reads it when something is wrong."""
    problems = missing_required()
    ok = not problems
    if ok and db.available():
        try:
            with db.cursor() as cur:
                cur.execute("SELECT 1")
        except Exception as exc:  # noqa: BLE001
            ok, problems = False, [f"database unreachable: {exc}"]
    return {
        "ok": ok,
        "problems": problems,
        "whatsapp": settings.whatsapp_available,
        "translation": translate.available(),
    }


@app.post("/v1/events", status_code=202, dependencies=[Depends(authorise)])
async def post_events(batch: EventBatch, background: BackgroundTasks) -> EventAck:
    """Accept capture events. Acknowledges receipt, not processing."""
    if not db.available():
        raise HTTPException(503, "no database configured")

    accepted, duplicates = store.record_events(batch.events)
    rejected: dict[str, str] = {}

    # Applied only for events this call actually stored. A retry must not
    # re-apply a capture or re-trigger a pipeline run.
    fresh = [e for e in batch.events if e.event_id in set(accepted)]
    for event in fresh:
        try:
            _apply(event, background)
        except Exception as exc:  # noqa: BLE001 - one bad event, not the batch
            logger.exception("could not apply %s", event.event_id)
            rejected[event.event_id] = str(exc)

    logger.info(
        "events: %d accepted, %d duplicate, %d rejected",
        len(accepted), len(duplicates), len(rejected),
    )
    return EventAck(
        accepted=[i for i in accepted if i not in rejected],
        duplicates=duplicates,
        rejected=rejected,
    )


def _apply(event, background: BackgroundTasks) -> None:  # noqa: ANN001
    """Fold one event onto its lead row."""
    if isinstance(event, CallStarted):
        store.upsert_call(
            event.call_id,
            caller_phone=event.caller_phone,
            dialled=event.dialled,
            started_at=event.started_at,
            status="in_progress",
        )
        return

    if isinstance(event, CallCaptured):
        # Language is the one field the agent resolves, because it has to know
        # which language to speak next. It lands on its own column; everything
        # else is raw text merged into `raw`.
        if event.field.value == "language":
            store.upsert_call(event.call_id, language=event.value)
        if event.interpreted:
            # What the agent understood a description to mean, confirmed with
            # the caller during the read-back. Not a captured value, so it does
            # not go in `raw`.
            store.upsert_call(event.call_id, service_inferred=event.interpreted)
        store.merge_raw(event.call_id, event.field, event.value)
        return

    if isinstance(event, CallEnded):
        store.upsert_call(
            event.call_id,
            status=event.status.value,
            confirmed=event.confirmed,
            transcript=_json([t.model_dump(mode="json") for t in event.transcript]),
            turn_gaps=_json(event.turn_gaps),
            ended_at=event.ended_at,
        )
        # The call is over and the caller has hung up. Everything expensive
        # happens here, after the agent has its 202.
        background.add_task(_process, event.call_id)
        return


async def _process(call_id: str) -> None:
    try:
        await pipeline.process(call_id)
    except Exception:  # noqa: BLE001 - a background task's exception goes nowhere
        logger.exception("pipeline failed for %s; the lead is stored and can be "
                         "replayed", call_id[:8])


@app.get("/v1/vendors", dependencies=[Depends(authorise)])
async def get_vendor(
    background: BackgroundTasks,
    name: str,
    city: str | None = None,
    language: Language = Language.ENGLISH,
    call_id: str | None = None,
) -> VendorReply:
    """The only call a caller waits on. Answers with a line to speak.

    The agent gets prose, not a record: digit grouping, and the decision that a
    match is too weak to read out, are directory policy and belong on the side
    that owns the directory.

    `call_id` ties the question to the call. Without it this endpoint answered
    and forgot, so a business the caller explicitly asked about never reached
    their WhatsApp — the one thing they actually wanted was the one thing left
    out. Recorded in the background, after the reply has gone: they are waiting
    on the number, not on us filing it.
    """
    if not brain.available():
        return VendorReply(
            match=MatchKind.NONE,
            say="I can't look that up right now.",
            instruction="Offer to have the team follow up.",
        )

    # The catalogue is English and matching is literal, so a name still in the
    # caller's script reaches nothing.
    english = await translate.english_name(name)

    if brain.looks_like_a_category(english):
        return VendorReply(
            match=MatchKind.CATEGORY,
            say=f"{english} is a type of business rather than one I can call up.",
            instruction=(
                "Ask which business they mean, by name. Do NOT list any businesses."
            ),
        )

    matches = brain.find_business(english, limit=4)
    if not matches:
        return VendorReply(
            match=MatchKind.NONE,
            say=f"I don't have {english} listed.",
            instruction="Tell the caller so, and do NOT guess a number.",
        )

    exact = [m for m in matches if m.title.lower() == english.strip().lower()]
    if len(matches) > 1 and not exact:
        return VendorReply(
            match=MatchKind.AMBIGUOUS,
            say=f"I have more than one business close to {english}.",
            instruction="Ask for the full name. Do NOT read out the list.",
        )

    hit = exact[0] if exact else matches[0]

    # Reached by spelling, not by a literal match. Naming it back is the whole
    # safeguard, so nothing structured is returned — even if the model ignores
    # `say`, it has no number to leak.
    if hit.approximate:
        return VendorReply(
            match=MatchKind.APPROXIMATE,
            say=f"I have a {hit.title} — is that the one you mean?",
            instruction="Confirm the name BEFORE reading any number.",
        )

    if not hit.phone:
        return VendorReply(
            match=MatchKind.NONE,
            say=f"I have {hit.title} listed, but no phone number for them.",
            instruction="Offer to have the team follow up.",
        )

    # Only an exact hit is recorded. An approximate one is still a question —
    # the agent has to name it back and ask before any number is read — and a
    # category or an ambiguous match names no business at all.
    if call_id:
        background.add_task(
            store.record_asked_vendor, call_id, hit.title, hit.phone, hit.category
        )

    return VendorReply(
        match=MatchKind.EXACT,
        say=f"{hit.title} — {brain.spoken_phone(hit.phone)}",
        instruction="Read the number in digit groups and offer to repeat it.",
        title=hit.title,
        phone=hit.phone,
    )


@app.exception_handler(ValidationError)
async def _validation(_request, exc: ValidationError):
    """A malformed event is the agent's bug, and it will not improve on retry."""
    from fastapi.responses import JSONResponse

    logger.warning("rejected malformed payload: %s", exc)
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


def _json(value):
    from psycopg.types.json import Jsonb

    return Jsonb(value)

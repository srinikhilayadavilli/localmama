"""FastAPI app: browser test UI, WebSocket conversation loop, and debug endpoints.

This is the zero-dependency path to a working demo. The browser supplies audio
via the Web Speech API (recognition + synthesis) and sends *text* over the
WebSocket, so the full multilingual workflow can be exercised without any STT
or TTS vendor keys. The LiveKit path in `agent.py` uses the same
`ConversationManager` — only the transport differs.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import session_store
from .config import ROOT_DIR, settings
from .languages import ENDONYM, LOCALE, Language
from .logger import get_logger, setup_logging
from .models import ConversationState
from .persistence import (
    lead_summary,
    list_leads,
    list_transcript_ids,
    load_lead,
    load_session,
)
from .providers.registry import describe as describe_providers
from .security import MIN_TURN_INTERVAL, TurnLimiter, is_safe_session_id
from .services.conversation_manager import ConversationManager, abandon
from .state_machine import FLOW, MANDATORY_FIELDS, missing_fields

logger = get_logger("localmama.api")
FRONTEND_DIR = ROOT_DIR / "frontend"

app = FastAPI(title="Local Mama", version="0.1.0")


@app.on_event("startup")
async def _startup() -> None:
    setup_logging()
    settings.ensure_dirs()
    logger.info("Local Mama API ready  http://%s:%s", settings.host, settings.port)
    logger.info("providers: %s", describe_providers())
    if not settings.llm_available:
        logger.info(
            "LLM extraction disabled (no ANTHROPIC_API_KEY) — using rules only. "
            "This is fine for the demo flow."
        )


# --------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------- #

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/admin")
async def admin() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "admin.html")


@app.get("/health")
async def health() -> JSONResponse:
    """Liveness probe for the host platform (Railway hits this on every deploy).

    Deliberately touches nothing — no disk, no vendor API. A health check that
    can fail for a reason unrelated to "is this process serving requests" turns
    a working deploy into a rollback loop.
    """
    return JSONResponse({"status": "ok"})


# --------------------------------------------------------------------- #
# Conversation over WebSocket
# --------------------------------------------------------------------- #


def _payload(result) -> dict:
    """Serialise a TurnResult for the browser UI."""
    session = result.session
    return {
        "type": "agent",
        "reply": result.reply,
        "state": result.state.value,
        "language": result.language.value,
        "locale": LOCALE[result.language],
        "is_final": result.is_final,
        "session_id": session.session_id,
        "captured": {
            "selected_language": session.selected_language.value
            if session.selected_language
            else None,
            "user_name": session.user_name,
            "requested_service": session.requested_service,
            "city_or_area": session.city_or_area,
        },
        "missing": missing_fields(session),
        "lead": result.lead.model_dump(mode="json") if result.lead else None,
    }


@app.websocket("/ws/chat")
async def chat(websocket: WebSocket) -> None:
    """One WebSocket == one call.

    Client -> server:  {"type": "start"} | {"type": "user_text", "text": "..."}
    Server -> client:  see `_payload`.
    """
    await websocket.accept()
    manager: ConversationManager | None = None
    # Transport-level flood control: rejects frames arriving faster than a
    # person could speak, independent of the per-call turn cap in the manager.
    flood = TurnLimiter(max_turns=10_000, min_interval=MIN_TURN_INTERVAL)
    try:
        while True:
            message = await websocket.receive_json()
            if not isinstance(message, dict):
                await websocket.send_json({"type": "error", "error": "bad frame"})
                continue
            kind = message.get("type")

            if kind == "start":
                manager = session_store.create()
                result = manager.start()
                await websocket.send_json(_payload(result))

            elif kind == "user_text":
                if manager is None:
                    manager = session_store.create()
                    await websocket.send_json(_payload(manager.start()))
                allowed, reason = flood.check()
                if not allowed:
                    logger.warning("ws flood control: %s", reason)
                    await websocket.send_json({"type": "error", "error": reason})
                    continue
                text = (message.get("text") or "").strip()
                result = await manager.handle(text)
                await websocket.send_json(_payload(result))

            else:
                await websocket.send_json({"type": "error", "error": f"unknown type {kind!r}"})

    except WebSocketDisconnect:
        if manager and manager.session.state is not ConversationState.COMPLETED:
            abandon(manager.session)
            session_store.drop(manager.session.session_id)
    except Exception as exc:  # noqa: BLE001 - a bad frame must not kill the server
        logger.exception("websocket error: %s", exc)
        try:
            await websocket.close(code=1011)
        except RuntimeError:
            pass


# --------------------------------------------------------------------- #
# Debug / admin API
# --------------------------------------------------------------------- #


@app.get("/api/config")
async def config() -> JSONResponse:
    return JSONResponse(
        {
            "languages": [
                {"value": lang.value, "label": lang.value.capitalize(),
                 "endonym": ENDONYM[lang], "locale": LOCALE[lang]}
                for lang in Language
            ],
            "states": [state.value for state in FLOW],
            "mandatory_fields": list(MANDATORY_FIELDS),
            "providers": describe_providers(),
            "llm_extraction": settings.llm_available,
            "llm_model": settings.llm_model if settings.llm_available else None,
            "livekit_configured": settings.livekit_configured,
        }
    )


@app.get("/api/leads")
async def leads() -> JSONResponse:
    return JSONResponse([lead_summary(lead) for lead in list_leads()])


@app.get("/api/leads/{session_id}")
async def lead_detail(session_id: str) -> JSONResponse:
    if not is_safe_session_id(session_id):
        raise HTTPException(status_code=400, detail="invalid session id")
    lead = load_lead(session_id)
    if lead is None:
        raise HTTPException(status_code=404, detail="lead not found")
    return JSONResponse(lead.model_dump(mode="json"))


@app.get("/api/transcripts")
async def transcripts() -> JSONResponse:
    return JSONResponse(list_transcript_ids())


@app.get("/api/transcripts/{session_id}")
async def transcript_detail(session_id: str) -> JSONResponse:
    """Session replay: return the saved turn-by-turn transcript."""
    if not is_safe_session_id(session_id):
        raise HTTPException(status_code=400, detail="invalid session id")
    session = load_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="transcript not found")
    return JSONResponse(
        {
            "session_id": session.session_id,
            "language": session.selected_language.value
            if session.selected_language
            else None,
            "status": session.conversation_status.value,
            "turns": [turn.model_dump(mode="json") for turn in session.transcript],
        }
    )


@app.post("/api/livekit/token")
async def livekit_token(body: dict) -> JSONResponse:
    """Mint a room token for the browser to join the LiveKit voice path."""
    if not settings.livekit_configured:
        raise HTTPException(
            status_code=503,
            detail="LiveKit is not configured; set LIVEKIT_URL, LIVEKIT_API_KEY, "
            "LIVEKIT_API_SECRET in .env",
        )
    try:
        from livekit import api
    except ImportError as exc:
        raise HTTPException(
            status_code=503, detail="livekit-api not installed"
        ) from exc

    room = (body or {}).get("room") or "local-mama"
    identity = (body or {}).get("identity") or "caller"
    token = (
        api.AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(api.VideoGrants(room_join=True, room=room))
        .to_jwt()
    )

    # A *named* worker only takes jobs dispatched to it by name — that is what
    # keeps one deployment from joining another's calls. The cost is that a
    # token alone is not enough: without this dispatch the caller joins an empty
    # room and hears silence. An unnamed worker auto-joins and needs none of it.
    dispatched = False
    agent_name = settings.livekit_agent_name.strip()
    if agent_name:
        try:
            lk = api.LiveKitAPI(
                settings.livekit_url, settings.livekit_api_key, settings.livekit_api_secret
            )
            try:
                await lk.agent_dispatch.create_dispatch(
                    api.CreateAgentDispatchRequest(agent_name=agent_name, room=room)
                )
                dispatched = True
            finally:
                await lk.aclose()
        except Exception as exc:  # noqa: BLE001 - a token is still useful
            # Most often: no worker registered under that name. Report it rather
            # than handing back a token that leads to a silent room.
            logger.warning("agent dispatch for %r failed: %s", agent_name, exc)

    return JSONResponse(
        {
            "token": token,
            "url": settings.livekit_url,
            "room": room,
            "agent_name": agent_name or None,
            "dispatched": dispatched,
        }
    )


def run() -> None:
    import uvicorn

    setup_logging()
    uvicorn.run(
        "backend.app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run()

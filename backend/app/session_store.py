"""In-memory registry of live sessions.

Process-local by design: a session only needs to outlive one call. Completed
sessions are persisted to disk by `persistence.py`, so nothing important is
lost on restart. Swap for Redis when you run more than one worker.
"""

from __future__ import annotations

import uuid

from .models import SessionData
from .services.conversation_manager import ConversationManager

_SESSIONS: dict[str, ConversationManager] = {}


def create(session_id: str | None = None) -> ConversationManager:
    session_id = session_id or str(uuid.uuid4())
    manager = ConversationManager(SessionData(session_id=session_id))
    _SESSIONS[session_id] = manager
    return manager


def get(session_id: str) -> ConversationManager | None:
    return _SESSIONS.get(session_id)


def get_or_create(session_id: str | None) -> ConversationManager:
    if session_id and (existing := get(session_id)):
        return existing
    return create(session_id)


def drop(session_id: str) -> None:
    _SESSIONS.pop(session_id, None)


def active_ids() -> list[str]:
    return list(_SESSIONS)

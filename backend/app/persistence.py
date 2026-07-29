"""Local JSON persistence for leads and transcripts.

One file per session keeps the MVP inspectable with `cat` and trivially
portable. Swap this module for SQLite or Postgres later — everything upstream
goes through `save_lead` / `list_leads`, so the call sites do not change.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import settings
from .logger import get_logger
from .models import Lead, SessionData

logger = get_logger("localmama.persistence")


def _write_json(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file then replace, so a crash mid-write cannot leave a
    # truncated JSON file behind.
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


def save_lead(lead: Lead) -> Path:
    settings.ensure_dirs()
    path = settings.leads_dir / f"{lead.session_id}.json"
    _write_json(path, lead.model_dump_json(indent=2))
    logger.info("saved lead -> %s", path)
    return path


def save_transcript(session: SessionData) -> Path:
    settings.ensure_dirs()
    path = settings.transcripts_dir / f"{session.session_id}.json"
    _write_json(path, session.model_dump_json(indent=2))
    return path


def load_lead(session_id: str) -> Lead | None:
    path = settings.leads_dir / f"{session_id}.json"
    if not path.exists():
        return None
    return Lead.model_validate_json(path.read_text(encoding="utf-8"))


def list_leads() -> list[Lead]:
    """All saved leads, newest first. Powers the debug/admin page."""
    settings.ensure_dirs()
    leads: list[Lead] = []
    for path in settings.leads_dir.glob("*.json"):
        try:
            leads.append(Lead.model_validate_json(path.read_text(encoding="utf-8")))
        except (ValueError, OSError) as exc:
            logger.warning("skipping unreadable lead %s: %s", path.name, exc)
    return sorted(leads, key=lambda lead: lead.started_at, reverse=True)


def load_session(session_id: str) -> SessionData | None:
    """Load a saved transcript back into a `SessionData` for replay."""
    path = settings.transcripts_dir / f"{session_id}.json"
    if not path.exists():
        return None
    return SessionData.model_validate_json(path.read_text(encoding="utf-8"))


def list_transcript_ids() -> list[str]:
    settings.ensure_dirs()
    return sorted(p.stem for p in settings.transcripts_dir.glob("*.json"))


def lead_summary(lead: Lead) -> dict:
    """Compact dict for list views."""
    return json.loads(
        json.dumps(
            {
                "session_id": lead.session_id,
                "selected_language": lead.selected_language.value
                if lead.selected_language
                else None,
                "user_name": lead.user_name,
                "requested_service": lead.requested_service,
                "city_or_area": lead.city_or_area,
                "conversation_status": lead.conversation_status.value,
                "started_at": lead.started_at.isoformat(),
                "completed_at": lead.completed_at.isoformat() if lead.completed_at else None,
                "turns": len(lead.transcript),
            }
        )
    )

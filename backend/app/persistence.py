"""Local JSON persistence for leads and transcripts.

One file per session keeps the MVP inspectable with `cat` and trivially
portable. Swap this module for SQLite or Postgres later — everything upstream
goes through `save_lead`, so the call sites do not change.
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







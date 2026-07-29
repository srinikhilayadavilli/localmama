"""Returning-caller memory.

Local services are a repeat business: someone who needed a plumber in March
needs an electrician in June. Recognising them is the difference between a
five-turn call and a two-turn one — on a phone line, that is the single
biggest UX win available.

The design point worth noticing: this needs **no changes to the state
machine**. `next_state()` already skips any state whose field is filled, so
"remembering" a caller is just prefilling `SessionData` before the call starts.
Memory is data, not control flow.

What is deliberately remembered, and what is not:

  * **preferred language** — stable, unambiguous, saves a whole turn. Remembered.
  * **name** — stable, and being greeted by name is the point. Remembered.
  * **service** — changes every call; that is *why* they are calling. Never
    prefilled, only recorded for analytics.
  * **area** — people move, and they may want service at a different address.
    Recorded but never assumed; treat it as a suggestion, not a fact.

Privacy. A phone number is personal data under India's DPDP Act 2023, so the
raw identifier is never stored: profiles are keyed by a salted SHA-256 hash,
which is enough to recognise a returning caller but not to enumerate numbers
from the files. `forget()` implements the erasure right, and profiles past
`PROFILE_RETENTION_DAYS` are dropped on read. The caller's *name* is still
personal data and is stored in clear, because greeting them is the feature —
so treat `data/profiles/` with the same care as `data/leads/`.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from .config import settings
from .languages import Language
from .logger import get_logger
from .models import SessionData, utcnow

logger = get_logger("localmama.profiles")

#: Profiles older than this are treated as absent and deleted on access.
PROFILE_RETENTION_DAYS = 180


class CallerProfile(BaseModel):
    """What we remember about one returning caller."""

    caller_key: str                       # salted hash — never the raw number
    preferred_language: Language | None = None
    name: str | None = None
    last_service: str | None = None
    last_area: str | None = None
    call_count: int = 0
    first_seen: datetime = Field(default_factory=utcnow)
    last_seen: datetime = Field(default_factory=utcnow)

    @property
    def is_returning(self) -> bool:
        return self.call_count > 0


# Settings is a frozen dataclass, so these thin accessors exist to give tests a
# seam: they can be patched, whereas the frozen fields cannot.
def _profiles_dir() -> Path:
    return settings.data_dir / "profiles"


def _salt() -> str:
    return settings.profile_salt


def _enabled() -> bool:
    return settings.caller_memory_enabled


def caller_key(caller_id: str) -> str:
    """Salted hash of a caller identifier (phone number, SIP URI, …).

    The salt keeps the hashes from being reversible with a rainbow table of
    Indian phone numbers, which is otherwise a very small search space.
    """
    salted = f"{_salt()}:{caller_id.strip()}"
    return hashlib.sha256(salted.encode("utf-8")).hexdigest()[:32]


def _path_for(key: str) -> Path:
    return _profiles_dir() / f"{key}.json"


def load(caller_id: str | None) -> CallerProfile | None:
    """Return the caller's profile, or None if unknown, stale, or unreadable."""
    if not caller_id or not _enabled():
        return None

    key = caller_key(caller_id)
    path = _path_for(key)
    if not path.exists():
        return None

    try:
        profile = CallerProfile.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        logger.warning("unreadable profile %s: %s", key[:8], exc)
        return None

    cutoff = datetime.now(timezone.utc) - timedelta(days=PROFILE_RETENTION_DAYS)
    if profile.last_seen < cutoff:
        logger.info("profile %s expired (retention); deleting", key[:8])
        path.unlink(missing_ok=True)
        return None

    return profile


def remember(caller_id: str | None, session: SessionData) -> CallerProfile | None:
    """Record what this call taught us about the caller.

    Called once a call completes. Fields are only overwritten when the call
    actually captured them, so a partial call never erases a good profile.
    """
    if not caller_id or not _enabled():
        return None

    key = caller_key(caller_id)
    profile = load(caller_id) or CallerProfile(caller_key=key)

    if session.selected_language:
        profile.preferred_language = session.selected_language
    if session.user_name:
        profile.name = session.user_name
    if session.requested_service:
        profile.last_service = session.requested_service
    if session.city_or_area:
        profile.last_area = session.city_or_area
    profile.call_count += 1
    profile.last_seen = utcnow()

    path = _path_for(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(path)
    logger.info(
        "profile %s updated (call #%d, lang=%s)",
        key[:8],
        profile.call_count,
        profile.preferred_language.value if profile.preferred_language else "?",
    )
    return profile


def apply_to_session(profile: CallerProfile | None, session: SessionData) -> list[str]:
    """Prefill a new session from a profile. Returns the fields prefilled.

    Only language and name are prefilled. The service is what the caller is
    ringing about, and the area may have changed — assuming either would put
    words in their mouth.
    """
    if profile is None or not profile.is_returning:
        return []

    prefilled: list[str] = []
    if profile.preferred_language and session.selected_language is None:
        session.selected_language = profile.preferred_language
        prefilled.append("selected_language")
    if profile.name and session.user_name is None:
        session.user_name = profile.name
        prefilled.append("user_name")
    return prefilled


def forget(caller_id: str) -> bool:
    """Erase a caller's profile. Implements the DPDP right to erasure."""
    key = caller_key(caller_id)
    path = _path_for(key)
    if path.exists():
        path.unlink()
        logger.info("profile %s erased on request", key[:8])
        return True
    return False


def count() -> int:
    d = _profiles_dir()
    return len(list(d.glob("*.json"))) if d.exists() else 0

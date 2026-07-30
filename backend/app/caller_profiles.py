"""What a call learned about the caller. Recorded, never replayed.

This began as returning-caller memory: prefill the name and language from the
last call and the state machine skips those questions, because `next_state()`
ignores any state whose field is already filled. Two turns saved on a phone
line is a real win, and it is not the win it looks like.

**Nothing here is read back into a conversation any more.** A prefilled value
is asserted to the caller as fact — the agent never asks, and reads it back
among the details it collected — so it has to be right, and it cannot be
guaranteed to be. A name captured wrongly once is then repeated on every later
call from that number with no turn in which the caller could correct it, and
anyone sharing a handset or arriving behind a PBX that presents one number is
greeted as somebody else. On the speech-to-speech path it was also load-bearing
in the wrong direction: the prefilled name went into the tools' HELD line,
whose whole purpose is to tell the model to stop asking.

So `remember()` still writes — the profiles are worth having for analytics, and
`forget()` is a legal obligation — and `load()` is left for the CLI and for
erasure. No conversation path calls either.

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

"""The business directory — names, categories and phone numbers.

Separate from `brain.py` on purpose, and the distinction matters:

  brain      utter.knowledge, shared with the other product, semantic search
             over descriptions. Answers "who does something like this?" and
             carries NO phone numbers for these rows.
  directory  localmama.businesses, tenant-owned, 120 rows every one of which
             has a phone. Answers "what is X's number?".

A caller asking "what's Brimmies Cafe's number?" wants an exact record, not the
nearest semantic neighbour — returning a plausible-but-different business with
a real phone number attached would send them to a stranger. So matching here is
deliberately literal: exact name, then prefix, then substring, then the curated
keywords. No embeddings, no similarity score, no "close enough".

Read-only. The directory is loaded and curated elsewhere; a voice agent that
could write to it is a much larger blast radius for no gain.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import settings
from ..logger import get_logger

logger = get_logger("localmama.directory")


@dataclass
class Business:
    name: str
    category: str | None
    phone: str | None

    def spoken_phone(self) -> str:
        """Digits grouped so a TTS engine reads them as a number, not a year.

        "9735927627" said as one token comes out as garbage on every engine we
        have tried; in groups it is dictatable.
        """
        digits = "".join(ch for ch in (self.phone or "") if ch.isdigit())
        if len(digits) == 10:
            return f"{digits[:5]} {digits[5:]}"
        if len(digits) == 12 and digits.startswith("91"):
            return f"+91 {digits[2:7]} {digits[7:]}"
        return self.phone or ""


def available() -> bool:
    return bool(settings.database_url)


def _connect():
    import psycopg

    return psycopg.connect(settings.database_url, connect_timeout=5)


def find(name: str, limit: int = 5) -> list[Business]:
    """Businesses whose name or keywords match, most exact first.

    Never raises: a directory outage costs an answer, not the call.
    """
    query = (name or "").strip()
    if not available() or not query:
        return []
    try:
        with _connect() as conn, conn.cursor() as cur:
            # Ordered by how literal the match is, so an exact name always wins
            # over a business that merely mentions the word in its keywords.
            cur.execute(
                "SELECT name, category, phone,"
                "  CASE"
                "    WHEN lower(name) = lower(%(q)s) THEN 0"
                "    WHEN lower(name) LIKE lower(%(q)s) || '%%' THEN 1"
                "    WHEN lower(name) LIKE '%%' || lower(%(q)s) || '%%' THEN 2"
                "    ELSE 3"
                "  END AS rank"
                " FROM localmama.businesses"
                " WHERE lower(name) LIKE '%%' || lower(%(q)s) || '%%'"
                "    OR EXISTS (SELECT 1 FROM unnest(keywords) k"
                "               WHERE lower(k) LIKE '%%' || lower(%(q)s) || '%%')"
                " ORDER BY rank, name LIMIT %(k)s",
                {"q": query, "k": limit},
            )
            found = [Business(*row[:3]) for row in cur.fetchall()]
        logger.info("directory %r -> %s", query[:40], [b.name for b in found])
        return found
    except Exception as exc:  # noqa: BLE001 - never end a call over a lookup
        logger.warning("directory lookup failed (%s); continuing without it", exc)
        return []


async def find_async(name: str, limit: int = 5) -> list[Business]:
    """`find` off the event loop — psycopg blocks, and this runs mid-call."""
    import asyncio

    return await asyncio.to_thread(find, name, limit)

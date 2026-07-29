"""Input hardening for untrusted caller speech.

Everything a caller says is untrusted input. It reaches regex extraction, gets
persisted to disk, and is rendered in a browser admin page, so it is sanitised
once at the boundary — in `ConversationManager.handle()` — which covers every
transport (WebSocket, LiveKit, CLI) from a single place.

The threats this addresses, and why each limit exists:

* **ReDoS / event-loop starvation.** Extraction runs ~100 regex and substring
  passes synchronously. Several location patterns use a lazy `.+?` and are
  O(n²): 2k chars take 30ms, 8k take 490ms, 100k take minutes. A single large
  utterance therefore blocks the whole async server. `MAX_UTTERANCE_CHARS` is
  the primary defence — a spoken turn is never this long.
* **Unbounded memory.** Transcripts are held in RAM and written to disk, so
  both turn count and per-turn size are capped.
* **Flooding.** A client can send frames as fast as it likes; turns are rate
  limited and capped per session.
* **Stored injection.** Control characters and markup are stripped from values
  before they are persisted or rendered.
* **Path traversal.** Session IDs become filenames, so they are validated
  against a strict UUID pattern before touching the filesystem.

Note on prompt injection: it is handled structurally rather than by filtering.
The LLM never chooses the next state and has no tools, so the worst a malicious
utterance can do is propose field *values*, which are then sanitised here and
validated by the state machine. See README §9.
"""

from __future__ import annotations

import re
import time
import unicodedata

#: A spoken turn. Long enough for a rambling sentence, short enough that the
#: quadratic extraction patterns stay in the sub-millisecond range.
MAX_UTTERANCE_CHARS = 300

#: A captured field (name, service, locality). Generous for Indian place names.
MAX_FIELD_CHARS = 80

#: Hard stop on a single call, so one session cannot grow without bound.
MAX_TURNS_PER_SESSION = 60

#: Minimum gap between accepted turns, in seconds. Enforced at the WebSocket
#: boundary only — the core manager must stay usable from tests, scripted
#: replays, and the CLI, which legitimately drive turns back to back.
MIN_TURN_INTERVAL = 0.15

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)

# Characters that have no place in speech and that make stored values unsafe to
# render or log: C0/C1 controls, zero-width and bidirectional overrides.
_UNSAFE_CHARS = re.compile(
    r"[\x00-\x08\x0b-\x1f\x7f-\x9f​-‏‪-‮⁦-⁩]"
)
_MARKUP = re.compile(r"[<>{}\\`]")


def sanitize_utterance(text: str) -> str:
    """Make a raw caller utterance safe to run extraction over.

    Truncates rather than rejects: a caller who rambles should still be
    understood, not dropped.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = _UNSAFE_CHARS.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > MAX_UTTERANCE_CHARS:
        text = text[:MAX_UTTERANCE_CHARS].rstrip()
    return text


def sanitize_field(value: str | None) -> str | None:
    """Clean a value before it is committed, persisted, or rendered.

    Returns None when nothing usable survives, so the caller re-prompts rather
    than storing junk.
    """
    if not value:
        return None
    value = unicodedata.normalize("NFC", str(value))
    value = _UNSAFE_CHARS.sub(" ", value)
    value = _MARKUP.sub(" ", value)
    value = re.sub(r"\s+", " ", value).strip(" .,;:!?-")
    if len(value) > MAX_FIELD_CHARS:
        value = value[:MAX_FIELD_CHARS].rstrip()
    return value or None


def is_safe_session_id(session_id: str) -> bool:
    """Session IDs become filenames — accept only server-generated UUIDs."""
    return bool(session_id) and bool(_UUID_RE.match(session_id))


class TurnLimiter:
    """Per-session flood control.

    Rejects turns that arrive faster than a human could speak, and stops a
    session that has run implausibly long.
    """

    def __init__(
        self,
        max_turns: int = MAX_TURNS_PER_SESSION,
        min_interval: float = 0.0,
    ) -> None:
        self.max_turns = max_turns
        self.min_interval = min_interval
        self.turns = 0
        self._last = 0.0

    def check(self, now: float | None = None) -> tuple[bool, str]:
        """Return (allowed, reason). `reason` is empty when allowed."""
        now = time.monotonic() if now is None else now
        if self.turns >= self.max_turns:
            return False, "turn limit reached"
        if self._last and (now - self._last) < self.min_interval:
            return False, "sending too fast"
        self.turns += 1
        self._last = now
        return True, ""

"""Validation contract for LLM-generated replies.

Free-form generation is allowed to make Mami sound human. It is not allowed to
change what the call *does*. This module is the gate between the two: every
generated line is checked against the current workflow state, and anything that
fails is discarded in favour of the deterministic template.

That fallback is the whole safety argument. The LLM can only ever make a turn
*nicer*; it can never make one *wrong*, because a wrong turn never reaches the
caller. Worst case — bad generation, timeout, no API key, hostile input — the
caller hears the same templated line the rules-only agent would have spoken.

What the guard enforces:

  * **No promises out of turn.** WhatsApp, "I'll send", "booked", "confirmed"
    are forbidden before CONFIRMATION. This is the one that matters most: an
    agent that says "I'll send the details" before it has the caller's city has
    lied to them.
  * **No invented entities.** A service or city named in the reply must be one
    we actually captured. Stops "I'll find you an electrician in Mumbai" when
    the caller said neither.
  * **No fabricated specifics.** Phone numbers, prices, times, URLs.
  * **Stays on task.** A state that asks a question must still ask it.
  * **Right language.** Generated text must be in the session's script.
  * **Voice-safe.** No markdown, emoji, or lists; bounded length.
"""

from __future__ import annotations

import re

from ..languages import Language
from ..models import ConversationState, SessionData
from .entity_extractor import CITY_SEEDS, SERVICE_CATALOG

#: Spoken replies are short. Anything longer is a sign the model is rambling.
MAX_REPLY_CHARS = 320

_EMOJI = re.compile(
    "[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff←-⇿⬀-⯿]"
)
_MARKDOWN = re.compile(r"(\*\*|__|```|^\s*[-*•]\s|^\s*#{1,6}\s|\[[^\]]*\]\([^)]*\))", re.M)
_URL = re.compile(r"(https?://|www\.)", re.I)
#: 4+ consecutive digits — phone numbers, prices, invented reference codes.
_LONG_NUMBER = re.compile(r"\d{4,}")

#: Commitments Mami may only make once every mandatory field is captured.
#: Romanised and native-script forms, because the reply may be in any language.
_PROMISE_TERMS = (
    "whatsapp", "व्हाट्सएप", "হোয়াটসঅ্যাপ", "వాట్సాప్", "வாட்ஸ்அப்", "ವಾಟ್ಸಾಪ್",
    "i will send", "i'll send", "sending you", "will share", "have booked",
    "is booked", "confirmed your", "your booking", "on the way", "assigned",
)

#: States in which a promise is legitimate.
_PROMISE_ALLOWED = {
    ConversationState.CONFIRMATION,
    ConversationState.CLOSING,
    ConversationState.COMPLETED,
}

#: States whose purpose is to elicit something.
_MUST_ASK = {
    ConversationState.REVIEW,
    ConversationState.LANGUAGE_SELECTION,
    ConversationState.ASK_NAME,
    ConversationState.ASK_SERVICE,
    ConversationState.ASK_LOCATION,
}

#: Minimum share of alphabetic characters that must be in the target script.
#: Indic sessions are lenient — Indian speech mixes English words in constantly,
#: and demanding a pure script would reject perfectly natural code-mixed
#: replies. An English session is held to a higher bar, because there is no
#: equivalent reason for an English answer to be mostly Devanagari.
_MIN_SCRIPT_RATIO_INDIC = 0.30
_MIN_SCRIPT_RATIO_ENGLISH = 0.70

_SCRIPT_RANGES: dict[Language, tuple[int, int]] = {
    Language.HINDI: (0x0900, 0x097F),
    Language.BENGALI: (0x0980, 0x09FF),
    Language.TAMIL: (0x0B80, 0x0BFF),
    Language.TELUGU: (0x0C00, 0x0C7F),
    Language.KANNADA: (0x0C80, 0x0CFF),
}


def _script_ratio(text: str, language: Language) -> float:
    """Share of alphabetic characters that belong to the session's script.

    English is checked too, not waved through: a Telugu reply in an English
    session is just as wrong as the reverse, and only checking Indic scripts
    let that through.
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0

    if language is Language.ENGLISH:
        return sum(1 for c in letters if c.isascii()) / len(letters)

    bounds = _SCRIPT_RANGES.get(language)
    if bounds is None:
        return 1.0
    low, high = bounds
    return sum(1 for c in letters if low <= ord(c) <= high) / len(letters)


def _mentions_foreign_entity(reply: str, session: SessionData) -> str | None:
    """Detect a service or city asserted in the reply that we never captured."""
    lowered = reply.casefold()

    captured_service = (session.requested_service or "").casefold()
    for canonical in SERVICE_CATALOG:
        if canonical in lowered and canonical not in captured_service:
            return f"names service {canonical!r} that was never captured"

    captured_area = (session.city_or_area or "").casefold()
    for city in CITY_SEEDS:
        if re.search(rf"\b{re.escape(city)}\b", lowered) and city not in captured_area:
            return f"names city {city!r} that was never captured"

    return None


def validate(reply: str, session: SessionData, language: Language) -> tuple[bool, str]:
    """Check a generated reply against the current state.

    Returns `(ok, reason)`; `reason` is empty when the reply is acceptable and
    describes the violation otherwise, so failures can be logged and studied
    rather than silently swallowed.
    """
    if not reply or not reply.strip():
        return False, "empty"

    reply = reply.strip()
    state = session.state

    if len(reply) > MAX_REPLY_CHARS:
        return False, f"too long ({len(reply)} chars)"
    if _EMOJI.search(reply):
        return False, "contains emoji"
    if _MARKDOWN.search(reply):
        return False, "contains markdown"
    if _URL.search(reply):
        return False, "contains a URL"
    if _LONG_NUMBER.search(reply):
        return False, "contains a fabricated number"

    if state not in _PROMISE_ALLOWED:
        lowered = reply.casefold()
        for term in _PROMISE_TERMS:
            if term in lowered:
                return False, f"promises {term!r} before confirmation"

    foreign = _mentions_foreign_entity(reply, session)
    if foreign:
        return False, foreign

    if state in _MUST_ASK and "?" not in reply:
        return False, "does not ask the pending question"

    threshold = (
        _MIN_SCRIPT_RATIO_ENGLISH
        if language is Language.ENGLISH
        else _MIN_SCRIPT_RATIO_INDIC
    )
    ratio = _script_ratio(reply, language)
    if ratio < threshold:
        return False, f"wrong script for {language.value} ({ratio:.0%} in-script)"

    return True, ""

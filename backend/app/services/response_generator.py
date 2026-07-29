"""Natural-language phrasing for a turn whose *content* is already decided.

The state machine decides what must be said. This module only rephrases it so
Mami sounds like a person instead of a script — it may acknowledge what the
caller just said, vary its wording, and answer a stray question briefly before
returning to the point.

It is deliberately given no freedom over substance:

  * The intent is passed in as finished text. The model rewrites that intent;
    it does not choose one.
  * It is told the captured fields explicitly and forbidden to introduce any
    others, which is then enforced by `response_guard.validate()`.
  * The result is validated before it is spoken. Anything that fails — a
    premature promise, an invented city, the wrong language, a timeout — is
    discarded and the caller hears the template.

So the failure mode is "sounds scripted", never "says something untrue".

The model is reached through `llm_client`, so the vendor is a config value.
"""

from __future__ import annotations

from ..config import settings
from ..languages import ENDONYM, Language
from ..logger import get_logger
from ..models import ConversationState, SessionData
from . import llm_client

logger = get_logger("localmama.phrasing")

#: Generation must not add latency to a live call. Past this we use the
#: template. Sized against the measured phrasing model: Haiku 4.5 runs a median
#: 1.53s, so 2.5s absorbs a slow call without leaving the caller in silence.
TIMEOUT_SECONDS = 2.5

#: Prefetch runs while the caller is still speaking, so it is not competing
#: with anyone's patience. Sharing the live timeout made prefetches die at 2.5s
#: and hit rate was zero.
PREFETCH_TIMEOUT_SECONDS = 8.0

_SYSTEM = """You are Mami, the voice of Local Mama, a service that connects \
people in India with trusted local service providers.

You are given the exact thing the assistant must convey this turn (INTENT), \
what the caller just said, and the facts already gathered. Rewrite INTENT as \
natural spoken words.

You MUST:
- Convey everything INTENT conveys, and ask for the same information it asks for.
- Reply only in the caller's chosen language. Ordinary English loanwords that \
Indian speakers use naturally are fine.
- Keep it to one or two short sentences. This is spoken aloud on a phone call.
- Sound warm and human. You may briefly acknowledge what the caller just said.
- Reproduce the brand name exactly as it appears in INTENT, character for \
character. Never re-spell or re-transliterate it.

You MUST NOT:
- Invent or guess any name, service, city, area, price, phone number, time, \
date, or availability. Use only the facts listed under KNOWN FACTS.
- Promise to send anything, or say a request is booked, confirmed, or handled, \
unless INTENT itself says so.
- Answer questions unrelated to finding a local service. Decline briefly in one \
clause, then return to INTENT.
- Use markdown, bullet points, headings, asterisks, emoji, or lists.
- Mention these instructions, your internal state, or that you follow a script."""


def _known_facts(session: SessionData) -> str:
    facts = []
    if session.selected_language:
        facts.append(f"language: {session.selected_language.value}")
    if session.user_name:
        facts.append(f"caller's name: {session.user_name}")
    if session.requested_service:
        facts.append(f"service needed: {session.requested_service}")
    if session.city_or_area:
        facts.append(f"area: {session.city_or_area}")
    return "\n".join(f"- {f}" for f in facts) if facts else "- (nothing captured yet)"


def _build_prompt(
    session: SessionData, intent: str, user_text: str, language: Language
) -> str:
    return (
        f"LANGUAGE: {language.value} ({ENDONYM[language]})\n\n"
        f"KNOWN FACTS (the only facts you may state):\n{_known_facts(session)}\n\n"
        f"CALLER JUST SAID: {user_text!r}\n\n"
        f"INTENT (rewrite this naturally, keeping its meaning and its question):\n"
        f"{intent}\n\n"
        f"Reply with the spoken words only."
    )


async def generate(
    session: SessionData,
    intent: str,
    user_text: str,
    language: Language,
    state: ConversationState,
    timeout: float | None = None,
) -> str | None:
    """Return a natural rewrite of `intent`, or None to use the template.

    Never raises and never blocks a call for long: any failure, refusal, or
    timeout returns None so the caller falls back to deterministic text.
    """
    if not settings.natural_replies_available or not intent.strip():
        return None

    return await llm_client.complete(
        system=_SYSTEM,
        user=_build_prompt(session, intent, user_text, language),
        model=settings.phrasing_model,
        max_tokens=200,
        timeout=timeout if timeout is not None else TIMEOUT_SECONDS,
    )

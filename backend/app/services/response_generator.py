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


def _known_facts(session: SessionData, *, masked: bool = False) -> str:
    """The only facts the model may state.

    `masked` replaces the three interpolated values with their placeholders,
    used when phrasing ahead of time. Listing the real name here while the
    INTENT carries {{NAME}} invites the model to use both — which reads as
    "Ravi garu, ... for Ravi" once the placeholder is filled in.
    """
    facts = []
    if session.selected_language:
        facts.append(f"language: {session.selected_language.value}")
    name = PLACEHOLDERS["name"] if masked else session.user_name
    service = PLACEHOLDERS["service"] if masked else session.requested_service
    area = PLACEHOLDERS["location"] if masked else session.city_or_area
    if name:
        facts.append(f"caller's name: {name}")
    if service:
        facts.append(f"service needed: {service}")
    if area:
        facts.append(f"area: {area}")
    return "\n".join(f"- {f}" for f in facts) if facts else "- (nothing captured yet)"


#: Stand-ins for values the caller has not given yet, so a line can be phrased
#: *before* they say it. `conversation_manager` substitutes the real values in
#: at speak time. Double braces because models reproduce that shape reliably
#: and it survives translation into an Indic script, which a bare word does not.
PLACEHOLDERS: dict[str, str] = {
    "name": "{{NAME}}",
    "service": "{{SERVICE}}",
    "location": "{{LOCATION}}",
}

_PLACEHOLDER_RULE = """
PLACEHOLDERS: the INTENT contains tokens like {{NAME}}, {{SERVICE}} and \
{{LOCATION}}. These stand for facts you have not been told yet.
- Reproduce every placeholder EXACTLY as written, including both braces and the \
capital letters, in the position where that value belongs in your sentence.
- Do NOT translate, transliterate, decline, or invent a value for a placeholder. \
Do not add case endings inside the braces.
- Do not drop a placeholder that appears in INTENT, and do not add one that does not.
- KNOWN FACTS may list the real value for a fact that also appears as a \
placeholder. Use the PLACEHOLDER, never the literal value, and never both — \
writing the name out and keeping {{NAME}} makes it appear twice when the \
placeholder is filled in.
"""


def _build_prompt(
    session: SessionData, intent: str, user_text: str, language: Language
) -> str:
    # Only shown when there is a placeholder to explain — an unconditional rule
    # would spend tokens and invite the model to invent braces on normal turns.
    placeholder_mode = "{{" in intent
    rule = _PLACEHOLDER_RULE if placeholder_mode else ""
    return (
        f"LANGUAGE: {language.value} ({ENDONYM[language]})\n\n"
        f"KNOWN FACTS (the only facts you may state):\n"
        f"{_known_facts(session, masked=placeholder_mode)}\n\n"
        f"CALLER JUST SAID: {user_text!r}\n\n"
        f"INTENT (rewrite this naturally, keeping its meaning and its question):\n"
        f"{intent}\n"
        f"{rule}\n"
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

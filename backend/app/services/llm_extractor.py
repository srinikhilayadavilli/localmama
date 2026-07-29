"""LLM-assisted entity extraction — the fallback when rules find nothing.

Deliberately narrow in scope. The LLM is *not* allowed to decide what happens
next in the conversation; it only reads one utterance and returns the fields it
can see. Workflow progression stays in `state_machine.py`.

Uses structured outputs so the response is schema-validated JSON rather than
prose we would have to parse defensively.
"""

from __future__ import annotations

import json

from ..config import settings
from ..languages import Language
from ..logger import get_logger
from ..models import ConversationState, Extraction
from . import llm_client

logger = get_logger("localmama.llm")

_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {
            "type": ["string", "null"],
            "description": "The caller's personal name, or null if not stated.",
        },
        "requested_service": {
            "type": ["string", "null"],
            "description": (
                "The local service requested, as a short lowercase English noun "
                "phrase such as 'electrician' or 'ac repair'. Null if not stated."
            ),
        },
        "city_or_area": {
            "type": ["string", "null"],
            "description": (
                "The city, locality, or area mentioned, in Title Case. "
                "Null if not stated."
            ),
        },
        "language": {
            # No sibling "type" here. Pairing `type: ["string","null"]` with an
            # enum makes Anthropic's structured-output validator reject the whole
            # request — "Enum value 'english' does not match declared type
            # ['string','null']" — which failed every extraction call with a 400
            # and silently disabled the LLM fallback. The enum alone constrains
            # the value, null included.
            "enum": [*[lang.value for lang in Language], None],
            "description": "Which supported language the caller asked to use, if any.",
        },
        "confidence": {
            "type": "number",
            "description": "0.0 to 1.0 confidence in the extracted values.",
        },
    },
    "required": ["name", "requested_service", "city_or_area", "language", "confidence"],
    "additionalProperties": False,
}

_SYSTEM = """You extract structured fields from a single utterance spoken by a \
caller to an Indian local-services phone assistant.

Rules:
- Extract only what the caller actually said. Never invent a name, service, or \
location. If a field is absent, return null for it.
- The caller may speak English, Hindi, Bengali, Telugu, Tamil, or Kannada, in \
native script or romanised, and may mix English words in.
- Speech-to-text output is often noisy; ignore filler words.
- Normalise the service to a short lowercase English noun phrase.
- Normalise the location to Title Case, keeping any locality qualifier.
- Return a person's name only when it is clearly a name, not a service request."""


def _build_user_prompt(text: str, expecting: ConversationState | None) -> str:
    asked = {
        ConversationState.LANGUAGE_SELECTION: "which language they want to speak",
        ConversationState.ASK_NAME: "their name",
        ConversationState.ASK_SERVICE: "which service they need",
        ConversationState.ASK_LOCATION: "which city or area they need it in",
    }.get(expecting or ConversationState.WELCOME)

    context = f"The assistant just asked the caller for {asked}.\n" if asked else ""
    return f"{context}Caller said: {text!r}\n\nExtract the fields."


async def extract_with_llm(
    text: str, expecting: ConversationState | None = None
) -> Extraction | None:
    """Return an `Extraction`, or None if the LLM is unavailable or errors.

    Never raises: a failed fallback must degrade to a re-prompt, not drop the
    call. The caller treats None as "rules found nothing and we still know
    nothing".
    """
    if not settings.llm_available or not text.strip():
        return None

    payload = await llm_client.complete(
        system=_SYSTEM,
        user=_build_user_prompt(text, expecting),
        model=settings.llm_model,
        max_tokens=2000,
        timeout=10.0,
        json_schema=_EXTRACTION_SCHEMA,
    )
    if not payload:
        return None

    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        logger.warning("LLM returned non-JSON payload")
        return None

    language = None
    if data.get("language"):
        try:
            language = Language(data["language"])
        except ValueError:
            language = None

    return Extraction(
        name=data.get("name") or None,
        requested_service=data.get("requested_service") or None,
        city_or_area=data.get("city_or_area") or None,
        language=language,
        confidence=float(data.get("confidence") or 0.0),
        raw_text=text,
        source="llm",
    )

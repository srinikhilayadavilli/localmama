"""The bridge between Gemini Live and the deterministic engine.

Gemini does the talking. It does **not** get to decide what was said. The only
way a fact reaches a lead is through one of these tools, and every one of them
runs the value through the same validation the typed pipeline uses:

  * `sanitize_field` — strips control characters and markup, caps length.
  * `entity_extractor` — normalises a spoken phrase to a canonical value, so
    "నేను ఎలక్ట్రిషన్ కోసం వెతుకుతున్నాను" is stored as `electrician`, not as a
    whole sentence, and a misheard trade is rescued or rejected.
  * `state_machine.missing_fields` — `save_lead` refuses while anything is
    outstanding, so the model cannot declare a call finished early.

What this recovers from the free-form path: leads actually persist, values are
normalised and sanitised, and completion is gated. What it still does not
recover: the *order* of questions, and the guarantee that a read-back happened.
Gemini is asked to do both, and asking is weaker than enforcing — see README.
"""

from __future__ import annotations

from typing import Annotated

from livekit.agents import function_tool

from .languages import resolve_language
from .logger import get_logger
from .models import ConversationState, ConversationStatus, SessionData, utcnow
from .persistence import save_lead, save_transcript
from .security import sanitize_field
from .services.entity_extractor import extract
from .state_machine import missing_fields

logger = get_logger("localmama.realtime.tools")


class LeadRecorder:
    """Holds the session and exposes the tools Gemini may call."""

    def __init__(self, caller_id: str | None = None) -> None:
        import uuid

        self.session = SessionData(session_id=str(uuid.uuid4()))
        self.caller_id = caller_id
        self.saved = False

    # -- helpers ---------------------------------------------------------

    def _short(self) -> str:
        return self.session.session_id[:8]

    def snapshot(self) -> dict:
        s = self.session
        return {
            "language": s.selected_language.value if s.selected_language else None,
            "name": s.user_name,
            "service": s.requested_service,
            "area": s.city_or_area,
        }

    # -- tools -----------------------------------------------------------

    def build_tools(self) -> list:
        rec = self

        @function_tool(
            name="set_language",
            description=(
                "Record which language the caller chose. Call this as soon as they "
                "say it. One of: english, hindi, bengali, telugu, tamil, kannada."
            ),
        )
        async def set_language(
            language: Annotated[str, "The language the caller chose."],
        ) -> str:
            resolved = resolve_language(language)
            if resolved is None:
                logger.info("session=%s  tool set_language rejected %r", rec._short(), language)
                return f"'{language}' is not a supported language. Ask them again."
            rec.session.selected_language = resolved
            logger.info("session=%s  CAPTURED language=%s", rec._short(), resolved.value)
            return f"Recorded language: {resolved.value}."

        @function_tool(
            name="set_name",
            description="Record the caller's name, exactly as they said it.",
        )
        async def set_name(
            name: Annotated[str, "The caller's name."],
        ) -> str:
            cleaned = sanitize_field(name)
            if not cleaned:
                return "That did not look like a name. Ask them again."
            rec.session.user_name = cleaned
            logger.info("session=%s  CAPTURED name=%r", rec._short(), cleaned)
            return f"Recorded name: {cleaned}."

        @function_tool(
            name="set_service",
            description=(
                "Record the service the caller needs, in their own words. "
                "It will be normalised — do not translate it yourself."
            ),
        )
        async def set_service(
            service: Annotated[str, "The service the caller asked for."],
        ) -> str:
            # Reuse the rule extractor so a spoken phrase in any language lands
            # on a canonical trade, and a garbled one is refused.
            result = extract(service, expecting=ConversationState.ASK_SERVICE)
            value = sanitize_field(result.requested_service or service)
            if not value:
                return "That did not look like a service. Ask them again."
            rec.session.requested_service = value
            logger.info(
                "session=%s  CAPTURED service=%r (from %r, conf=%.2f)",
                rec._short(), value, service, result.confidence,
            )
            return f"Recorded service: {value}."

        @function_tool(
            name="set_area",
            description="Record the city or area where the caller needs the service.",
        )
        async def set_area(
            area: Annotated[str, "The city, locality, or area."],
        ) -> str:
            result = extract(area, expecting=ConversationState.ASK_LOCATION)
            value = sanitize_field(result.city_or_area or area)
            if not value:
                return "That did not look like a place. Ask them again."
            rec.session.city_or_area = value
            logger.info("session=%s  CAPTURED area=%r", rec._short(), value)
            return f"Recorded area: {value}."

        @function_tool(
            name="save_lead",
            description=(
                "Save the lead. Call this ONLY after you have read all the details "
                "back to the caller and they confirmed they are correct. It will "
                "refuse if anything is still missing."
            ),
        )
        async def save_lead_tool() -> str:
            outstanding = missing_fields(rec.session)
            if outstanding:
                logger.warning(
                    "session=%s  save_lead REFUSED, missing %s", rec._short(), outstanding
                )
                readable = ", ".join(
                    {"selected_language": "language", "user_name": "name",
                     "requested_service": "service", "city_or_area": "area"}[f]
                    for f in outstanding
                )
                return (
                    f"Cannot save yet — still missing: {readable}. "
                    f"Ask the caller for it, then try again."
                )

            s = rec.session
            s.state = ConversationState.COMPLETED
            s.conversation_status = ConversationStatus.COMPLETED
            s.completed_at = utcnow()
            save_lead(s.to_lead())
            save_transcript(s)
            rec.saved = True

            from . import caller_profiles

            caller_profiles.remember(rec.caller_id, s)
            logger.info(
                "session=%s  LEAD SAVED: %s / %s / %s (%s)",
                rec._short(), s.user_name, s.requested_service, s.city_or_area,
                s.selected_language.value if s.selected_language else "?",
            )
            logger.info(
                "session=%s  [SIMULATED WHATSAPP] would send %s details in %s to %s",
                rec._short(), s.requested_service, s.city_or_area, s.user_name,
            )
            return "Lead saved. Thank the caller and close the call warmly."

        return [set_language, set_name, set_service, set_area, save_lead_tool]


def prefill_from_profile(rec: LeadRecorder) -> list[str]:
    """Apply returning-caller memory, same as the deterministic path."""
    from . import caller_profiles

    profile = caller_profiles.load(rec.caller_id)
    return caller_profiles.apply_to_session(profile, rec.session)

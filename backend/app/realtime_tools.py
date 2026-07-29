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

import asyncio

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
        #: A language change requested once but not yet confirmed by the
        #: caller. Cleared on any successful set.
        self.pending_language = None

    # -- helpers ---------------------------------------------------------

    def _short(self) -> str:
        return self.session.session_id[:8]

    def snapshot(self) -> dict:
        s = self.session
        return {
            "language": s.selected_language.value if s.selected_language else None,
            "name": s.user_name,
            "service": s.requested_service,
            "city": s.city_or_area,
        }


    def state_line(self) -> str:
        """Everything held so far, appended to every tool result.

        The model tracks the conversation in its own context, and our tools used
        to answer "Recorded name: X." — mentioning only the field just written.
        After a correction it would re-anchor on that reply and re-ask for the
        service and city it had already collected. The values were never lost;
        the model simply stopped being told about them.

        Repeating the full state on every write is cheap and makes forgetting
        structurally hard: the last tool result in context always lists what is
        held and what is outstanding.
        """
        s = self.session
        held = {
            "language": s.selected_language.value if s.selected_language else None,
            "name": s.user_name,
            "service": s.requested_service,
            "city": s.city_or_area,
        }
        have = ", ".join(f"{k}={v}" for k, v in held.items() if v) or "nothing yet"
        missing = [k for k, v in held.items() if not v]
        need = ", ".join(missing) if missing else "nothing — you may read back and save"
        return f" [HELD: {have}. STILL NEEDED: {need}. Do not ask again for anything HELD.]"

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

            current = rec.session.selected_language
            if current is not None and resolved is not current:
                # Changing language mid-call needs asking twice.
                #
                # A caller on a real call said "ఏది?" — Telugu for "which?" —
                # and the model switched the whole conversation to Hindi off
                # that one word. Background noise and a half-heard syllable do
                # the same thing, and the caller then finds themselves in a
                # language they did not ask for with no obvious way back.
                #
                # A genuine request survives being asked to confirm; a misheard
                # one does not, because the caller says "no" and the model never
                # calls this a second time.
                if rec.pending_language is not resolved:
                    rec.pending_language = resolved
                    logger.info(
                        "session=%s  language change %s -> %s needs confirming",
                        rec._short(), current.value, resolved.value,
                    )
                    return (
                        f"The call is already in {current.value}. Do NOT switch yet. "
                        f"Ask the caller, in {current.value}, whether they want to "
                        f"continue in {resolved.value}. Only if they clearly say yes, "
                        f"call this again with {resolved.value}."
                    )
                logger.info(
                    "session=%s  language change %s -> %s confirmed",
                    rec._short(), current.value, resolved.value,
                )

            rec.pending_language = None
            rec.session.selected_language = resolved
            logger.info("session=%s  CAPTURED language=%s", rec._short(), resolved.value)
            return f"Recorded language: {resolved.value}." + rec.state_line()

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
            return f"Recorded name: {cleaned}." + rec.state_line()

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
            return f"Recorded service: {value}." + rec.state_line()

        @function_tool(
            name="set_city",
            # Named for the city rather than the area because that is what the
            # downstream actually uses: the WhatsApp template and the vendor
            # match key on city and ignore the locality. Asking for "area" got
            # answers like "second floor" that no matcher can use.
            description=(
                "Record the CITY where the caller needs the service, e.g. "
                "Hyderabad or Bengaluru. A locality alone (Madhapur, Koramangala) "
                "is acceptable if that is all they give, but prefer the city."
            ),
        )
        async def set_city(
            city: Annotated[str, "The city, or the locality if that is all they said."],
        ) -> str:
            result = extract(city, expecting=ConversationState.ASK_LOCATION)
            value = sanitize_field(result.city_or_area or city)
            if not value:
                return "That did not look like a place. Ask them again."
            rec.session.city_or_area = value
            logger.info("session=%s  CAPTURED city=%r", rec._short(), value)
            return f"Recorded city: {value}." + rec.state_line()

        @function_tool(
            name="lookup_vendor_contact",
            description=(
                "Look up a business's phone number when the caller ASKS for it — "
                "\"what is X's number\", \"how do I contact X\", \"can I call them\". "
                "Only call this when they ask. Never volunteer businesses or "
                "numbers they did not ask about."
            ),
        )
        async def lookup_vendor_contact(
            business: Annotated[str, "The business name the caller asked about."],
        ) -> str:
            from .services import directory

            if not directory.available():
                return "The directory is unavailable. Say you cannot look it up right now."

            # A category is not a request for anyone in particular. "wash" or
            # "events" names a kind of business, so there is no number to give
            # — and reeling off the businesses that happen to match would
            # volunteer vendors the caller never asked about.
            if await asyncio.to_thread(directory.looks_like_a_category, business):
                return (
                    f"{business!r} is a category, not a business. Ask the caller for "
                    f"the NAME of the business they want the number for. Do not list "
                    f"any businesses."
                )

            matches = await directory.find_async(business, limit=4)
            if not matches:
                return (
                    f"No business called {business!r} is listed. Tell the caller you "
                    f"do not have that one, and do NOT guess a number."
                )

            # An exact name is an answer; anything vaguer is a question — asked
            # without naming the candidates, for the same reason as above.
            exact = [m for m in matches if m.name.lower() == business.strip().lower()]
            if len(matches) > 1 and not exact:
                return (
                    f"{business!r} matches more than one business. Ask the caller for "
                    f"the full name of the one they mean. Do not read out the list."
                )

            hit = exact[0] if exact else matches[0]
            if not hit.phone:
                return (
                    f"{hit.name} is listed but has no phone number on record. Tell "
                    f"the caller it is not available and offer to have the team follow up."
                )
            # Grouped so it is read as a dictatable number rather than one long token.
            return (
                f"{hit.name} ({hit.category}): {hit.spoken_phone()}. "
                f"Read the number out clearly, in digit groups, and offer to repeat it."
            )

        @function_tool(
            name="save_lead",
            description=(
                "Save the lead. Call this ONLY after you have read all the details "
                "back to the caller and they confirmed they are correct. It will "
                "refuse if anything is still missing."
            ),
        )
        async def save_lead_tool() -> str:
            # Idempotent. The lead file is keyed by session id so a second call
            # merely rewrites it, but the side effects are not idempotent: the
            # caller would get a second WhatsApp message and their profile's
            # call_count would double. A model that re-confirms at the end of a
            # call — or retries after a slow response — does exactly this.
            if rec.saved:
                logger.info("session=%s  save_lead called again; ignoring", rec._short())
                return "Already saved. Just close the call warmly; do not save again."

            outstanding = missing_fields(rec.session)
            if outstanding:
                logger.warning(
                    "session=%s  save_lead REFUSED, missing %s", rec._short(), outstanding
                )
                readable = ", ".join(
                    {"selected_language": "language", "user_name": "name",
                     "requested_service": "service", "city_or_area": "city"}[f]
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
            lead = s.to_lead()
            save_lead(lead)
            save_transcript(s)
            rec.saved = True

            # Same handoff as the deterministic pipeline, after the lead is
            # already durable. Skips with a log line when unconfigured or when
            # the transport gave us no phone number.
            from .services import brain, lead_store, whatsapp

            lead_store.save(lead, caller_phone=rec.caller_id or "")

            # Name the businesses we actually matched in template param {{4}}
            # instead of the generic "our team is shortlisting" line. The
            # lookup already ran during the call, so this is warm; it is also
            # after the lead is durable, and an empty result just falls back.
            options = ""
            if s.requested_service:
                hits = await brain.retrieve_async(s.requested_service, city=s.city_or_area)
                options = brain.options_line(hits)
            whatsapp.fire(lead, phone=rec.caller_id or "", options=options)

            from . import caller_profiles

            caller_profiles.remember(rec.caller_id, s)
            logger.info(
                "session=%s  LEAD SAVED: %s / %s / %s (%s)",
                rec._short(), s.user_name, s.requested_service, s.city_or_area,
                s.selected_language.value if s.selected_language else "?",
            )
            return "Lead saved. Thank the caller and close the call warmly."

        return [set_language, set_name, set_service, set_city,
                lookup_vendor_contact, save_lead_tool]


def prefill_from_profile(rec: LeadRecorder) -> list[str]:
    """Apply returning-caller memory, same as the deterministic path."""
    from . import caller_profiles

    profile = caller_profiles.load(rec.caller_id)
    return caller_profiles.apply_to_session(profile, rec.session)

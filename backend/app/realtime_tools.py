"""The bridge between a realtime model and the deterministic engine.

The model does the talking. It does **not** get to decide what was said. The
only way a fact reaches a lead is through one of these tools, and every one of
them runs the value through the same four gates:

  * `grounding` — the value must be traceable to what the caller was
    transcribed saying. A realtime model that mishears does not fail loudly; it
    produces a fluent, plausible name, and this is what catches it.
  * `entity_extractor` — normalises a spoken phrase to a canonical value, so
    "నేను ఎలక్ట్రిషన్ కోసం వెతుకుతున్నాను" is stored as `electrician`, not as a
    whole sentence, and a misheard trade is rescued or rejected.
  * `translate` — the value is stored in **English**, whatever language it was
    spoken in. Names and places are transliterated, services translated.
  * `sanitize_field` — strips control characters and markup, caps length.

and `state_machine.missing_fields` gates `save_lead` while anything is
outstanding, so the model cannot declare a call finished early.

Nothing is prefilled from a previous call. `caller_profiles` still records what
a call learned, but this path never reads it back into a session: a name the
caller has not said on *this* call is not a fact, and a remembered one both
skips the question and survives being wrong forever.

What this still does not recover from the free-form path: the *order* of
questions, and the guarantee that a read-back happened. The model is asked to
do both, and asking is weaker than enforcing — see README.
"""

from __future__ import annotations

import asyncio

from collections import deque
from typing import Annotated

from livekit.agents import function_tool

from .languages import resolve_language
from .logger import get_logger
from .models import ConversationState, ConversationStatus, SessionData, utcnow
from .persistence import save_lead, save_transcript
from .security import sanitize_field
from .services import grounding, translate
from .services.entity_extractor import extract, format_name
from .services.grounding import Verdict
from .state_machine import missing_fields

logger = get_logger("localmama.realtime.tools")

#: How many of the caller's turns a value may be grounded in. The model does not
#: always call the tool on the turn the answer arrived — it may greet, then
#: record — but a name from six turns ago is not evidence for what was just
#: said, and widening this only weakens the check.
GROUNDING_WINDOW = 4


def _log_if_failed(task) -> None:  # noqa: ANN001 - asyncio.Task
    """Surface an exception from post-call work.

    asyncio drops the result of a task nobody awaits, so a failure here reached
    neither the logs nor the caller. The lead is on disk and in the outbox
    either way, but a silent failure is one nobody goes looking for.
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("post-call work failed", exc_info=exc)


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
        #: Strong refs to post-call work, so asyncio cannot collect it.
        self.background: set = set()
        #: What the caller was transcribed saying, most recent last. The only
        #: evidence there is that a value came from them rather than from the
        #: model — fed by `agent_realtime`'s `user_input_transcribed` handler.
        self.heard: deque[str] = deque(maxlen=GROUNDING_WINDOW)

    # -- helpers ---------------------------------------------------------

    def _short(self) -> str:
        return self.session.session_id[:8]

    def note_caller_speech(self, text: str) -> None:
        """Record one final transcript of the caller's own audio."""
        if (text or "").strip():
            self.heard.append(text.strip())

    def grounded(self, value: str, *, every_word: bool) -> Verdict:
        """Whether `value` traces back to the caller's recent audio.

        `NO_TRANSCRIPT` is not a failure and must not be treated as one: a
        provider or model with transcription off would otherwise refuse every
        value on the call. It is logged, because running blind is worth knowing
        about.
        """
        verdict = grounding.check(value, list(self.heard), every_word=every_word)
        if verdict is Verdict.NO_TRANSCRIPT:
            logger.warning(
                "session=%s  no caller transcript yet; accepting %r unverified",
                self._short(), value[:40],
            )
        elif verdict is Verdict.NOT_HEARD:
            logger.warning(
                "session=%s  REJECTED %r — not in what the caller said (%s)",
                self._short(), value[:40], list(self.heard),
            )
        return verdict

    def extractor_agrees(self, canonical: str | None) -> bool:
        """Whether the rules reach the same trade from the caller's own words.

        The second chance a service gets and a name does not. "AC repair" shares
        no word with "मुझे एसी ठीक करवाना है" — grounding sees a value the caller
        never said — but the catalogue maps "एसी" to `ac repair` straight off the
        transcript, and two independent routes to the same trade is exactly the
        corroboration being asked for.
        """
        if not canonical:
            return False
        for turn in self.heard:
            found = extract(turn, expecting=ConversationState.ASK_SERVICE)
            if found.requested_service == canonical:
                logger.info(
                    "session=%s  %r corroborated by the rules from %r",
                    self._short(), canonical, turn[:40],
                )
                return True
        return False

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
            description=(
                "Record the caller's name, in the caller's OWN WORDS and script, "
                "exactly as they said it. Do not translate, romanise or correct "
                "it — that is done for you. Only call this after the caller has "
                "actually said their name."
            ),
        )
        async def set_name(
            name: Annotated[str, "The caller's name, exactly as they said it."],
        ) -> str:
            spoken = (name or "").strip()
            # Grounded before anything else, and on every word: a name is a
            # proper noun that exists only because the caller uttered it. This
            # is the check that was missing — a Hindi call filed a fluent,
            # invented name because nothing here ever compared it to the audio.
            if rec.grounded(spoken, every_word=True) is Verdict.NOT_HEARD:
                # With the state line, like every other reply from these tools:
                # a refusal is the moment the model is most likely to re-anchor
                # and start re-asking for details it already holds.
                return (
                    "You did not hear that name in what the caller said. Do NOT "
                    "record a name you are unsure of. Ask them, in their language, "
                    "to say their name again."
                ) + rec.state_line()
            # The same validation the typed pipeline uses: strips "मेरा नाम X है"
            # down to X, and refuses greetings, service requests and chit-chat.
            result = extract(spoken, expecting=ConversationState.ASK_NAME)
            if not result.name:
                return "That did not look like a name. Ask them again."
            # Stored in English, like every other field. Transliterated rather
            # than translated — a translator turns "आशा" into "Hope".
            cleaned = sanitize_field(format_name(await translate.english_name(result.name)))
            if not cleaned:
                return "That did not look like a name. Ask them again."
            rec.session.user_name = cleaned
            logger.info(
                "session=%s  CAPTURED name=%r (heard %r)", rec._short(), cleaned, spoken
            )
            return f"Recorded name: {cleaned}." + rec.state_line()

        @function_tool(
            name="set_service",
            description=(
                "Record the service the caller needs, in their own words and "
                "script. It will be normalised and translated — do not translate "
                "it yourself. Only call this after they have said what they need."
            ),
        )
        async def set_service(
            service: Annotated[str, "The service the caller asked for."],
        ) -> str:
            spoken = (service or "").strip()
            # Reuse the rule extractor so a spoken phrase in any language lands
            # on a canonical trade, and a garbled one is refused.
            result = extract(spoken, expecting=ConversationState.ASK_SERVICE)
            # One content word is enough here, unlike a name: a caller says
            # "मुझे एसी ठीक करवाना है" and "AC repair" is a fair reading of it,
            # even though "repair" was never spoken.
            if (
                rec.grounded(spoken, every_word=False) is Verdict.NOT_HEARD
                and not rec.extractor_agrees(result.requested_service)
            ):
                return (
                    "You did not hear that in what the caller said. Do NOT record "
                    "a service they did not ask for. Ask them again what they need."
                ) + rec.state_line()
            # English, because the catalogue is English and matching is literal:
            # "कार वॉश" matches none of the 50 categories, "car wash" matches one.
            # The rule extractor has already done this for the trades it knows,
            # in which case this is free — nothing non-Latin is left to send.
            english = await translate.english_service(result.requested_service or spoken)
            value = sanitize_field(english)
            if not value:
                return "That did not look like a service. Ask them again."
            rec.session.requested_service = value
            logger.info(
                "session=%s  CAPTURED service=%r (from %r, conf=%.2f)",
                rec._short(), value, spoken, result.confidence,
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
                "Hyderabad or Bengaluru, in their own words and script. A locality "
                "alone (Madhapur, Koramangala) is acceptable if that is all they "
                "give, but prefer the city. Only call this after they said it."
            ),
        )
        async def set_city(
            city: Annotated[str, "The city, or the locality if that is all they said."],
        ) -> str:
            spoken = (city or "").strip()
            # Every word, as with a name: a place name is a proper noun too, and
            # a city the caller never mentioned sends the lead to the wrong one.
            if rec.grounded(spoken, every_word=True) is Verdict.NOT_HEARD:
                return (
                    "You did not hear that place in what the caller said. Do NOT "
                    "record a city you are unsure of. Ask them again where they need "
                    "the service."
                ) + rec.state_line()
            result = extract(spoken, expecting=ConversationState.ASK_LOCATION)
            # Transliterated, not translated: "मोती नगर" is Moti Nagar, and a
            # translator would offer "Pearl City".
            english = await translate.english_place(result.city_or_area or spoken)
            value = sanitize_field(format_name(english))
            if not value:
                return "That did not look like a place. Ask them again."
            rec.session.city_or_area = value
            logger.info(
                "session=%s  CAPTURED city=%r (heard %r)", rec._short(), value, spoken
            )
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
            from .services import brain

            if not brain.available():
                return "The directory is unavailable. Say you cannot look it up right now."

            # Grounded loosely: the caller may have said a two-word name of which
            # the transcript caught one. A wrong name here cannot invent a number
            # — the lookup either finds a record or does not — and an approximate
            # hit is read back for confirmation before any number is given.
            if rec.grounded(business, every_word=False) is Verdict.NOT_HEARD:
                return (
                    "You did not hear that business name in what the caller said. "
                    "Ask them to repeat which business they mean. Do not guess."
                )
            # The catalogue is English and matching is literal, so a name still
            # in the caller's script reaches nothing.
            business = await translate.english_name(business)

            # A category is not a request for anyone in particular. "wash" or
            # "events" names a kind of business, so there is no number to give
            # — and reeling off the businesses that happen to match would
            # volunteer vendors the caller never asked about.
            if await asyncio.to_thread(brain.looks_like_a_category, business):
                return (
                    f"{business!r} is a category, not a business. Ask the caller for "
                    f"the NAME of the business they want the number for. Do not list "
                    f"any businesses."
                )

            matches = await brain.find_business_async(business, limit=4)
            if not matches:
                return (
                    f"No business called {business!r} is listed. Tell the caller you "
                    f"do not have that one, and do NOT guess a number."
                )

            # An exact name is an answer; anything vaguer is a question — asked
            # without naming the candidates, for the same reason as above.
            exact = [m for m in matches if m.title.lower() == business.strip().lower()]
            if len(matches) > 1 and not exact:
                return (
                    f"{business!r} matches more than one business. Ask the caller for "
                    f"the full name of the one they mean. Do not read out the list."
                )

            hit = exact[0] if exact else matches[0]
            # Reached by spelling, not by a literal match. Naming it back is the
            # whole safeguard: "Pax Jewellers" resolving to "Pax Jwellers" is
            # almost certainly right, and giving a stranger's number because it
            # almost certainly was is the one outcome worth a turn to avoid.
            if hit.approximate:
                return (
                    f"No business is spelled {business!r}, but {hit.title} is close. "
                    f"Ask the caller if that is the one they mean BEFORE giving any "
                    f"number. Do not read the number yet."
                )
            if not hit.phone:
                return (
                    f"{hit.title} is listed but has no phone number on record. Tell "
                    f"the caller it is not available and offer to have the team follow up."
                )
            # Grouped so it is read as a dictatable number rather than one long token.
            return (
                f"{hit.title} ({hit.category}): {brain.spoken_phone(hit.phone)}. "
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
            # Local JSON only — a file write, fast enough to keep inline.
            save_lead(lead)
            save_transcript(s)
            rec.saved = True

            from . import caller_profiles

            caller_profiles.remember(rec.caller_id, s)

            # Everything else happens AFTER this tool returns.
            #
            # The caller is waiting in silence for the whole duration of this
            # call: the model cannot speak its closing line until the tool
            # answers. Doing the durable write, the knowledge-base lookup and
            # the WhatsApp send inline cost 15 seconds of dead air on a real
            # call — a synchronous psycopg round trip, a cold embedding model,
            # then three WhatsApp retries against a provider that is down.
            #
            # None of it needs to be finished before she says goodbye, and the
            # lead is already on disk, so a failure here loses nothing.
            async def _finish_in_background() -> None:
                from .services import brain, lead_store, whatsapp

                await asyncio.to_thread(
                    lead_store.save, lead, rec.caller_id or ""
                )
                options = ""
                if s.requested_service:
                    # No translation step here any more: the service and the city
                    # were converted to English when they were captured, so what
                    # is matched is exactly what is stored, shown in the logs, and
                    # sent to the caller. This used to translate a Devanagari
                    # value for the lookup only, which meant the lead and the
                    # WhatsApp message kept a script the rest of the stack — and
                    # the person actioning the lead — could not read.
                    #
                    # Literal match, not semantic: the message names businesses
                    # and gives their numbers, and semantic search matched
                    # "electrician" to an EV charging company.
                    hits = await brain.matches_for_service_async(
                        s.requested_service, city=s.city_or_area
                    )
                    options = brain.options_line(hits)
                # Awaited, not fired: the outcome is recorded so a lead that
                # did not get through stays in the outbox and is retried later,
                # rather than being lost when this process exits.
                result = await whatsapp.send(
                    lead, rec.caller_id or "", options=options
                )
                lead_store.mark_whatsapp(
                    lead.session_id,
                    bool(result.get("ok")),
                    str(result.get("reason") or result.get("error") or ""),
                )

            task = asyncio.create_task(_finish_in_background())
            # Held, or asyncio may collect it mid-await and the lead never
            # reaches Postgres — the same failure that once swallowed turns.
            rec.background.add(task)
            task.add_done_callback(rec.background.discard)
            task.add_done_callback(_log_if_failed)
            logger.info(
                "session=%s  LEAD SAVED: %s / %s / %s (%s)",
                rec._short(), s.user_name, s.requested_service, s.city_or_area,
                s.selected_language.value if s.selected_language else "?",
            )
            return "Lead saved. Thank the caller and close the call warmly."

        return [set_language, set_name, set_service, set_city,
                lookup_vendor_contact, save_lead_tool]



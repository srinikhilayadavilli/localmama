"""The deterministic conversation controller.

This is the source of truth for the call. For each user utterance it:

  1. records the turn,
  2. extracts entities (rules first, LLM only as fallback),
  3. validates and commits any fields it found,
  4. asks the state machine where to go next,
  5. renders the reply from the per-language message table.

The LLM never decides step 4. It only ever contributes candidate field values
in step 2, which are validated before being committed.
"""

from __future__ import annotations

import asyncio
import uuid

from .. import caller_profiles
from ..config import settings
from ..languages import match_language
from ..logger import get_logger, log_transition
from ..models import (
    ConversationState,
    ConversationStatus,
    Extraction,
    SessionData,
    TurnResult,
    utcnow,
)
from ..persistence import save_lead, save_transcript
from ..prompts.messages import MessageKey, get_message
from ..security import TurnLimiter, sanitize_field, sanitize_utterance
from ..state_machine import REQUIRED_FIELD, missing_fields, next_state
from . import (
    confirmation,
    lead_store,
    response_generator,
    response_guard,
    smalltalk,
    whatsapp,
)
from .entity_extractor import (
    MIN_CONFIDENCE,
    UNKNOWN_SERVICE_CONFIDENCE,
    extract,
)
from .llm_extractor import extract_with_llm

logger = get_logger("localmama.conversation")

#: Confidence at which an extraction counts as an explicit, patterned answer
#: rather than a positional guess. Above this, a real answer outranks small talk.
EXPLICIT_CONFIDENCE = 0.9

#: Prompt used when entering each state for the first time.
_PROMPT_FOR_STATE: dict[ConversationState, MessageKey] = {
    ConversationState.LANGUAGE_SELECTION: MessageKey.ASK_LANGUAGE,
    ConversationState.ASK_NAME: MessageKey.ASK_NAME,
    ConversationState.ASK_SERVICE: MessageKey.ASK_SERVICE,
    ConversationState.ASK_LOCATION: MessageKey.ASK_LOCATION,
    ConversationState.REVIEW: MessageKey.REVIEW,
}

#: Prompt used when we failed to extract the field and must ask again.
_REPROMPT_FOR_STATE: dict[ConversationState, MessageKey] = {
    ConversationState.LANGUAGE_SELECTION: MessageKey.LANGUAGE_NOT_UNDERSTOOD,
    ConversationState.ASK_NAME: MessageKey.REPROMPT_NAME,
    ConversationState.ASK_SERVICE: MessageKey.REPROMPT_SERVICE,
    ConversationState.ASK_LOCATION: MessageKey.REPROMPT_LOCATION,
}


class ConversationManager:
    """Drives one session. Transport-agnostic: LiveKit, WebSocket, and tests
    all call `start()` then `handle()` and render whatever text comes back."""

    def __init__(
        self, session: SessionData | None = None, caller_id: str | None = None
    ) -> None:
        self.session = session or SessionData(session_id=str(uuid.uuid4()))
        #: Stable caller identity (phone number / SIP URI) when the transport
        #: provides one. Only ever used as a lookup key; never stored raw.
        self.caller_id = caller_id
        self.profile = caller_profiles.load(caller_id)
        # Per-call turn cap only; wall-clock flood control lives in the transport.
        self.limiter = TurnLimiter()
        self._consecutive_smalltalk = 0
        #: Phrasing generated ahead of time for the line we expect to say next,
        #: keyed by the exact template text it was generated from.
        self._prefetched: dict[str, str] = {}
        self._prefetch_task: asyncio.Task | None = None
        #: Strong refs to post-call work so asyncio cannot collect it.
        self._background: set = set()

    # ------------------------------------------------------------------ #

    def start(self) -> TurnResult:
        """Open the call.

        For a returning caller, known fields are prefilled and the state
        machine skips straight past the questions they answer — no flow logic
        is involved, because `next_state()` already ignores satisfied states.
        """
        session = self.session
        old = session.state

        prefilled = caller_profiles.apply_to_session(self.profile, session)
        if prefilled:
            logger.info(
                "session=%s  returning caller (call #%d), prefilled %s",
                session.session_id[:8],
                self.profile.call_count,
                ", ".join(prefilled),
            )
            session.state = next_state(session)
            greeting = get_message(
                MessageKey.WELCOME_BACK, session.language, name=session.user_name or ""
            )
            # Use the returning-caller phrasing so we don't greet someone we
            # just named as though meeting them for the first time.
            key = _PROMPT_FOR_STATE.get(session.state, MessageKey.ASK_SERVICE)
            if key is MessageKey.ASK_SERVICE:
                key = MessageKey.ASK_SERVICE_RETURNING
            question = self._render(key)
            reply = f"{greeting} {question}"
        else:
            session.state = ConversationState.LANGUAGE_SELECTION
            reply = (
                f"{get_message(MessageKey.WELCOME, session.language)} "
                f"{get_message(MessageKey.ASK_LANGUAGE, session.language)}"
            )

        log_transition(logger, session.session_id, old.value, session.state.value, "call start")
        session.add_turn("agent", reply)
        result = self._result(reply)
        # The greeting is long; use the caller's first reply to get the next
        # line generated for free.
        self._start_prefetch(result)
        return result

    async def handle(self, text: str) -> TurnResult:
        """Process one user utterance and return the agent's reply.

        Deterministic logic runs first and fully decides the turn. Natural
        phrasing is applied afterwards as a presentation layer, so there is no
        path by which generated text can reach the caller unvalidated.
        """
        cleaned = sanitize_utterance(text)
        result = await self._decide(text)
        return await self._phrase(result, cleaned)

    async def _phrase(self, result: TurnResult, user_text: str) -> TurnResult:
        """Rewrite the decided reply naturally, if that is enabled and safe.

        The template is the floor: a generation that is slow, refused, absent,
        or invalid is simply discarded. `response_guard` decides validity
        against the state the workflow is *actually* in.
        """
        if not settings.natural_replies_available:
            return result

        session = self.session

        # A hit here removes the whole LLM round trip from the caller's wait.
        # The cache is keyed on the placeholder form, because the line was
        # phrased before the caller supplied the values it contains.
        candidate = self._prefetched.pop(self._placeholder_form(result.reply), None)
        prefetched = candidate is not None
        if not prefetched:
            try:
                candidate = await response_generator.generate(
                    session=session,
                    intent=result.reply,
                    user_text=user_text,
                    language=result.language,
                    state=result.state,
                )
            except Exception as exc:  # noqa: BLE001
                # llm_client already swallows provider errors, but phrasing is
                # cosmetic and must never be able to end a call. Defence in
                # depth: any escape lands on the template.
                logger.warning("phrasing raised (%s); using template", exc)
                candidate = None

        # Both paths, not just the prefetched one: a live generation can emit a
        # stray brace too, and "Nice to meet you, {{NAME}}" read aloud is the
        # worst outcome available here.
        if candidate is not None:
            filled = self._fill_placeholders(candidate)
            if filled is None:
                logger.info(
                    "session=%s  discarded phrasing with unusable placeholders (%s)",
                    session.session_id[:8],
                    "prefetched" if prefetched else "live",
                )
            candidate = filled

        if candidate is not None and prefetched:
            logger.info("session=%s  used prefetched phrasing", session.session_id[:8])

        if candidate is None:
            self._start_prefetch(result)
            return result

        ok, reason = response_guard.validate(candidate, session, result.language)
        if not ok:
            logger.warning(
                "session=%s  rejected generated reply (%s): %r",
                session.session_id[:8],
                reason,
                candidate[:120],
            )
            self._start_prefetch(result)
            return result

        logger.info("session=%s  natural phrasing applied", session.session_id[:8])
        self._start_prefetch(result)
        # Keep the transcript honest: store what the caller actually heard.
        for turn in reversed(session.transcript):
            if turn.role == "agent":
                turn.text = candidate
                break
        return result.model_copy(update={"reply": candidate})

    #: Values short enough that swapping them for a placeholder could corrupt
    #: unrelated text. A miss costs one live call; a corrupted key would cost
    #: correctness, so anything this short is simply not substituted.
    _MIN_SUBSTITUTABLE = 2

    def _placeholder_form(self, text: str) -> str:
        """Rewrite captured values back into placeholders.

        The inverse of rendering a template, used to look up a line that was
        phrased before those values were known. Exact string replacement is
        safe here because these are the very strings we interpolated.
        """
        session = self.session
        for field, token in (
            (session.user_name, response_generator.PLACEHOLDERS["name"]),
            (session.requested_service, response_generator.PLACEHOLDERS["service"]),
            (session.city_or_area, response_generator.PLACEHOLDERS["location"]),
        ):
            if field and len(field) >= self._MIN_SUBSTITUTABLE:
                text = text.replace(field, token)
        return text

    def _fill_placeholders(self, text: str) -> str | None:
        """Put the real values back, or None if the text cannot be completed.

        None means the model mangled or dropped a placeholder, or a value is
        still unknown. Either way the cached line is unusable and the caller
        falls back to a live generation — never to a half-filled sentence.
        """
        session = self.session
        for field, token in (
            (session.user_name, response_generator.PLACEHOLDERS["name"]),
            (session.requested_service, response_generator.PLACEHOLDERS["service"]),
            (session.city_or_area, response_generator.PLACEHOLDERS["location"]),
        ):
            if token in text:
                if not field:
                    return None
                text = text.replace(token, field)
        # A stray brace means the model invented or garbled a token. Speaking
        # "{{NAM}}" aloud is worse than sounding scripted.
        if "{{" in text or "}}" in text:
            return None
        return text

    def _predicted_next_intent(self) -> str | None:
        """The line we will most likely say next, if the caller answers.

        Only possible because progression is deterministic: fill in the field
        currently being asked for, ask the state machine where that lands, and
        render that state's prompt. A free-form agent cannot know this.

        Values the caller has not given yet are rendered as placeholders rather
        than skipped. Bailing out on them is what held the hit rate to 1 turn in
        6 — every interesting line (the greeting by name, the read-back) embeds
        exactly the value being collected, so those were never prefetched and
        always paid the full 1.1-1.6s live.
        """
        session = self.session
        field = REQUIRED_FIELD.get(session.state)
        if field is None:
            return None

        # Before a language is chosen, any prefetch is phrased in the wrong one
        # and thrown away. Skip rather than spend the call.
        if session.selected_language is None:
            return None

        probe = session.model_copy(deep=True)
        setattr(probe, field, "x")   # pretend the caller answered
        target = next_state(probe)
        key = _PROMPT_FOR_STATE.get(target)
        if key is None:
            return None

        # Every interpolated field becomes a placeholder, not just the unknown
        # ones. `_placeholder_form` cannot tell at lookup time which values were
        # known when the line was phrased, so it swaps all of them — rendering
        # only *some* here made the two keys differ and the read-back never hit.
        return get_message(
            key,
            session.language,
            **{field: token for field, token in response_generator.PLACEHOLDERS.items()},
        )

    def _start_prefetch(self, result: TurnResult) -> None:
        """Generate the next line while the caller is still speaking.

        Fire-and-forget: if it does not finish in time, the turn simply pays
        the normal cost. It can never block or fail a call.
        """
        if not settings.natural_replies_available or result.is_final:
            return
        intent = self._predicted_next_intent()
        if not intent or intent in self._prefetched:
            return

        session = self.session

        async def _run() -> None:
            try:
                text = await response_generator.generate(
                    session=session,
                    intent=intent,
                    user_text="",
                    language=session.language,
                    state=session.state,
                    timeout=response_generator.PREFETCH_TIMEOUT_SECONDS,
                )
                if text:
                    self._prefetched[intent] = text
            except Exception:  # noqa: BLE001 - prefetch must never affect a call
                pass

        try:
            self._prefetch_task = asyncio.create_task(_run())
        except RuntimeError:
            pass   # no running loop (sync context); skip prefetching

    #: An utterance this short that names a language is a correction, not a
    #: passing mention. "Telugu", "Telugu please", "speak in Telugu" all fit;
    #: "my son studies in English medium school" does not.
    _LANGUAGE_SWITCH_MAX_TOKENS = 4

    def _maybe_switch_language(self, text: str) -> TurnResult | None:
        """Let a caller correct a wrong language mid-call.

        Language was captured once and never revisited, so a caller who landed
        in the wrong one — easily done, since a recogniser writing Telugu in
        Devanagari used to select Hindi — was stuck there for the whole call
        with no way to say so.

        Deliberately narrow. It keys on an explicit language *name*
        (`match_language`), never on script detection: the caller speaks the
        chosen language every turn, so script would re-trigger constantly. And
        it requires a short utterance, so mentioning a language in passing does
        not derail a call.
        """
        session = self.session
        if session.selected_language is None:
            return None            # first selection is the normal path
        if len(text.split()) > self._LANGUAGE_SWITCH_MAX_TOKENS:
            return None

        named = match_language(text)
        if named is None or named is session.selected_language:
            return None

        logger.info(
            "session=%s  caller switched language %s -> %s",
            session.session_id[:8], session.selected_language.value, named.value,
        )
        session.selected_language = named
        # Re-ask whatever we were already on, now in the new language. The
        # captured fields stand: they were about the service, not the wording.
        key = _PROMPT_FOR_STATE.get(session.state)
        reply = self._render(key) if key else get_message(
            MessageKey.DIDNT_CATCH, session.language
        )
        return self._result(reply)

    async def _decide(self, text: str) -> TurnResult:
        """Deterministic turn logic. Owns all state and field changes."""
        session = self.session

        if session.state is ConversationState.COMPLETED:
            return self._result(get_message(MessageKey.CLOSING, session.language), final=True)

        # Flood control before any work is done, so a spamming client cannot
        # make us burn CPU or grow the transcript.
        allowed, reason = self.limiter.check()
        if not allowed:
            logger.warning(
                "session=%s  turn rejected: %s", session.session_id[:8], reason
            )
            return self._result(self._render(MessageKey.DIDNT_CATCH))

        # Single sanitisation choke point for every transport. Caps length
        # (the extraction regexes are O(n^2)) and strips control characters.
        raw_len = len(text or "")
        text = sanitize_utterance(text)
        if raw_len > len(text):
            logger.info(
                "session=%s  utterance truncated %d -> %d chars",
                session.session_id[:8],
                raw_len,
                len(text),
            )

        session.add_turn("user", text)

        if not text:
            return self._reprompt("empty utterance")

        switched = self._maybe_switch_language(text)
        if switched is not None:
            return switched

        if session.state is ConversationState.REVIEW:
            return self._handle_review(text)

        expecting = session.state
        extraction = extract(text, expecting=expecting)

        # Chit-chat beats a *weak* reading, but never an explicit one. Without
        # this, "thank you so much" answered "May I know your name?" and was
        # stored as the caller's name, because positional inference happily
        # accepts any bare phrase. An explicit match ("my name is Ravi",
        # confidence 0.9) still wins, so "thanks, my name is Ravi" works.
        if extraction.confidence < EXPLICIT_CONFIDENCE:
            reply = self._maybe_smalltalk(text)
            if reply is not None:
                return reply

        # Fall back to the LLM only when the rules produced nothing usable for
        # the field we actually asked for.
        if not self._satisfies_expected(extraction, expecting):
            llm_result = await extract_with_llm(text, expecting=expecting)
            if llm_result is not None:
                extraction = self._merge(extraction, llm_result)

        # Only commit once the field we actually asked for is satisfied.
        # Committing opportunistically on a failed turn is how a greeting ended
        # up stored as a city.
        if not self._satisfies_expected(extraction, expecting):
            reply = self._maybe_smalltalk(text)
            if reply is not None:
                return reply
            return self._reprompt(f"no value for {REQUIRED_FIELD.get(expecting, 'field')}")

        self._commit(extraction, expecting)
        self._consecutive_smalltalk = 0
        return self._advance()

    def _handle_review(self, text: str) -> TurnResult:
        """Read-back turn: agree, correct a value, or say what is wrong.

        Speech recognition is nondeterministic — the same audio gave "plumber"
        once and "puma" another time on a live call — so this is the last point
        at which a mistranscription can be caught before it becomes a lead.

        Nothing here advances on ambiguity: only explicit agreement moves the
        call forward.
        """
        session = self.session

        if confirmation.is_affirmative(text):
            logger.info("session=%s  caller confirmed the read-back", session.session_id[:8])
            return self._advance()

        # A correction with the value in it — "no, my name is Ravi Kumar",
        # "it's an electrician", "Bangalore". Explicit patterns only: this runs
        # with expecting=None so a bare word cannot silently overwrite a field.
        correction = extract(text, expecting=None)
        applied = self._apply_correction(correction)
        if applied:
            logger.info(
                "session=%s  corrected %s at read-back",
                session.session_id[:8],
                ", ".join(applied),
            )
            reply = self._render(MessageKey.REVIEW)
            session.add_turn("agent", reply)
            return self._result(reply)

        # A field named without a new value — "the area is wrong". Clear it and
        # go back to the question that collects it.
        field = confirmation.field_to_correct(text)
        if field:
            setattr(session, field, None)
            target = next(st for st, f in REQUIRED_FIELD.items() if f == field)
            old = session.state
            session.state = target
            session.retry_counts.pop(target.value, None)
            log_transition(logger, session.session_id, old.value, target.value, f"correcting {field}")
            reply = self._render(_PROMPT_FOR_STATE[target])
            session.add_turn("agent", reply)
            return self._result(reply)

        # A bare "no", or something we could not parse: ask what to change.
        session.bump_retry(ConversationState.REVIEW)
        key = (
            MessageKey.REVIEW_WHAT_TO_CHANGE
            if confirmation.is_negative(text)
            else MessageKey.REVIEW
        )
        reply = self._render(key)
        session.add_turn("agent", reply)
        return self._result(reply)

    def _apply_correction(self, extraction: Extraction) -> list[str]:
        """Overwrite already-captured fields from an explicit correction.

        Distinct from `_commit`, which never overwrites — at the read-back step
        overwriting is exactly the point.
        """
        session = self.session
        applied: list[str] = []
        if extraction.confidence < EXPLICIT_CONFIDENCE:
            return applied

        for field, value in (
            ("user_name", sanitize_field(extraction.name)),
            ("requested_service", sanitize_field(extraction.requested_service)),
            ("city_or_area", sanitize_field(extraction.city_or_area)),
        ):
            if value and value != getattr(session, field):
                setattr(session, field, value)
                applied.append(field)
        return applied

    def _maybe_smalltalk(self, text: str) -> TurnResult | None:
        """Acknowledge chit-chat warmly, then re-ask the pending question.

        Returns None when the utterance is not small talk, or when the caller
        has chatted too many turns in a row and should get a plain re-prompt.

        This deliberately does not touch `session` fields or state — it only
        chooses a phrase. Whatever the caller says here, the workflow position
        is exactly where it was.
        """
        session = self.session
        category = smalltalk.classify(text)
        if category is None:
            return None

        if self._consecutive_smalltalk >= smalltalk.MAX_CONSECUTIVE:
            logger.info(
                "session=%s  small-talk cap reached, re-prompting plainly",
                session.session_id[:8],
            )
            return None

        self._consecutive_smalltalk += 1
        session.bump_retry(session.state)  # still counts toward giving up
        logger.info(
            "session=%s  small talk (%s) in %s — acknowledged, state unchanged",
            session.session_id[:8],
            category.value,
            session.state.value,
        )

        acknowledgement = self._render(category)
        question = self._render(
            _PROMPT_FOR_STATE.get(session.state, MessageKey.DIDNT_CATCH)
        )
        reply = f"{acknowledgement} {question}"
        session.add_turn("agent", reply)
        return self._result(reply)

    # ------------------------------------------------------------------ #

    def _satisfies_expected(
        self, extraction: Extraction, expecting: ConversationState
    ) -> bool:
        field = REQUIRED_FIELD.get(expecting)
        if field is None:
            return True
        value = self._value_for_field(extraction, field)
        if value is None:
            return False

        return extraction.confidence >= self._acceptance_threshold(expecting)

    def _acceptance_threshold(self, expecting: ConversationState) -> float:
        """Confidence required to accept a value for the field being asked.

        A trade we do not recognise is reported below MIN_CONFIDENCE so the
        caller is asked once more — speech recognition turned "plumber" into
        "puma" on a real call, and "puma" reached the lead. After one re-prompt
        we accept it anyway, so a genuinely novel service ("chimney sweep") is
        not blocked forever by a catalog that cannot list everything.

        Both the advance decision and the commit use this, so they can never
        disagree — an earlier version let the state move forward while the
        field stayed empty.
        """
        if (
            expecting is ConversationState.ASK_SERVICE
            and self.session.retries(expecting) >= 1
        ):
            return UNKNOWN_SERVICE_CONFIDENCE
        return MIN_CONFIDENCE

    @staticmethod
    def _value_for_field(extraction: Extraction, field: str):
        return {
            "selected_language": extraction.language,
            "user_name": extraction.name,
            "requested_service": extraction.requested_service,
            "city_or_area": extraction.city_or_area,
        }.get(field)

    @staticmethod
    def _merge(rules: Extraction, llm: Extraction) -> Extraction:
        """Rules win where they found something; the LLM fills the gaps."""
        return Extraction(
            name=rules.name or llm.name,
            requested_service=rules.requested_service or llm.requested_service,
            city_or_area=rules.city_or_area or llm.city_or_area,
            language=rules.language or llm.language,
            confidence=max(rules.confidence, llm.confidence),
            raw_text=rules.raw_text,
            source="llm" if not rules.confidence else "rules",
        )

    def _commit(self, extraction: Extraction, expecting: ConversationState) -> bool:
        """Write validated values onto the session.

        Returns True if the field for the *expected* state was captured, which
        is what decides whether we advance or re-prompt. Extra fields found in
        the same utterance are committed opportunistically so a caller who
        answers three questions at once is not asked again.
        """
        session = self.session
        if (
            extraction.confidence < self._acceptance_threshold(expecting)
            and extraction.language is None
        ):
            return False

        captured: list[str] = []

        # Language is only ever set during selection, then locked for the call.
        # Sanitise before committing: these values are persisted to disk and
        # rendered in the admin page, and may have come from an LLM that read
        # attacker-controlled text.
        name = sanitize_field(extraction.name)
        service = sanitize_field(extraction.requested_service)
        area = sanitize_field(extraction.city_or_area)

        if extraction.language and session.selected_language is None:
            session.selected_language = extraction.language
            captured.append("selected_language")

        if name and not session.user_name:
            session.user_name = name
            captured.append("user_name")

        if service and not session.requested_service:
            session.requested_service = service
            captured.append("requested_service")

        if area and not session.city_or_area:
            session.city_or_area = area
            captured.append("city_or_area")

        if captured:
            logger.info(
                "session=%s  captured %s via %s (conf=%.2f)",
                session.session_id[:8],
                ", ".join(captured),
                extraction.source,
                extraction.confidence,
            )

        expected_field = REQUIRED_FIELD.get(expecting)
        if expected_field is None:
            return True
        return getattr(session, expected_field, None) not in (None, "")

    def _reprompt(self, reason: str) -> TurnResult:
        session = self.session
        attempts = session.bump_retry(session.state)
        logger.info(
            "session=%s  reprompt in %s (attempt %d): %s",
            session.session_id[:8],
            session.state.value,
            attempts,
            reason,
        )
        key = _REPROMPT_FOR_STATE.get(session.state, MessageKey.DIDNT_CATCH)
        # After repeated failures, switch to the simplest possible phrasing.
        if attempts >= 3:
            key = _REPROMPT_FOR_STATE.get(session.state, MessageKey.DIDNT_CATCH)
        reply = self._render(key)
        session.add_turn("agent", reply)
        return self._result(reply)

    def _advance(self) -> TurnResult:
        """Move to the next state and speak its prompt."""
        session = self.session
        old = session.state
        target = next_state(session)

        # Requirement 11: never reach confirmation with a mandatory field empty.
        if target in (ConversationState.CONFIRMATION, ConversationState.CLOSING):
            outstanding = missing_fields(session)
            if outstanding:
                logger.warning(
                    "session=%s  blocked confirmation, missing %s",
                    session.session_id[:8],
                    outstanding,
                )
                target = next(
                    state
                    for state, field in REQUIRED_FIELD.items()
                    if field == outstanding[0]
                )

        session.state = target
        log_transition(logger, session.session_id, old.value, target.value)

        if target is ConversationState.CONFIRMATION:
            return self._finish()

        key = _PROMPT_FOR_STATE.get(target, MessageKey.DIDNT_CATCH)
        reply = self._render(key)
        session.add_turn("agent", reply)
        return self._result(reply)

    def _finish(self) -> TurnResult:
        """Speak confirmation + closing, persist the lead, and end the call.

        Confirmation and closing are separate workflow states but are delivered
        in a single utterance — making the caller wait a turn between "I'll send
        the details" and "thank you" is unnatural on a phone call.
        """
        session = self.session
        confirmation = self._render(MessageKey.CONFIRMATION)
        session.add_turn("agent", confirmation)

        log_transition(
            logger,
            session.session_id,
            ConversationState.CONFIRMATION.value,
            ConversationState.CLOSING.value,
        )
        session.state = ConversationState.CLOSING
        closing = self._render(MessageKey.CLOSING)
        session.add_turn("agent", closing)

        log_transition(
            logger,
            session.session_id,
            ConversationState.CLOSING.value,
            ConversationState.COMPLETED.value,
            "lead captured",
        )
        session.state = ConversationState.COMPLETED
        session.conversation_status = ConversationStatus.COMPLETED
        session.completed_at = utcnow()

        lead = session.to_lead()
        save_lead(lead)
        save_transcript(session)
        caller_profiles.remember(self.caller_id, session)
        logger.info(
            "session=%s  LEAD: %s / %s / %s (%s)",
            session.session_id[:8],
            lead.user_name,
            lead.requested_service,
            lead.city_or_area,
            lead.selected_language.value if lead.selected_language else "?",
        )
        # Everything remote happens after this returns.
        #
        # The caller is listening to silence until the closing line is spoken,
        # and both of these reach the network: a psycopg round trip to Neon and
        # an HTTPS POST that retries three times when the provider is down. Run
        # inline they cost the caller seconds — measured at 15s on the
        # speech-to-speech path before the same fix was applied there.
        #
        # The lead is already on disk above, so nothing is lost if either fails.
        self._finish_durably(lead)

        reply = f"{confirmation} {closing}"
        return self._result(reply, lead=lead, final=True)


    def _finish_durably(self, lead) -> None:  # noqa: ANN001 - models.Lead
        """Persist to Postgres and hand off to WhatsApp, off the caller's clock.

        Falls back to running inline when there is no event loop — the CLI and
        the tests call this synchronously, and there a blocking write is
        correct rather than merely tolerable.
        """
        async def _run() -> None:
            await asyncio.to_thread(lead_store.save, lead, self.caller_id or "")
            whatsapp.fire(lead, phone=self.caller_id or "")

        try:
            task = asyncio.create_task(_run())
        except RuntimeError:
            lead_store.save(lead, caller_phone=self.caller_id or "")
            whatsapp.fire(lead, phone=self.caller_id or "")
            return
        # Held: asyncio keeps only a weak reference, and a collected task means
        # the lead silently never reaching Postgres.
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    def _render(self, key: MessageKey) -> str:
        session = self.session
        return get_message(
            key,
            session.language,
            name=session.user_name or "",
            service=session.requested_service or "",
            location=session.city_or_area or "",
        )

    def _result(self, reply: str, *, lead=None, final: bool = False) -> TurnResult:
        session = self.session
        return TurnResult(
            reply=reply,
            state=session.state,
            language=session.language,
            session=session,
            lead=lead,
            is_final=final,
        )


def abandon(session: SessionData) -> None:
    """Mark a session that ended before completion and persist what we have.

    A call where nothing at all was captured is not a lead — it is noise. A
    Gemini test run that never got past the language question wrote an
    all-null record, which is worse than useless in `data/leads/`.
    """
    if session.conversation_status is ConversationStatus.COMPLETED:
        return
    if not any(
        (session.selected_language, session.user_name,
         session.requested_service, session.city_or_area)
    ):
        logger.info(
            "session=%s  ended with nothing captured; not saving an empty lead",
            session.session_id[:8],
        )
        return
    session.conversation_status = ConversationStatus.ABANDONED
    session.completed_at = utcnow()
    save_lead(session.to_lead())
    save_transcript(session)
    logger.info(
        "session=%s  abandoned in %s, missing %s",
        session.session_id[:8],
        session.state.value,
        missing_fields(session),
    )

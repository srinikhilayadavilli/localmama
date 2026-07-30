"""What the model may do, and the only way a fact leaves this process.

Every tool here is a dictionary write plus a queued event. None of them awaits
the network, none of them touches a database, and none of them normalises a
value — the caller's words go out exactly as they arrived, and the backend
turns them into a lead.

That is a deliberate reversal. These tools used to transliterate a name,
translate a service and match a catalogue while the caller waited, which cost
up to six seconds of a call in network I/O, and which made every capture
unrepeatable: a value normalised at capture can never be re-normalised when the
normaliser improves.

What is still enforced here, because it is a *conversation* concern rather than
a data one:

  * `save_lead` refuses while a mandatory field is missing, so the model cannot
    declare a call finished early. This is a local dictionary check — no
    network, no wait.
  * A mid-call language change has to be asked for twice. A caller said "ఏది?"
    — Telugu for "which?" — and the model switched the whole conversation to
    Hindi off that one word.
  * Every tool result restates everything held. The model tracks the
    conversation in its own context, and a reply naming only the field just
    written made it re-ask for details it already had.

What is NOT enforced here, and deliberately: whether a value is plausible.
A name that was misheard is caught by the read-back while the caller is still
on the line, and by the backend's audit afterwards. Refusing values in a tool
put callers in loops at the exact moment they expected to hang up.
"""

from __future__ import annotations

import re
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Annotated

from livekit.agents import function_tool

from contract import (
    CallCaptured,
    CallEnded,
    CallStarted,
    CallStatus,
    CapturedField,
    Language,
)

from .backend_client import EventQueue, lookup_vendor
from .languages import resolve_language
from .logger import get_logger

logger = get_logger("localmama.tools")

#: A captured value is a name, a trade or a city — a handful of words. Longer
#: than this is a transcription accident, and truncating at the boundary is
#: kinder than storing a paragraph and matching against it.
MAX_FIELD_CHARS = 200

#: Characters that have no place in speech and that make a stored value unsafe
#: to render or log: C0/C1 controls, zero-width and bidirectional overrides,
#: plus markup a transcript should never contain.
_UNSAFE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f​-‏‪-‮⁦-⁩]")
_MARKUP = re.compile(r"[<>{}\\`]")


def sanitize_field(value: str | None) -> str | None:
    """Clean a value before it is stored and sent.

    Returns None when nothing usable survives, so the tool asks again rather
    than recording junk. This is the only validation the agent does — whether a
    value is *plausible* is the backend's question, and answering it here put
    callers in re-prompt loops.
    """
    if not value:
        return None
    value = unicodedata.normalize("NFC", str(value))
    value = _UNSAFE.sub(" ", value)
    value = _MARKUP.sub(" ", value)
    value = re.sub(r"\s+", " ", value).strip(" .,;:!?-")
    if len(value) > MAX_FIELD_CHARS:
        value = value[:MAX_FIELD_CHARS].rstrip()
    return value or None


#: Field order is the order they are asked for, so `still_needed` reads as the
#: next question rather than an unordered set.
MANDATORY = (
    CapturedField.LANGUAGE,
    CapturedField.NAME,
    CapturedField.SERVICE,
    CapturedField.CITY,
)

_SPOKEN = {
    CapturedField.LANGUAGE: "language",
    CapturedField.NAME: "name",
    CapturedField.SERVICE: "service",
    CapturedField.CITY: "city",
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class CallSession:
    """Everything this process knows about one call.

    Four strings and a clock. Every other fact about the caller lives in the
    backend, which is why this can be a dataclass rather than a model.
    """

    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    caller_phone: str | None = None
    dialled: str | None = None
    started_at: datetime = field(default_factory=_now)
    started_mono: float = 0.0

    captured: dict[CapturedField, str] = field(default_factory=dict)
    language: Language | None = None

    #: Whether the details were read back and the caller agreed. `None` means
    #: no read-back happened, which is not the same as the caller disagreeing —
    #: the backend scores them differently.
    confirmed: bool | None = None
    saved: bool = False

    #: A language change asked for once but not yet confirmed by the caller.
    pending_language: Language | None = None

    transcript: list = field(default_factory=list)
    turn_gaps: list[float] = field(default_factory=list)
    _seq: int = 0

    def short(self) -> str:
        return self.call_id[:8]

    def next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def missing(self) -> list[CapturedField]:
        return [f for f in MANDATORY if not self.captured.get(f)]

    def state_line(self) -> str:
        """Everything held so far, appended to every tool result.

        Repeating the full state on every write is cheap and makes forgetting
        structurally hard: the last tool result in context always lists what is
        held and what is outstanding.
        """
        held = ", ".join(
            f"{_SPOKEN[f]}={v}" for f, v in self.captured.items() if v
        ) or "nothing yet"
        missing = self.missing()
        need = (
            ", ".join(_SPOKEN[f] for f in missing) if missing
            else "nothing — read the details back, then save"
        )
        return f" [HELD: {held}. STILL NEEDED: {need}. Do not ask again for anything HELD.]"


class Recorder:
    """Holds one call's session and exposes the tools the model may call."""

    def __init__(self, events: EventQueue, caller_phone: str | None = None,
                 dialled: str | None = None, started_mono: float = 0.0) -> None:
        self.session = CallSession(
            caller_phone=caller_phone, dialled=dialled, started_mono=started_mono
        )
        self.events = events

    # -- events ----------------------------------------------------------

    def _at(self) -> float:
        import time

        return max(0.0, time.monotonic() - self.session.started_mono)

    def announce_start(self, agent_version: str = "") -> None:
        s = self.session
        self.events.send(CallStarted(
            event_id=str(uuid.uuid4()), call_id=s.call_id, seq=s.next_seq(),
            at=0.0, caller_phone=s.caller_phone, dialled=s.dialled,
            started_at=s.started_at, agent_version=agent_version,
        ))

    def _capture(self, field_: CapturedField, value: str, corrected: bool) -> None:
        s = self.session
        s.captured[field_] = value
        self.events.send(CallCaptured(
            event_id=str(uuid.uuid4()), call_id=s.call_id, seq=s.next_seq(),
            at=self._at(), field=field_, value=value, corrected=corrected,
        ))
        logger.info("call=%s  CAPTURED %s=%r%s", s.short(), field_.value,
                    value[:40], "  (corrected)" if corrected else "")

    def announce_end(self, status: CallStatus) -> None:
        s = self.session
        self.events.send(CallEnded(
            event_id=str(uuid.uuid4()), call_id=s.call_id, seq=s.next_seq(),
            at=self._at(), status=status, confirmed=s.confirmed,
            transcript=s.transcript, turn_gaps=s.turn_gaps, ended_at=_now(),
        ))

    def note_caller_speech(self, text: str) -> None:
        """One final transcript of the caller's own audio.

        Kept, and shipped with `call.ended`. It is an independent decode of the
        same audio the model heard, and the only evidence the backend's audit
        has that a captured value came from the caller.
        """
        from contract import TranscriptTurn

        if not (text or "").strip():
            return
        self.session.transcript.append(
            TranscriptTurn(role="caller", text=text.strip(), at=self._at())
        )

    def note_agent_speech(self, text: str) -> None:
        from contract import TranscriptTurn

        if (text or "").strip():
            self.session.transcript.append(
                TranscriptTurn(role="agent", text=text.strip(), at=self._at())
            )

    def caller_spoke_last(self) -> bool:
        """Whether the caller has said anything since Mami last spoke.

        The read-back is only a verification if someone answered it.
        """
        for turn in reversed(self.session.transcript):
            return turn.role == "caller"
        return False

    def snapshot(self) -> dict:
        return {_SPOKEN[f]: v for f, v in self.session.captured.items()}

    # -- tools -----------------------------------------------------------

    def build_tools(self) -> list:
        rec = self
        s = self.session

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
                return f"'{language}' is not a supported language. Ask them again."

            current = s.language
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
                if s.pending_language is not resolved:
                    s.pending_language = resolved
                    logger.info("call=%s  language change %s -> %s needs confirming",
                                s.short(), current.value, resolved.value)
                    return (
                        f"The call is already in {current.value}. Do NOT switch yet. "
                        f"Ask the caller, in {current.value}, whether they want to "
                        f"continue in {resolved.value}. Only if they clearly say yes, "
                        f"call this again with {resolved.value}."
                    )

            corrected = current is not None and resolved is not current
            s.pending_language = None
            s.language = resolved
            rec._capture(CapturedField.LANGUAGE, resolved.value, corrected)
            return f"Recorded language: {resolved.value}." + s.state_line()

        def _setter(field_: CapturedField, noun: str):
            async def record(value: str) -> str:
                cleaned = sanitize_field((value or "").strip())
                if not cleaned:
                    return f"That was empty. Ask them for their {noun} again."
                corrected = bool(s.captured.get(field_))
                rec._capture(field_, cleaned, corrected)
                return f"Recorded {noun}: {cleaned}." + s.state_line()
            return record

        _name, _service, _city = (
            _setter(CapturedField.NAME, "name"),
            _setter(CapturedField.SERVICE, "service"),
            _setter(CapturedField.CITY, "city"),
        )

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
            return await _name(name)

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
            return await _service(service)

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
            return await _city(city)

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
            # The only backend call a caller ever waits on, and the only place
            # this process blocks on the network at all. Hard-bounded and
            # degrades to an honest "I can't look that up" — see backend_client.
            reply = await lookup_vendor(
                business,
                s.captured.get(CapturedField.CITY),
                s.language or Language.ENGLISH,
            )
            logger.info("call=%s  vendor lookup %r -> %s",
                        s.short(), business[:40], reply.match.value)
            return f"{reply.say} {reply.instruction}".strip()

        @function_tool(
            name="save_lead",
            description=(
                "Save the lead. Call this ONLY after you have read the details "
                "back to the caller and they confirmed they are correct. It will "
                "refuse if anything is still missing."
            ),
        )
        async def save_lead() -> str:
            # Idempotent. The model re-confirms at the end of a call, or retries
            # after a slow response, and doing this twice would send the caller
            # two WhatsApp messages.
            if s.saved:
                return "Already saved. Just close the call warmly; do not save again."

            outstanding = s.missing()
            if outstanding:
                readable = ", ".join(_SPOKEN[f] for f in outstanding)
                logger.warning("call=%s  save_lead REFUSED, missing %s",
                               s.short(), readable)
                return (
                    f"Cannot save yet — still missing: {readable}. "
                    f"Ask the caller for it, then try again."
                )

            # Whether the caller actually got to agree, rather than whether we
            # reached this tool.
            #
            # This used to assert True unconditionally, on the reasoning that
            # the flow puts save_lead after the read-back. The flow is a prompt.
            # On a real call the model read the details back and called this
            # two seconds later, with no caller turn in between — and the lead
            # went out marked confirmed, having been confirmed by nobody.
            #
            # A caller turn after the last thing Mami said is the evidence. If
            # the transcript for their "yes" has not landed yet this reads
            # False, which over-flags rather than over-trusts; that is the
            # direction to be wrong in.
            s.confirmed = rec.caller_spoke_last()
            if not s.confirmed:
                logger.warning(
                    "call=%s  saved without the caller answering the read-back",
                    s.short(),
                )
            s.saved = True
            logger.info("call=%s  LEAD COMPLETE: %s", s.short(), rec.snapshot())
            # Nothing is awaited. The captures are already queued, `call.ended`
            # goes out when the session shuts down, and the backend does the
            # rest after the caller has hung up.
            return "Saved. Thank the caller and close the call warmly."

        return [set_language, set_name, set_service, set_city,
                lookup_vendor_contact, save_lead]

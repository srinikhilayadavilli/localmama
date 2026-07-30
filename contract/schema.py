"""The contract between the voice agent and the lead backend.

Two deployables, one file. The agent runs on LiveKit and does nothing but talk;
the backend runs on Render and does everything else. This module is the only
thing they share, and it is imported by both — which is what stops the payload
drifting on one side without the other noticing.

**The agent sends raw.** Every captured value crosses this boundary in the
caller's own words and script: "రవి", not "Ravi"; "కార్ వాష్", not "car wash".
Normalisation, transliteration and translation are the backend's job. That is
not a stylistic preference — putting Sarvam on the capture path cost up to two
seconds per field with the caller listening to silence, and a value normalised
at capture can never be re-normalised later when the normaliser improves.

**Capture is a stream, not a form.** A call that ends after the caller gives
their name and service is still a lead — arguably the most interesting kind,
because something went wrong. Each field is posted as it is captured, so the
backend holds everything up to the moment the caller hung up rather than
nothing at all. `call.ended` closes the record; it does not carry the payload.

**Everything is idempotent.** Events are keyed by `event_id`, calls by
`call_id`. The agent may retry any event any number of times — the backend
deduplicates and the agent never has to reason about whether a send landed.

Versioning: `schema_version` rides on every batch. The backend accepts the
current version and the one before it, so the two sides can be deployed
independently and in either order.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field

#: Bumped when a change is not backwards compatible. Adding an optional field
#: is not such a change; renaming or removing one is.
SCHEMA_VERSION = "1.0"

#: Cap on any single captured value. A name, a trade or a city is a handful of
#: words — anything longer is a transcription accident (a whole sentence, or a
#: stuck recogniser repeating itself) and should be truncated at the boundary
#: rather than stored, matched against and read aloud.
MAX_VALUE_CHARS = 200


class Language(str, Enum):
    """The six languages the agent speaks.

    Lives here rather than in either deployable because both sides key on it:
    the agent to pick a voice, the backend to pick a WhatsApp template.
    """

    ENGLISH = "english"
    HINDI = "hindi"
    BENGALI = "bengali"
    TELUGU = "telugu"
    TAMIL = "tamil"
    KANNADA = "kannada"


class CapturedField(str, Enum):
    """The four things a call exists to collect."""

    LANGUAGE = "language"
    NAME = "name"
    SERVICE = "service"
    CITY = "city"


class CallStatus(str, Enum):
    #: The caller confirmed and the agent closed the call normally.
    COMPLETED = "completed"
    #: The call ended before all four fields were captured — caller hung up,
    #: line dropped, or the agent hit its wall-clock ceiling.
    ABANDONED = "abandoned"


# ---------------------------------------------------------------------------
# Events: agent -> backend
# ---------------------------------------------------------------------------


class _Event(BaseModel):
    """Fields every event carries.

    `seq` is per-call and monotonic. It is not used for ordering — events are
    independent and may arrive in any order — but a gap in it tells the backend
    an event was lost, which is the difference between "the caller never gave a
    city" and "we never received the city".
    """

    event_id: str = Field(description="UUID. The deduplication key.")
    call_id: str = Field(description="UUID minted by the agent when the call is answered.")
    seq: int = Field(ge=0, description="Monotonic per call; gaps mean a lost event.")
    at: float = Field(
        ge=0.0,
        description=(
            "Seconds since the call was answered. An offset rather than a "
            "wall-clock time so the two services need no clock agreement."
        ),
    )


class CallStarted(_Event):
    """The agent answered. Sent before the caller has said anything."""

    type: Literal["call.started"] = "call.started"
    #: E.164. Absent for a browser or WebRTC caller, which is not an error —
    #: the WhatsApp handoff simply has nowhere to go.
    caller_phone: str | None = None
    #: The DID the caller dialled, for routing and attribution.
    dialled: str | None = None
    started_at: datetime = Field(description="Wall clock, for the lead record.")
    agent_version: str = ""


class CallCaptured(_Event):
    """One field, exactly as the caller said it.

    Fire-and-forget: the agent does not await this and the tool that triggered
    it returns immediately. Nothing the caller hears depends on it landing.
    """

    type: Literal["call.captured"] = "call.captured"
    field: CapturedField
    #: RAW. The caller's own words, in the caller's own script, untranslated
    #: and untransliterated. For `field=language` this is the `Language` value,
    #: which is the one field the agent does resolve — it has to, to know which
    #: language to speak next.
    value: str = Field(min_length=1, max_length=MAX_VALUE_CHARS)
    #: True when this replaces a value the caller corrected during the
    #: read-back. A corrected field is better evidence, not worse, and the
    #: backend scores it accordingly.
    corrected: bool = False


class TranscriptTurn(BaseModel):
    """One side of the conversation, as transcribed.

    The caller's turns are an *independent decode* of the same audio the model
    heard — a different model, from the same stream — which is what makes them
    usable as evidence that a captured value came from the caller rather than
    from the model's imagination. That check runs in the backend now.
    """

    role: Literal["caller", "agent"]
    text: str
    at: float = Field(ge=0.0, description="Seconds since the call was answered.")


class CallEnded(_Event):
    """The call is over. Closes the record; carries no captured values.

    Sent on every ending, including an abandoned one — the backend cannot tell
    "still in progress" from "hung up" without it, and a lead that is never
    closed is never processed.
    """

    type: Literal["call.ended"] = "call.ended"
    status: CallStatus
    #: Whether the agent read the details back and the caller agreed.
    #: `None` means no read-back happened at all, which is not the same as the
    #: caller disagreeing and is scored differently: an unconfirmed lead is
    #: usable, it just carries no human verification.
    confirmed: bool | None = None
    transcript: list[TranscriptTurn] = Field(default_factory=list)
    #: Seconds between the caller finishing and the agent starting to reply.
    #: The only latency number that reflects what the caller actually felt.
    turn_gaps: list[float] = Field(default_factory=list)
    ended_at: datetime


#: Discriminated on `type`, so one endpoint accepts all three and the client
#: needs one retry path rather than three.
Event = Annotated[
    CallStarted | CallCaptured | CallEnded,
    Field(discriminator="type"),
]


class EventBatch(BaseModel):
    """`POST /v1/events`.

    A list rather than a single event so the agent's client can queue while the
    backend is briefly unreachable and flush on recovery. A batch is not atomic:
    each event is accepted or rejected on its own, because one malformed event
    must not discard the others in the same flush.
    """

    schema_version: str = SCHEMA_VERSION
    events: list[Event] = Field(min_length=1, max_length=100)


class EventAck(BaseModel):
    """`202 Accepted`. Acknowledges receipt, not processing.

    Processing happens after the response — matching a vendor and sending a
    WhatsApp message take seconds, and the agent is not waiting for them.
    """

    accepted: list[str] = Field(default_factory=list, description="event_ids stored.")
    duplicates: list[str] = Field(default_factory=list, description="Already seen; no-ops.")
    rejected: dict[str, str] = Field(
        default_factory=dict, description="event_id -> why. The agent should not retry these."
    )


# ---------------------------------------------------------------------------
# Vendor lookup: agent -> backend, synchronously, during the call
# ---------------------------------------------------------------------------


class MatchKind(str, Enum):
    #: The caller named a business and there is exactly one with that name.
    EXACT = "exact"
    #: Reached by spelling, not by a literal match. "Pax Jewellers" resolving to
    #: "Pax Jwellers" is almost certainly right — and giving a stranger's number
    #: because it almost certainly was is worth a turn to avoid, so the agent is
    #: told to confirm the name before reading any number.
    APPROXIMATE = "approximate"
    #: The caller named a kind of business ("a car wash"), not a business. There
    #: is no number to give, and listing the businesses that match would
    #: volunteer vendors the caller never asked about.
    CATEGORY = "category"
    #: More than one business matches and none exactly.
    AMBIGUOUS = "ambiguous"
    NONE = "none"


class VendorReply(BaseModel):
    """What the agent should say, and what it must not do.

    `say` is prose, already phrased: digits grouped so a voice reads them as a
    number rather than a year, business name spelled as the caller would hear
    it. The agent reads it out. It does not format phone numbers, decide when a
    match is too weak to share, or reason about categories — all of that is
    directory policy and belongs on the side of the boundary that owns the
    directory.
    """

    match: MatchKind
    #: The line for the agent to speak, in the caller's language.
    say: str
    #: A behavioural constraint for this specific reply, appended to the tool
    #: result. e.g. "Confirm the name before reading the number."
    instruction: str = ""
    #: Present only for EXACT. Structured for the lead record, never for the
    #: agent to format — that is what `say` is for.
    title: str | None = None
    phone: str | None = None

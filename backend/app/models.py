"""Typed models for conversation state, extraction output, and the saved lead."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from .languages import Language


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ConversationState(str, Enum):
    """Deterministic workflow states. The controller — not the LLM — owns these."""

    WELCOME = "WELCOME"
    LANGUAGE_SELECTION = "LANGUAGE_SELECTION"
    ASK_NAME = "ASK_NAME"
    ASK_SERVICE = "ASK_SERVICE"
    ASK_LOCATION = "ASK_LOCATION"
    #: Read the captured details back and wait for the caller to agree. This is
    #: the only defence against a mistranscription reaching a real lead.
    REVIEW = "REVIEW"
    CONFIRMATION = "CONFIRMATION"
    CLOSING = "CLOSING"
    COMPLETED = "COMPLETED"


class ConversationStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class TranscriptTurn(BaseModel):
    role: Literal["agent", "user"]
    text: str
    state: ConversationState
    language: Language | None = None
    at: datetime = Field(default_factory=utcnow)


class Extraction(BaseModel):
    """Structured output of the entity extractor.

    `confidence` is a coarse 0..1 signal: rule matches on an explicit pattern
    score high, bare-phrase fallbacks score low. The conversation manager uses
    it to decide whether to accept a value or re-prompt.
    """

    name: str | None = None
    requested_service: str | None = None
    city_or_area: str | None = None
    language: Language | None = None
    confidence: float = 0.0
    raw_text: str = ""
    source: Literal["rules", "llm", "none"] = "none"


class SessionData(BaseModel):
    """Everything we know about one caller session."""

    session_id: str
    state: ConversationState = ConversationState.WELCOME
    selected_language: Language | None = None
    user_name: str | None = None
    requested_service: str | None = None
    city_or_area: str | None = None
    conversation_status: ConversationStatus = ConversationStatus.IN_PROGRESS
    transcript: list[TranscriptTurn] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    completed_at: datetime | None = None
    #: Per-state count of failed extraction attempts, used to avoid loops.
    retry_counts: dict[str, int] = Field(default_factory=dict)

    @property
    def language(self) -> Language:
        """Active language, defaulting to English before selection happens."""
        return self.selected_language or Language.ENGLISH

    def add_turn(self, role: Literal["agent", "user"], text: str) -> None:
        self.transcript.append(
            TranscriptTurn(
                role=role, text=text, state=self.state, language=self.selected_language
            )
        )
        self.updated_at = utcnow()

    def retries(self, state: ConversationState) -> int:
        return self.retry_counts.get(state.value, 0)

    def bump_retry(self, state: ConversationState) -> int:
        count = self.retries(state) + 1
        self.retry_counts[state.value] = count
        return count

    def to_lead(self) -> "Lead":
        return Lead(
            session_id=self.session_id,
            selected_language=self.selected_language,
            user_name=self.user_name,
            requested_service=self.requested_service,
            city_or_area=self.city_or_area,
            conversation_status=self.conversation_status,
            transcript=self.transcript,
            started_at=self.started_at,
            completed_at=self.completed_at,
        )


class Lead(BaseModel):
    """The structured payload persisted at the end of a call.

    This is the object that would be handed to the WhatsApp dispatcher in
    production. In this MVP it is written to `data/leads/` and shown in the UI.
    """

    session_id: str
    selected_language: Language | None
    user_name: str | None
    requested_service: str | None
    city_or_area: str | None
    conversation_status: ConversationStatus
    transcript: list[TranscriptTurn]
    started_at: datetime
    completed_at: datetime | None


class TurnResult(BaseModel):
    """What the conversation manager returns after handling one user utterance."""

    reply: str
    state: ConversationState
    language: Language
    session: SessionData
    lead: Lead | None = None
    is_final: bool = False

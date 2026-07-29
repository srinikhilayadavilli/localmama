"""Deterministic conversation workflow.

This module is the single source of truth for flow progression. It is pure:
given a state and the set of fields already captured, it decides the next
state. It never calls an LLM and never inspects raw text — entity extraction
happens upstream, and only validated field values reach here.
"""

from __future__ import annotations

from .models import ConversationState, SessionData

#: The field each state is responsible for collecting. States absent from this
#: map (WELCOME, CONFIRMATION, CLOSING, COMPLETED) collect nothing.
REQUIRED_FIELD: dict[ConversationState, str] = {
    ConversationState.LANGUAGE_SELECTION: "selected_language",
    ConversationState.ASK_NAME: "user_name",
    ConversationState.ASK_SERVICE: "requested_service",
    ConversationState.ASK_LOCATION: "city_or_area",
}

#: Linear happy path. Progression only ever moves forward along this order.
FLOW: tuple[ConversationState, ...] = (
    ConversationState.WELCOME,
    ConversationState.LANGUAGE_SELECTION,
    ConversationState.ASK_NAME,
    ConversationState.ASK_SERVICE,
    ConversationState.ASK_LOCATION,
    ConversationState.REVIEW,
    ConversationState.CONFIRMATION,
    ConversationState.CLOSING,
    ConversationState.COMPLETED,
)

#: Mandatory fields that must be present before we may reach CONFIRMATION.
MANDATORY_FIELDS: tuple[str, ...] = (
    "selected_language",
    "user_name",
    "requested_service",
    "city_or_area",
)


def is_satisfied(session: SessionData, state: ConversationState) -> bool:
    """Has the field this state collects already been captured?"""
    field = REQUIRED_FIELD.get(state)
    if field is None:
        return True
    return getattr(session, field, None) not in (None, "")


def missing_fields(session: SessionData) -> list[str]:
    """Mandatory fields still outstanding, in flow order."""
    return [f for f in MANDATORY_FIELDS if getattr(session, f, None) in (None, "")]


def next_state(session: SessionData) -> ConversationState:
    """Advance from the current state to the first state still needing input.

    Because a single utterance may fill several fields at once ("I'm Ravi, I
    need an electrician in Madhapur"), this skips over any state whose field is
    already satisfied rather than stepping exactly one position.
    """
    current = session.state
    if current is ConversationState.COMPLETED:
        return current

    index = FLOW.index(current)
    for candidate in FLOW[index + 1 :]:
        # Never reach confirmation while a mandatory field is outstanding.
        if candidate is ConversationState.CONFIRMATION and missing_fields(session):
            # Fall back to the earliest unsatisfied collecting state.
            for earlier in FLOW:
                if earlier in REQUIRED_FIELD and not is_satisfied(session, earlier):
                    return earlier
        if not is_satisfied(session, candidate):
            return candidate
        # REVIEW collects no field, so `is_satisfied` would wave it through.
        # It must be a hard stop: the caller has to agree before we commit.
        if candidate in (
            ConversationState.REVIEW,
            ConversationState.CONFIRMATION,
            ConversationState.CLOSING,
            ConversationState.COMPLETED,
        ):
            return candidate
    return ConversationState.COMPLETED

"""Types for the rule-based extractor.

`Expecting` is what remains of the agent's conversation state machine on this
side of the boundary. The backend does not run a conversation — but the
extractor's accuracy depends on knowing *which question was being answered*,
because a bare "Ravi" is a name only if someone just asked for one, and a bare
"Madhapur" is a place only if someone asked where. The agent knows that, and
tells us via the captured field.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel

from contract import CapturedField


class Expecting(str, Enum):
    """Which question the value being extracted was an answer to."""

    ASK_NAME = "ASK_NAME"
    ASK_SERVICE = "ASK_SERVICE"
    ASK_LOCATION = "ASK_LOCATION"


#: The agent captures fields; the extractor thinks in questions. One map, here,
#: rather than the caller of `extract` knowing about both vocabularies.
FOR_FIELD: dict[CapturedField, Expecting] = {
    CapturedField.NAME: Expecting.ASK_NAME,
    CapturedField.SERVICE: Expecting.ASK_SERVICE,
    CapturedField.CITY: Expecting.ASK_LOCATION,
}


class Extraction(BaseModel):
    """Structured output of the extractor.

    `confidence` is a coarse 0..1 signal: rule matches on an explicit pattern
    score high, bare-phrase fallbacks score low.
    """

    name: str | None = None
    requested_service: str | None = None
    city_or_area: str | None = None
    language: str | None = None
    confidence: float = 0.0
    raw_text: str = ""
    source: Literal["rules", "none"] = "none"

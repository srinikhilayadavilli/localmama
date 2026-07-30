"""Caller identity from a LiveKit SIP participant.

A call arriving over a SIP trunk carries the caller's number in participant
attributes, and it is the only thing that gives the WhatsApp handoff somewhere
to send to. Browser and WebRTC callers have none, which is not an error: the
lead is still captured and recorded, it just cannot be delivered.

Attribute names verified against the Vaani bridge, which reads the same fields
off a live trunk.
"""

from __future__ import annotations

from .logger import get_logger

logger = get_logger("localmama.telephony")

#: Set by LiveKit on the SIP participant. `sip.phoneNumber` is the caller's
#: number (the "from"); `sip.trunkPhoneNumber` is the DID they dialled.
CALLER_NUMBER_ATTR = "sip.phoneNumber"
DIALLED_NUMBER_ATTR = "sip.trunkPhoneNumber"


def caller_id(participant) -> str:  # noqa: ANN001 - LiveKit participant object
    """The caller's phone number, or "" for a non-SIP participant.

    Never raises: a missing attribute must cost the phone number, not the call.
    """
    if participant is None:
        return ""
    try:
        attrs = getattr(participant, "attributes", None) or {}
        number = str(attrs.get(CALLER_NUMBER_ATTR) or "").strip()
        if number:
            dialled = str(attrs.get(DIALLED_NUMBER_ATTR) or "").strip()
            # Logged masked: a phone number is the one piece of caller PII here,
            # and logs are the easiest place to leak it.
            logger.info(
                "SIP call from %s to %s", mask(number), dialled or "(unknown DID)"
            )
        return number
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not read caller id: %s", exc)
        return ""


def dialled_number(participant) -> str:  # noqa: ANN001 - LiveKit participant object
    """The DID the caller dialled, or "". For routing and attribution.

    Never raises: a missing attribute must cost an analytics field, not the call.
    """
    if participant is None:
        return ""
    try:
        attrs = getattr(participant, "attributes", None) or {}
        return str(attrs.get(DIALLED_NUMBER_ATTR) or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def mask(number: str) -> str:
    """"+919739960092" -> "+91****0092". For logs and dashboards."""
    if not number:
        return ""
    tail = number[-4:]
    return f"{number[:3]}****{tail}" if len(number) > 7 else "****" + tail

"""Caller identity from a LiveKit SIP participant.

A call arriving over a SIP trunk carries the caller's number in participant
attributes. That single value unlocks two things nothing else can provide:

  * **The WhatsApp handoff.** `services/whatsapp.py` has no one to send to
    without it, which is why every browser call skips the send.
  * **Returning-caller memory.** `caller_profiles` is keyed by a salted hash of
    the caller id, so it stays dormant for anonymous callers and only starts
    remembering people once telephony is wired.

Browser and WebRTC callers have no number, and that is not an error — both
features are designed to degrade to nothing rather than fail.

Attribute name verified against the Vaani bridge, which reads the same field
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


def mask(number: str) -> str:
    """"+919739960092" -> "+91****0092". For logs and dashboards."""
    if not number:
        return ""
    tail = number[-4:]
    return f"{number[:3]}****{tail}" if len(number) > 7 else "****" + tail

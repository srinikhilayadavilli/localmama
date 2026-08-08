"""Lead handoff over a signed webhook.

Replaces the CampaignBot WhatsApp sender. The delivery mechanism changed; the
contract around it did not. This still fires once a lead is saved, still never
raises, and is still best-effort by construction — the lead is on disk before
this runs, so a receiver that is down, slow or misconfigured costs us a retry
and never a lead.

What goes in the body is every detail the pipeline produced: who called, what
they asked for, where, what we matched, how much of it we believe, and whether
a human should look. What does not go in it:

  * the WhatsApp columns, which are the record of a channel that no longer
    exists and mean nothing to a receiver;
  * the transcript, which is everything the caller said and personal data under
    the DPDP Act. It stays here, under a retention sweep, rather than being
    copied to a third party that has its own retention rules and its own
    breaches. A receiver that needs it can ask for a call_id.

Configuration:

  WEBHOOK_URL       where to POST. Blank disables the handoff cleanly.
  WEBHOOK_SECRET    HMAC-SHA256 key for the signature header.
  WEBHOOK_TIMEOUT   per-attempt timeout in seconds (default 10).
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time

import httpx

from ..config import settings
from ..logger import get_logger
from . import meter

logger = get_logger("localmama.webhook")

#: Bumped when the body's shape changes in a way a receiver must care about.
#: Sent in the payload and in a header, so a receiver can reject what it does
#: not understand instead of silently misreading it.
PAYLOAD_VERSION = "1"

SIGNATURE_HEADER = "X-Localmama-Signature"
TIMESTAMP_HEADER = "X-Localmama-Timestamp"
VERSION_HEADER = "X-Localmama-Version"


def _norm_phone(raw: str) -> str:
    """Digits to E.164-ish.

    Kept from the WhatsApp sender, where it was learned the hard way: Indian
    callers arrive as "09739960092", and prepending a country code without
    stripping the national trunk "0" yields "+0…" — which the old provider
    accepted and never delivered. The receiver deserves a number it can dial.
    """
    digits = "".join(ch for ch in str(raw or "") if ch.isdigit())
    if not digits:
        return ""
    digits = digits.lstrip("0")
    if len(digits) == 10:          # bare Indian mobile
        digits = "91" + digits
    return "+" + digits


def build_payload(lead: dict, *, subject: str = "", vendors: list | None = None) -> dict | None:
    """Map a processed lead onto the webhook body.

    None means there is no usable recipient — an anonymous or WebRTC caller —
    which is not an error. A lead nobody can be called back on is still a lead
    worth storing; it is just not one worth delivering.
    """
    number = _norm_phone(lead.get("caller_phone") or "")
    if not number:
        return None

    vendors = vendors if vendors is not None else (lead.get("vendors") or [])
    confidence = lead.get("confidence") or {}

    return {
        "version": PAYLOAD_VERSION,
        "event": "lead.captured",
        "call_id": lead.get("call_id"),
        "caller": {
            "phone": number,
            "language": lead.get("language") or None,
            "dialled": lead.get("dialled") or None,
        },
        "lead": {
            "name": lead.get("name") or None,
            # What they asked for, three ways, because they differ and the
            # difference matters: `service` is normalised, `service_said` is
            # their own words, `service_inferred` is what the agent understood
            # when their words matched nothing.
            "service": lead.get("service") or None,
            "service_said": lead.get("service_said") or None,
            "service_inferred": lead.get("service_inferred") or None,
            "city": lead.get("city") or None,
            # Usually the service; the name of a business when that is what the
            # caller asked for by name. See `pipeline._what_the_message_is_about`.
            "subject": subject or lead.get("service") or None,
        },
        "vendors": [
            {
                "title": v.get("title"),
                "phone": v.get("phone"),
                "category": v.get("category"),
                "city": v.get("city"),
            }
            for v in vendors if v.get("title") and v.get("phone")
        ],
        # Per-field scores, not one number. A lead can be perfectly confident
        # about the service and unsure of the name, and a receiver routing on
        # the average would never know which.
        "confidence": confidence,
        "review": {
            "needs_review": bool(lead.get("needs_review")),
            "reason": lead.get("review_reason") or None,
        },
        "call": {
            "status": lead.get("status") or None,
            # True only if the details were read back and the caller agreed.
            # None means it never happened, which is not the same as disagreeing.
            "confirmed": lead.get("confirmed"),
            "started_at": _iso(lead.get("started_at")),
            "ended_at": _iso(lead.get("ended_at")),
        },
    }


def _iso(value) -> str | None:
    return value.isoformat() if hasattr(value, "isoformat") else (value or None)


def sign(body: bytes, timestamp: str) -> str:
    """`v1=<hex>` over "<timestamp>.<body>", HMAC-SHA256.

    The timestamp is inside the signed string rather than merely alongside it,
    so a captured request cannot be replayed later with a fresh header — the
    signature would no longer match. Receivers should reject anything whose
    timestamp is more than a few minutes old.
    """
    mac = hmac.new(
        settings.webhook_secret.encode(),
        f"{timestamp}.".encode() + body,
        hashlib.sha256,
    )
    return "v1=" + mac.hexdigest()


async def _post(payload: dict, attempts: int) -> dict:
    """POST, retrying. Metered as one delivery, not one per attempt.

    The unit is a lead delivered, so a run of failures is zero and a success is
    one — which is why the quantity is set at the outcome rather than in the
    loop. The attempts are not lost: latency on the row shows a send that took
    three tries, and a failed one carries its last error.
    """
    with meter.metered("webhook", "deliveries", operation="handoff") as measured:
        result = await _attempt(payload, attempts)
        if result.get("ok"):
            measured.succeeded(1)
        else:
            measured.failed(0, str(result.get("error") or "")[:300])
        return result


async def _attempt(payload: dict, attempts: int) -> dict:
    # Serialised once, outside the loop: the signature covers these exact
    # bytes, and re-encoding per attempt risks signing something subtly
    # different from what gets sent.
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    err = ""
    for attempt in range(1, attempts + 1):
        timestamp = str(int(time.time()))
        headers = {
            "Content-Type": "application/json",
            VERSION_HEADER: PAYLOAD_VERSION,
            TIMESTAMP_HEADER: timestamp,
            SIGNATURE_HEADER: sign(body, timestamp),
        }
        try:
            async with httpx.AsyncClient(timeout=settings.webhook_timeout) as client:
                res = await client.post(settings.webhook_url, content=body, headers=headers)
            if res.status_code < 300:
                logger.info("delivered %s to the webhook: HTTP %d (attempt %d)",
                            str(payload.get("call_id"))[:8], res.status_code, attempt)
                return {"ok": True, "status": res.status_code}
            err = f"HTTP {res.status_code}: {res.text[:200]!r}"
            logger.warning("webhook rejected (attempt %d/%d): %s", attempt, attempts, err)
            # 4xx is a bad body or a dead secret. Retrying cannot fix either,
            # and 25 sweeps of the same rejection is 25 identical log lines.
            # 429 is the exception: it is the receiver asking for later, and
            # later is exactly what the sweep provides.
            if res.status_code < 500 and res.status_code != 429:
                return {"ok": False, "error": err, "status": res.status_code}
        except Exception as exc:  # noqa: BLE001 - network/timeout is worth a retry
            err = repr(exc)
            logger.warning("webhook failed (attempt %d/%d): %s", attempt, attempts, err)
        if attempt < attempts:
            await asyncio.sleep(1.5 * attempt)
    return {"ok": False, "error": err}


async def send(lead: dict, *, subject: str = "", vendors: list | None = None,
               attempts: int = 3) -> dict:
    """Deliver one lead. Returns a result dict; never raises.

    `attempts` is 1 from the periodic sweep: the sweep *is* the retry, so
    three tries with backoff on every lead only makes each pass longer while a
    receiver is down — 29 owed leads at three attempts each is minutes of work
    to learn one fact.
    """
    # Guarded here rather than at the call site, because there is no second
    # entry point to hold the guard. Without it an unconfigured deployment
    # POSTs to an empty URL on every completed call, three times with backoff,
    # then leaves the lead pending for the sweep to retry twenty-five more
    # times — all to rediscover that no webhook is configured.
    if not settings.webhook_available:
        logger.info(
            "webhook not configured (url=%s secret=%s) — nothing delivered; "
            "the lead is still saved",
            bool(settings.webhook_url), bool(settings.webhook_secret),
        )
        return {"ok": False, "skipped": True, "reason": "not configured"}

    payload = build_payload(lead, subject=subject, vendors=vendors)
    if payload is None:
        logger.info("no caller phone on this lead — nothing delivered")
        return {"ok": False, "skipped": True, "reason": "no phone"}
    return await _post(payload, attempts=attempts)

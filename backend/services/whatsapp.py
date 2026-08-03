"""WhatsApp lead handoff via CampaignBot.

Ported from the Vaani bridge (`bridge/whatsapp.py`), which already runs this
against the live provider. Kept deliberately close to that implementation —
the retry rules and the success test below encode provider behaviour that was
learned the hard way, and re-deriving them here would just re-learn it.

Fires once a lead is saved. Best-effort by construction: a down, throttled, or
misconfigured provider must never fail the call or lose the lead, which is
already on disk before this runs.

Configuration, all via env so a template or key change is not a code change:

  WHATSAPP_ENABLED        "1" to actually send (default off until configured)
  WHATSAPP_API_KEY        CampaignBot Bearer token
  WHATSAPP_TEMPLATE_NAME  registered template id, e.g. we_found_these_for_you
  WHATSAPP_LANG_CODE      template language code (default en_US)
  WHATSAPP_PARAM4         text for {{4}} when there are no live matches yet
  WHATSAPP_API_URL        endpoint override

Template body → positional params:
  {{1}} name · {{2}} service · {{3}} location · {{4}} options line
"""

from __future__ import annotations

import asyncio

import httpx

from ..config import settings
from ..logger import get_logger
from . import meter

logger = get_logger("localmama.whatsapp")


def _norm_phone(raw: str) -> str:
    """Digits to E.164-ish.

    Strips the national trunk "0" — Indian callers arrive as "09739960092", and
    prepending a country code without stripping it yields "+0…", which the
    provider accepts and never delivers.
    """
    digits = "".join(ch for ch in str(raw or "") if ch.isdigit())
    if not digits:
        return ""
    digits = digits.lstrip("0")
    if len(digits) == 10:          # bare Indian mobile
        digits = "91" + digits
    return "+" + digits


def build_payload(
    phone: str,
    *,
    name: str | None,
    service: str | None,
    city: str | None,
    options: str = "",
) -> dict | None:
    """Map a processed lead onto the CampaignBot template request.

    Plain values rather than a lead object: this is the last step of the
    pipeline and it needs four strings. Taking a model here only couples the
    sender to whatever shape a lead happens to have this month.

    None means there is no usable recipient, which is not an error — a browser
    or WebRTC caller is anonymous, and a phone number only exists once the call
    arrives over SIP.
    """
    number = _norm_phone(phone)
    if not number:
        return None
    name = (name or "there").strip() or "there"
    service = (service or "the service").strip() or "the service"
    # The Vaani sender matches on city and ignores the locality; this field is
    # whichever of the two the caller gave, so it is passed through as-is.
    location = (city or "").strip() or "your area"
    opts = (options or settings.whatsapp_param4).strip() or settings.whatsapp_param4
    return {
        "recipientPhone": number,
        "recipientName": name,
        "messageType": "template",
        "templateName": settings.whatsapp_template_name,
        "languageCode": settings.whatsapp_lang_code,
        "templateParams": [name, service, location, opts],   # {{1}}..{{4}}
    }


def _ok(status_code: int, body: dict) -> bool:
    """CampaignBot signals failure two ways: a 4xx status, or HTTP 200 with a
    body carrying "status": false / statusCode >= 400. Success only when the
    body agrees with the status line."""
    if status_code >= 400:
        return False
    if body.get("success") is True or body.get("status") is True:
        return True
    if body.get("status") is False:
        return False
    inner = (body.get("data") or {}).get("statusCode") or body.get("statusCode")
    return inner is None or int(inner) < 400


def _message_id(body: dict) -> str:
    for holder in ((body.get("data") or {}), body):
        mid = (holder.get("payload") or {}).get("messageId")
        if mid:
            return mid
    return ""


async def _post(payload: dict, attempts: int = 3) -> dict:
    """Send, retrying. Metered as one billable message, not one per attempt.

    The provider charges for a conversation that was delivered, so a run of
    failed attempts costs nothing and a success costs one — which is why the
    quantity is set at the outcome rather than in the loop. The attempt count
    is not lost: a send that took three tries shows up in the latency on the
    row, and a send that never worked shows up as not-ok with the last error.
    """
    headers = {
        "Authorization": f"Bearer {settings.whatsapp_api_key}",
        "Content-Type": "application/json",
    }
    with meter.metered("campaignbot", "messages", operation="handoff") as measured:
        result = await _attempt(payload, headers, attempts)
        if result.get("ok"):
            measured.succeeded(1)
        else:
            measured.failed(0, str(result.get("error") or "")[:300])
        return result


async def _attempt(payload: dict, headers: dict, attempts: int) -> dict:
    err = ""
    for attempt in range(1, attempts + 1):
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                res = await client.post(settings.whatsapp_api_url, json=payload, headers=headers)
            try:
                body = res.json()
            except ValueError:
                body = {}
            if _ok(res.status_code, body):
                logger.info(
                    "sent template=%s to %s id=%s (attempt %d)",
                    payload["templateName"], payload["recipientPhone"],
                    _message_id(body), attempt,
                )
                return {"ok": True, "messageId": _message_id(body)}
            err = f"HTTP {res.status_code}: {body or res.text[:200]!r}"
            logger.warning("send rejected (attempt %d/%d): %s", attempt, attempts, err)
            # A 4xx is a bad payload or a dead token; retrying cannot fix it.
            if res.status_code < 500:
                return {"ok": False, "error": err}
        except Exception as exc:  # noqa: BLE001 - network/timeout is worth a retry
            err = repr(exc)
            logger.warning("send failed (attempt %d/%d): %s", attempt, attempts, err)
        if attempt < attempts:
            await asyncio.sleep(1.5 * attempt)
    return {"ok": False, "error": err}


async def send(
    phone: str,
    *,
    name: str | None = None,
    service: str | None = None,
    city: str | None = None,
    options: str = "",
    attempts: int = 3,
) -> dict:
    """Send synchronously. Returns a result dict; never raises.

    `attempts` is 1 for the periodic sweep: the sweep *is* the retry, so trying
    three times with backoff on every lead only makes each pass longer while a
    provider is down — 29 owed leads at three attempts each is minutes of work
    to learn one fact.
    """
    # Checked here rather than at the call site, because there is no longer a
    # second entry point that could hold the guard. Without it an unconfigured
    # deployment POSTs an empty bearer token on every completed call, three
    # times with backoff, and then leaves the lead `pending` for the sweep to
    # retry twenty-five more times — all to rediscover that WhatsApp is off.
    if not settings.whatsapp_available:
        logger.info(
            "WhatsApp not configured (enabled=%s key=%s template=%s) — nothing "
            "sent; the lead is still saved",
            settings.whatsapp_enabled,
            bool(settings.whatsapp_api_key),
            bool(settings.whatsapp_template_name),
        )
        return {"ok": False, "skipped": True, "reason": "not configured"}
    payload = build_payload(
        phone, name=name, service=service, city=city, options=options
    )
    if payload is None:
        logger.info("no recipient phone on this lead — nothing sent")
        return {"ok": False, "skipped": True, "reason": "no phone"}
    return await _post(payload, attempts=attempts)

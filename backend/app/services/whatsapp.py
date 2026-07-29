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
from ..models import Lead

logger = get_logger("localmama.whatsapp")

#: Strong references to in-flight sends. Without these asyncio may collect a
#: task mid-flight and the message silently never goes out — the same failure
#: that swallowed whole turns in agent.py.
_tasks: set[asyncio.Task] = set()


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


def build_payload(lead: Lead, phone: str, options: str = "") -> dict | None:
    """Map a saved lead onto the CampaignBot template request.

    None means there is no usable recipient, which is the common case here:
    browser and WebRTC callers are anonymous. A phone number only exists once
    the call arrives over SIP, or the caller is asked for one.
    """
    number = _norm_phone(phone)
    if not number:
        return None
    name = (lead.user_name or "there").strip() or "there"
    service = (lead.requested_service or "the service").strip() or "the service"
    # The Vaani sender matches on city and ignores the locality; this field is
    # whichever of the two the caller gave, so it is passed through as-is.
    location = (lead.city_or_area or "").strip() or "your area"
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
    headers = {
        "Authorization": f"Bearer {settings.whatsapp_api_key}",
        "Content-Type": "application/json",
    }
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


async def send(lead: Lead, phone: str, options: str = "", attempts: int = 3) -> dict:
    """Send synchronously. Returns a result dict; never raises.

    `attempts` is 1 for the periodic sweep: the sweep *is* the retry, so trying
    three times with backoff on every lead only makes each pass longer while a
    provider is down — 29 owed leads at three attempts each is minutes of work
    to learn one fact.
    """
    payload = build_payload(lead, phone, options)
    if payload is None:
        logger.info("no recipient phone on this lead — nothing sent")
        return {"ok": False, "skipped": True, "reason": "no phone"}
    return await _post(payload, attempts=attempts)


def fire(lead: Lead, phone: str, options: str = "") -> None:
    """Fire-and-forget. Never blocks the call and never raises into it."""
    if not settings.whatsapp_available:
        logger.info(
            "WhatsApp not sent (enabled=%s key=%s template=%s) — lead is still "
            "saved to disk",
            settings.whatsapp_enabled,
            bool(settings.whatsapp_api_key),
            bool(settings.whatsapp_template_name),
        )
        return
    try:
        task = asyncio.create_task(send(lead, phone, options))
    except RuntimeError:
        # No running loop: the CLI and tests call this synchronously.
        logger.info("no event loop; WhatsApp send skipped")
        return
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)

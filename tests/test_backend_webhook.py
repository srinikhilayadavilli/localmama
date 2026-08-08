"""What goes over the webhook, what is left out, and what gets retried.

The sender itself, rather than the pipeline's decision to call it — see
`test_backend_notify.py` for when a lead is handed off at all.
"""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest

from backend.config import settings as backend_settings
from backend.services import webhook


@pytest.fixture()
def configured():
    """A URL and a secret, restored afterwards. `settings` is frozen."""
    old = (backend_settings.webhook_url, backend_settings.webhook_secret)
    object.__setattr__(backend_settings, "webhook_url", "https://example.test/hook")
    object.__setattr__(backend_settings, "webhook_secret", "shhh")
    yield
    object.__setattr__(backend_settings, "webhook_url", old[0])
    object.__setattr__(backend_settings, "webhook_secret", old[1])


def _lead(**over) -> dict:
    base = {
        "call_id": "c1", "caller_phone": "09739960092", "language": "telugu",
        "dialled": "+918071581496", "name": "Ravi", "service": "plumber",
        "service_said": "plumber", "service_inferred": None, "city": "Hyderabad",
        "status": "completed", "confirmed": True, "needs_review": False,
        "review_reason": None, "confidence": {"name": 0.9, "service": 1.0},
        "vendors": [{"title": "Infinity", "phone": "+919876543210",
                     "category": "Plumbing", "city": "Hyderabad"}],
        "transcript": "my name is Ravi ... I need a plumber",
        "whatsapp_status": "sent", "whatsapp_message_id": "wamid.xyz",
    }
    base.update(over)
    return base


# --- what the body carries -------------------------------------------------


def test_the_payload_carries_the_lead():
    body = webhook.build_payload(_lead())
    assert body["lead"]["name"] == "Ravi"
    assert body["lead"]["service"] == "plumber"
    assert body["lead"]["city"] == "Hyderabad"
    assert body["vendors"][0]["phone"] == "+919876543210"
    assert body["confidence"]["service"] == 1.0
    assert body["call"]["confirmed"] is True


def test_the_payload_carries_no_whatsapp_information():
    """The channel is gone. Its columns are the record of messages sent to real
    people during the WhatsApp era, and they mean nothing to a receiver."""
    flat = json.dumps(webhook.build_payload(_lead())).lower()

    assert "whatsapp" not in flat
    assert "wamid" not in flat


def test_the_payload_carries_no_transcript():
    """Everything the caller said, which is personal data under the DPDP Act.
    It stays here under a retention sweep rather than being copied to a third
    party with its own retention rules and its own breaches."""
    body = webhook.build_payload(_lead())

    assert "transcript" not in json.dumps(body)
    assert "I need a plumber" not in json.dumps(body)


def test_the_caller_number_is_dialable():
    """"09739960092" with a country code bolted on is "+09739960092", which the
    old provider accepted and never delivered. The receiver is going to ring
    this number."""
    body = webhook.build_payload(_lead(caller_phone="09739960092"))

    assert body["caller"]["phone"] == "+919739960092"


def test_an_anonymous_caller_is_not_a_failure():
    """A browser or WebRTC caller has no number. The lead is still worth
    storing; it is just not one worth delivering."""
    assert webhook.build_payload(_lead(caller_phone="")) is None


def test_a_vendor_without_a_number_is_left_out():
    """A name without a number is a teaser, not an answer."""
    body = webhook.build_payload(_lead(
        vendors=[{"title": "Brimmies Cafe", "phone": None},
                 {"title": "Infinity", "phone": "+91987"}],
    ))

    assert [v["title"] for v in body["vendors"]] == ["Infinity"]


def test_all_three_readings_of_the_service_survive():
    """They differ, and the difference is the whole point: what they said, what
    it normalised to, and what the agent understood when their words matched
    nothing."""
    body = webhook.build_payload(_lead(
        service="plumber", service_said="my geyser is not working",
        service_inferred="plumber",
    ))

    assert body["lead"]["service"] == "plumber"
    assert body["lead"]["service_said"] == "my geyser is not working"
    assert body["lead"]["service_inferred"] == "plumber"


# --- signing ---------------------------------------------------------------


def test_the_signature_covers_the_timestamp_and_the_body(configured):
    """The timestamp is inside the signed string, not merely beside it, so a
    captured request cannot be replayed later under a fresh header."""
    body = b'{"a":1}'
    expected = hmac.new(b"shhh", b"1700000000." + body, hashlib.sha256).hexdigest()

    assert webhook.sign(body, "1700000000", "shhh") == "v1=" + expected


def test_a_different_body_does_not_verify(configured):
    assert webhook.sign(b'{"a":1}', "1", "s") != webhook.sign(b'{"a":2}', "1", "s")


def test_a_different_timestamp_does_not_verify(configured):
    assert webhook.sign(b'{"a":1}', "1", "s") != webhook.sign(b'{"a":1}', "2", "s")


# --- when it sends, and when it gives up -----------------------------------


@pytest.mark.asyncio
async def test_nothing_is_sent_without_a_url_or_secret():
    """Guarded in the sender because there is no second entry point to hold the
    guard. Without it an unconfigured deployment POSTs to an empty URL on every
    call, three times with backoff, then leaves the lead for 25 more sweeps."""
    result = await webhook.send(_lead())

    assert result["ok"] is False
    assert result["skipped"] is True


@pytest.mark.asyncio
async def test_a_4xx_is_not_retried(configured, monkeypatch):
    """A bad body or a dead secret. Retrying cannot fix either, and 25 sweeps
    of the same rejection is 25 identical log lines."""
    attempts = []

    async def _post(self, url, content=None, headers=None):
        attempts.append(url)
        return _Response(400, "bad signature")

    monkeypatch.setattr("httpx.AsyncClient.post", _post)
    result = await webhook.send(_lead(), attempts=3)

    assert len(attempts) == 1
    assert result["ok"] is False
    assert result["status"] == 400


@pytest.mark.asyncio
async def test_a_429_is_retried(configured, monkeypatch):
    """Not a refusal — the receiver asking for later, and later is exactly what
    the retry provides."""
    attempts = []

    async def _post(self, url, content=None, headers=None):
        attempts.append(url)
        return _Response(429, "slow down")

    monkeypatch.setattr("httpx.AsyncClient.post", _post)
    monkeypatch.setattr("asyncio.sleep", _no_sleep)
    await webhook.send(_lead(), attempts=3)

    assert len(attempts) == 3


@pytest.mark.asyncio
async def test_a_5xx_is_retried_then_reported(configured, monkeypatch):
    attempts = []

    async def _post(self, url, content=None, headers=None):
        attempts.append(url)
        return _Response(503, "down")

    monkeypatch.setattr("httpx.AsyncClient.post", _post)
    monkeypatch.setattr("asyncio.sleep", _no_sleep)
    result = await webhook.send(_lead(), attempts=3)

    assert len(attempts) == 3
    assert result["ok"] is False


@pytest.mark.asyncio
async def test_a_2xx_is_a_delivery(configured, monkeypatch):
    sent = {}

    async def _post(self, url, content=None, headers=None):
        sent["headers"] = headers
        sent["body"] = content
        return _Response(202, "queued")

    monkeypatch.setattr("httpx.AsyncClient.post", _post)
    result = await webhook.send(_lead())

    assert result == {"ok": True, "status": 202}
    # The signature must verify against the exact bytes that went out, not a
    # re-encoding of the same object.
    ts = sent["headers"][webhook.TIMESTAMP_HEADER]
    assert sent["headers"][webhook.SIGNATURE_HEADER] == webhook.sign(
        sent["body"], ts, "shhh")


class _Response:
    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text


async def _no_sleep(_seconds):
    return None


# --- what the review found -------------------------------------------------


@pytest.mark.asyncio
async def test_a_4xx_is_terminal_not_merely_failed(configured, monkeypatch):
    """It was returned as a plain failure, which `mark_handoff` mapped to
    `pending` because the error text was not in `_NOT_WORTH_RETRYING` — so a
    permanently rejected body was re-POSTed every five minutes for 25 attempts
    and then dropped past the ceiling."""
    async def _post(self, url, content=None, headers=None):
        return _Response(400, "bad signature")

    monkeypatch.setattr("httpx.AsyncClient.post", _post)
    result = await webhook.send(_lead(), attempts=3)

    assert result["terminal"] is True


@pytest.mark.asyncio
async def test_a_429_is_not_terminal(configured, monkeypatch):
    """Rate limiting is the receiver asking for later, not refusing the lead."""
    async def _post(self, url, content=None, headers=None):
        return _Response(429, "slow down")

    monkeypatch.setattr("httpx.AsyncClient.post", _post)
    monkeypatch.setattr("asyncio.sleep", _no_sleep)
    result = await webhook.send(_lead(), attempts=2)

    assert not result.get("terminal")


@pytest.mark.asyncio
async def test_a_5xx_is_not_terminal(configured, monkeypatch):
    async def _post(self, url, content=None, headers=None):
        return _Response(503, "down")

    monkeypatch.setattr("httpx.AsyncClient.post", _post)
    monkeypatch.setattr("asyncio.sleep", _no_sleep)
    result = await webhook.send(_lead(), attempts=2)

    assert not result.get("terminal")


@pytest.mark.asyncio
async def test_redirects_are_followed(configured, monkeypatch):
    """A receiver mounted at `/hook` configured here as `/hook/` answers 307 —
    the FastAPI/Flask/Django default. Unfollowed, that is a permanent silent
    non-delivery caused by a trailing slash."""
    import httpx

    seen = {}
    real_init = httpx.AsyncClient.__init__

    def _init(self, *a, **kw):          # __init__ is sync, not a coroutine
        seen.update(kw)
        real_init(self, *a, **kw)

    async def _post(self, url, content=None, headers=None):
        return _Response(200, "ok")

    monkeypatch.setattr(httpx.AsyncClient, "__init__", _init)
    monkeypatch.setattr(httpx.AsyncClient, "post", _post)
    await webhook.send(_lead())

    assert seen.get("follow_redirects") is True

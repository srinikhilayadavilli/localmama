"""Where a lead is delivered, and who decides.

The customer's subscription wins over the environment. The environment is what
keeps a deployment with an empty table behaving as it did before the table
existed — not a second source of truth to reconcile.
"""

from __future__ import annotations

import pytest

from backend.config import settings as backend_settings
from backend.services import webhook


@pytest.fixture()
def env_fallback():
    """A URL and secret in the environment, restored afterwards."""
    old = (backend_settings.webhook_url, backend_settings.webhook_secret)
    object.__setattr__(backend_settings, "webhook_url", "https://env.test/hook")
    object.__setattr__(backend_settings, "webhook_secret", "env-secret")
    yield
    object.__setattr__(backend_settings, "webhook_url", old[0])
    object.__setattr__(backend_settings, "webhook_secret", old[1])


@pytest.fixture()
def subscription(monkeypatch):
    """Control what the table returns."""
    box = {"row": None}
    monkeypatch.setattr("backend.store.active_webhook",
                        lambda agent_id=None: box["row"])
    return box


def test_the_customers_subscription_wins(subscription, env_fallback):
    """The point of the table is that a business can change where its leads go
    without waiting on our deploy queue."""
    subscription["row"] = {"id": "sub_1", "url": "https://theirs.test/hook",
                           "secret": "theirs"}

    assert webhook.destination()["url"] == "https://theirs.test/hook"
    assert webhook.destination()["secret"] == "theirs"


def test_the_environment_is_the_fallback(subscription, env_fallback):
    """An empty table behaves exactly as the deployment did before it existed."""
    subscription["row"] = None

    assert webhook.destination()["url"] == "https://env.test/hook"
    assert webhook.destination()["id"] == "env"


def test_nothing_configured_is_no_destination(subscription):
    """Not an error — a state of the deployment. The sweep reads this to decide
    whether to claim, because claiming for a channel that cannot send still
    burns the lead's attempt budget."""
    subscription["row"] = None

    assert webhook.destination() is None
    assert webhook.configured() is False


def test_a_subscription_without_a_secret_is_refused(subscription, env_fallback):
    """The dashboard offers the secret as optional. It must never mean
    unsigned: this payload carries a caller's name and phone number, and an
    endpoint that cannot tell us from anyone else believes whoever finds the
    URL. The table mints one by default; if a row somehow arrives without,
    it is ignored rather than sent to in the clear."""
    subscription["row"] = {"id": "sub_1", "url": "https://theirs.test/hook",
                           "secret": ""}

    # Falls back rather than sending unsigned.
    assert webhook.destination()["id"] == "env"


def test_a_url_without_a_secret_in_the_environment_is_refused(subscription):
    subscription["row"] = None
    old = (backend_settings.webhook_url, backend_settings.webhook_secret)
    object.__setattr__(backend_settings, "webhook_url", "https://env.test/hook")
    object.__setattr__(backend_settings, "webhook_secret", "")
    try:
        assert webhook.destination() is None
    finally:
        object.__setattr__(backend_settings, "webhook_url", old[0])
        object.__setattr__(backend_settings, "webhook_secret", old[1])


@pytest.mark.asyncio
async def test_the_delivery_is_signed_with_the_subscriptions_secret(
    subscription, env_fallback, monkeypatch
):
    """Not the environment's. A customer who rotates their secret must have the
    next delivery signed with the new one."""
    subscription["row"] = {"id": "sub_1", "url": "https://theirs.test/hook",
                           "secret": "theirs"}
    sent = {}

    async def _post(self, url, content=None, headers=None):
        sent["url"] = url
        sent["headers"] = headers
        sent["body"] = content
        return _Response(200, "ok")

    monkeypatch.setattr("httpx.AsyncClient.post", _post)
    await webhook.send({"call_id": "c1", "caller_phone": "+919739960092"})

    assert sent["url"] == "https://theirs.test/hook"
    ts = sent["headers"][webhook.TIMESTAMP_HEADER]
    assert sent["headers"][webhook.SIGNATURE_HEADER] == webhook.sign(
        sent["body"], ts, "theirs")
    # …and emphatically not with the environment's key.
    assert sent["headers"][webhook.SIGNATURE_HEADER] != webhook.sign(
        sent["body"], ts, "env-secret")


class _Response:
    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text

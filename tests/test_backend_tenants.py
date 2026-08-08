"""Which business a call belongs to, and how the system works it out.

`agent_id` was on every lead from the first migration and was always the same
string, because it came from an environment variable. The routing key was being
captured and discarded: `dialled`, the number the caller rang, documented in
the contract as "for routing and attribution". These tests are that promise
being kept.
"""

from __future__ import annotations

import pytest

from backend import store
from backend.config import settings as backend_settings
from backend.services import webhook


# --- resolving a tenant from the number ------------------------------------


def test_an_unmapped_number_falls_back_to_the_default_tenant(monkeypatch):
    """A call on a number nobody has claimed is still a lead worth keeping. It
    lands with the deployment's default rather than being dropped."""
    monkeypatch.setattr(store.db, "available", lambda: False)

    assert store.agent_for_did("+911111111111") == backend_settings.brain_agent_id


def test_no_dialled_number_falls_back_too(monkeypatch):
    """A WebRTC or browser caller arrives without one."""
    monkeypatch.setattr(store.db, "available", lambda: False)

    assert store.agent_for_did(None) == backend_settings.brain_agent_id


def test_a_database_that_is_down_does_not_lose_the_lead(monkeypatch):
    """Routing is worth less than the lead. An unreachable database degrades to
    the default tenant rather than raising into the event handler."""
    def _boom():
        raise RuntimeError("neon is unreachable")

    monkeypatch.setattr(store.db, "available", lambda: True)
    monkeypatch.setattr(store.db, "cursor", _boom)

    assert store.agent_for_did("+918071581496") == backend_settings.brain_agent_id


# --- the tenant follows the lead through delivery --------------------------


def test_the_destination_is_looked_up_for_that_tenant(monkeypatch):
    """Two businesses, two endpoints. The lead's own `agent_id` picks."""
    rows = {
        "acme": {"id": "sub_a", "url": "https://acme.test/hook", "secret": "a"},
        "sri-sai": {"id": "sub_s", "url": "https://sri-sai.test/hook", "secret": "s"},
    }
    monkeypatch.setattr(store, "active_webhook", lambda agent_id=None: rows.get(agent_id))

    assert webhook.destination("acme")["url"] == "https://acme.test/hook"
    assert webhook.destination("sri-sai")["url"] == "https://sri-sai.test/hook"
    assert webhook.destination("nobody") is None


@pytest.mark.asyncio
async def test_a_lead_is_delivered_to_its_own_tenants_endpoint(monkeypatch):
    """The failure this prevents is the worst kind: one business's customer
    details posted to another business's server."""
    rows = {
        "acme": {"id": "sub_a", "url": "https://acme.test/hook", "secret": "a"},
        "sri-sai": {"id": "sub_s", "url": "https://sri-sai.test/hook", "secret": "s"},
    }
    monkeypatch.setattr(store, "active_webhook", lambda agent_id=None: rows.get(agent_id))
    sent = []

    async def _post(self, url, content=None, headers=None):
        sent.append(url)
        return _Response(200, "ok")

    monkeypatch.setattr("httpx.AsyncClient.post", _post)

    await webhook.send({"call_id": "c1", "agent_id": "sri-sai",
                        "caller_phone": "+919739960092"})

    assert sent == ["https://sri-sai.test/hook"]


@pytest.mark.asyncio
async def test_a_tenant_with_no_endpoint_delivers_nowhere(monkeypatch):
    """Not an error, and not another tenant's endpoint either."""
    monkeypatch.setattr(store, "active_webhook", lambda agent_id=None: None)
    old = (backend_settings.webhook_url, backend_settings.webhook_secret)
    object.__setattr__(backend_settings, "webhook_url", "")
    object.__setattr__(backend_settings, "webhook_secret", "")
    try:
        result = await webhook.send({"call_id": "c1", "agent_id": "acme",
                                     "caller_phone": "+919739960092"})
        assert result["skipped"] is True
    finally:
        object.__setattr__(backend_settings, "webhook_url", old[0])
        object.__setattr__(backend_settings, "webhook_secret", old[1])


# --- what the sweep is allowed to claim ------------------------------------


def test_the_sweep_only_claims_for_tenants_that_can_receive(monkeypatch):
    """Claiming a lead for a tenant with no endpoint still increments its
    attempt count, and `OUTBOX_MAX_ATTEMPTS` is a permanent write-off."""
    from backend import outbox_worker

    monkeypatch.setattr(store, "active_agents", lambda: ["acme"])
    object.__setattr__(backend_settings, "webhook_url", "")
    object.__setattr__(backend_settings, "webhook_secret", "")

    claimed = []

    async def _drain_channel(channel, limit=50, agents=None):
        claimed.append((channel, agents))
        return 0, 0

    monkeypatch.setattr(outbox_worker, "_drain_channel", _drain_channel)
    monkeypatch.setattr(outbox_worker.webhook, "configured", lambda agent_id=None: True)
    monkeypatch.setattr(backend_settings.__class__, "whatsapp_available",
                        property(lambda self: True))

    import asyncio
    asyncio.run(outbox_worker.drain())

    by_channel = dict(claimed)
    assert by_channel["webhook"] == ["acme"]   # restricted
    assert by_channel["whatsapp"] is None      # every tenant


class _Response:
    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text

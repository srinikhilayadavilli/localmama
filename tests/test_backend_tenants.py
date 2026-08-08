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
    class _Cur:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, *a): pass
        def fetchone(self): return None

    monkeypatch.setattr(store.db, "available", lambda: True)
    monkeypatch.setattr(store.db, "cursor", lambda: _Cur())

    assert store.agent_for_did("+911111111111") == backend_settings.brain_agent_id


def test_no_dialled_number_falls_back_too(monkeypatch):
    """A WebRTC or browser caller arrives without one."""
    monkeypatch.setattr(store.db, "available", lambda: False)

    assert store.agent_for_did(None) == backend_settings.brain_agent_id


def test_an_unreachable_database_says_unknown_rather_than_guessing(monkeypatch):
    """The distinction that matters. "Nobody owns this number" is an answer;
    "the database blinked" is not, and returning the default for the second one
    wrote a guess that nothing ever revisited — `call.started` is deduplicated
    by `event_id`, so one blip misfiled a lead under the wrong business for
    good. None lets the caller leave the column alone and resolve again."""
    def _boom():
        raise RuntimeError("neon is unreachable")

    monkeypatch.setattr(store.db, "available", lambda: True)
    monkeypatch.setattr(store.db, "cursor", _boom)

    assert store.agent_for_did("+918071581496") is None


def test_no_database_at_all_says_unknown_too(monkeypatch):
    monkeypatch.setattr(store.db, "available", lambda: False)

    assert store.agent_for_did("+918071581496") is None


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


# --- what the review caught -------------------------------------------------


def test_get_lead_carries_the_tenant():
    """The bug this file was supposed to prevent and did not.

    `webhook.send` resolves the tenant from `lead["agent_id"]`, and the live
    path hands it whatever `store.get_lead` returns. That SELECT omitted
    `agent_id`, so every real-time delivery resolved to the default tenant's
    subscription — while the tests passed, because they hand-built a lead dict
    with `agent_id` already in it. Assert the contract at the boundary that
    actually carries it.
    """
    import inspect

    src = inspect.getsource(store.get_lead)
    assert "agent_id" in src.split("cols =")[0], "the SELECT must fetch agent_id"
    assert '"agent_id"' in src.split("cols =")[1], "and the column list must name it"


@pytest.mark.asyncio
async def test_the_live_path_delivers_to_the_leads_own_tenant(monkeypatch):
    """End to end through `pipeline.notify`, not through a dict I wrote."""
    from backend import pipeline

    stored = {
        "call_id": "c1", "agent_id": "sri-sai", "caller_phone": "+919739960092",
        "name": "Ravi", "city": "Hyderabad", "service": "plumber",
    }
    rows = {
        "localmama": {"id": "sub_d", "url": "https://default.test/hook", "secret": "d"},
        "sri-sai": {"id": "sub_s", "url": "https://sri-sai.test/hook", "secret": "s"},
    }
    sent = []

    class _Store:
        def get_lead(self, call_id):
            return dict(stored)

        def mark_handoff(self, *a, **k):
            pass

        def active_webhook(self, agent_id=None):
            return rows.get(agent_id)

    monkeypatch.setattr(pipeline, "store", _Store())
    monkeypatch.setattr(store, "active_webhook", _Store().active_webhook)

    async def _wa(*a, **k):
        return {"ok": True, "messageId": "m"}

    async def _post(self, url, content=None, headers=None):
        sent.append(url)
        return _Response(200, "ok")

    monkeypatch.setattr(pipeline.whatsapp, "send", _wa)
    monkeypatch.setattr("httpx.AsyncClient.post", _post)

    await pipeline.notify("c1", "+919739960092", [], "plumber")

    assert sent == ["https://sri-sai.test/hook"], (
        "the live path delivered to the wrong tenant")


def test_the_environment_fallback_belongs_to_the_default_tenant_only(monkeypatch):
    """A tenant mid-onboarding has no subscription row. Falling back to the
    environment sent their callers' names and numbers to whichever customer the
    single-tenant deployment was configured for — signed with a secret that
    receiver already holds, and marked sent so no retry corrects it."""
    monkeypatch.setattr(store, "active_webhook", lambda agent_id=None: None)
    old = (backend_settings.webhook_url, backend_settings.webhook_secret)
    object.__setattr__(backend_settings, "webhook_url", "https://default.test/hook")
    object.__setattr__(backend_settings, "webhook_secret", "d")
    try:
        assert webhook.destination("localmama")["id"] == "env"   # the default: yes
        assert webhook.destination("acme") is None               # anyone else: no
    finally:
        object.__setattr__(backend_settings, "webhook_url", old[0])
        object.__setattr__(backend_settings, "webhook_secret", old[1])


def test_a_did_is_matched_on_its_digits(monkeypatch):
    """`dialled` is whatever the trunk sent. Indian SIP providers present the
    same number as +91…, 91…, 0… and sip:+91…@host, and an exact string match
    files every call on that tenant's number under the default tenant."""
    assert store._digits("+918071581496") == "918071581496"
    assert store._digits("918071581496") == "918071581496"
    # The national form: strip the trunk 0, then restore the country code —
    # the same rule the senders apply to a caller's number.
    assert store._digits("08071581496") == "918071581496"
    assert store._digits("sip:+918071581496@trunk.example") == "918071581496"

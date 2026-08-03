"""The operator surface, and the backend-side meter that feeds it.

Two things worth holding down. The ops endpoints carry the business's unit
economics and must never be reachable without the token — including in the
misconfigured case, where the temptation is to fail open. And the meter must
record a provider call that *failed*, because a table holding only successes
is a table where an outage looks like an idle afternoon.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend import api, ops
from backend.config import settings
from backend.services import meter

OPS_TOKEN = "ops-token"


@pytest.fixture()
def client(monkeypatch):
    object.__setattr__(settings, "ops_token", OPS_TOKEN)
    # No DSN, so every costing read short-circuits to empty — which is exactly
    # what a dashboard should do when the database is away, and is why these
    # tests run in milliseconds rather than blocking on a pool timeout apiece.
    object.__setattr__(settings, "database_url", "")
    with TestClient(api.app) as c:
        yield c


def _auth(token: str = OPS_TOKEN) -> dict:
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.parametrize("path", [
    "/metrics", "/v1/ops/summary", "/v1/ops/rates", "/v1/ops/calls/c1",
])
def test_the_ops_surface_is_closed_without_the_token(client, path):
    assert client.get(path).status_code == 401
    assert client.get(path, headers=_auth("wrong")).status_code == 401


def test_an_unconfigured_token_refuses_rather_than_opens(client):
    """The failure mode that matters: a deployment that forgot to set the token
    must not publish its cost per lead. 503 says "not configured"; 200 would
    say "help yourself"."""
    object.__setattr__(settings, "ops_token", "")
    try:
        assert client.get("/metrics").status_code == 503
    finally:
        object.__setattr__(settings, "ops_token", OPS_TOKEN)


def test_the_agent_token_does_not_open_the_ops_surface(client):
    """Different principals, different secrets. Rotating one must not be a way
    into the other."""
    object.__setattr__(settings, "agent_token", "agent-token")
    assert client.get("/metrics", headers=_auth("agent-token")).status_code == 401


def test_metrics_renders_even_with_no_data(client):
    """A brand-new deployment scrapes cleanly rather than 500ing at the
    monitoring system that is supposed to tell it something is wrong."""
    response = client.get("/metrics", headers=_auth())
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")


def test_a_scrape_with_no_database_answers_promptly(client):
    """A summary is a dozen queries and the pool waits ten seconds before
    giving up, so an unreachable database once turned one scrape into a
    two-minute hang — at the exact moment monitoring is supposed to be saying
    something is wrong. The scraper times out and the outage reads as missing
    data instead."""
    import time as _time

    started = _time.monotonic()
    assert client.get("/metrics", headers=_auth()).status_code == 200
    assert _time.monotonic() - started < 2.0


def test_one_connection_failure_short_circuits_the_rest(monkeypatch):
    """The breaker trips on "no database" and not on a bad query — this
    module's own SQL bug must not blank every other panel on the page."""
    import psycopg

    from backend import costing

    calls = []

    def _boom(exc):
        def _cursor():
            calls.append(1)
            raise exc
        return _cursor

    object.__setattr__(settings, "database_url", "postgresql:///nope")
    costing._breaker_open_until = 0.0

    monkeypatch.setattr(costing.db, "cursor",
                        _boom(psycopg.OperationalError("connection refused")))
    for _ in range(5):
        assert costing._rows("SELECT 1") == []
    assert len(calls) == 1, "an unreachable database should be tried once, not five times"

    # A programming error is different: every call still attempts.
    costing._breaker_open_until = 0.0
    calls.clear()
    monkeypatch.setattr(costing.db, "cursor",
                        _boom(psycopg.ProgrammingError("column does not exist")))
    for _ in range(3):
        assert costing._rows("SELECT bad") == []
    assert len(calls) == 3


def test_every_metric_family_is_one_contiguous_group():
    """Prometheus requires all samples of a family under one HELP/TYPE. Emitting
    two metrics inside the same loop interleaves them, which reads fine and
    which a strict parser rejects outright."""
    exposition = ops._Exposition()
    for provider in ("openai", "livekit"):
        exposition.add("cost", 1.0, provider=provider, help_="spend")
        exposition.add("failures", 0.0, provider=provider, help_="failures")

    seen, current = set(), None
    for line in exposition.render().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name = line.split("{")[0].split(" ")[0]
        if name != current:
            assert name not in seen, f"{name} appears in more than one group"
            seen.add(name)
            current = name
    assert seen == {"cost", "failures"}


def test_a_none_sample_is_omitted_rather_than_zeroed():
    """`costing` reports None for "the denominator was zero". Rendering that as
    0 would turn "we delivered nothing" into "leads are free"."""
    exposition = ops._Exposition()
    exposition.add("cost_per_delivered", None)
    exposition.add("calls", 3)
    assert "cost_per_delivered" not in exposition.render()


def test_label_values_are_escaped():
    exposition = ops._Exposition()
    exposition.add("thing", 1.0, model='weird"name\\here')
    line = exposition.render().splitlines()[-1]
    assert r"weird\"name\\here" in line


# ── the backend meter ───────────────────────────────────────────────────────


@pytest.fixture()
def recorded(monkeypatch):
    """Capture what the meter would have written, without a database."""
    rows = []
    monkeypatch.setattr(meter, "record_service_call",
                        lambda call_id, **kw: rows.append({"call_id": call_id, **kw}))
    return rows


def test_a_metered_call_is_attributed_to_the_call_in_scope(recorded):
    with meter.for_call("call-1"), meter.metered("sarvam", "characters",
                                                 operation="translate") as m:
        m.succeeded(42)
    assert recorded[0]["call_id"] == "call-1"
    assert recorded[0]["quantity"] == 42
    assert recorded[0]["ok"] is True


def test_work_outside_a_call_is_not_attributed(recorded):
    """No call, nothing to bill it to. The caller's own logs still have it."""
    with meter.metered("sarvam", "characters") as m:
        m.succeeded(10)
    assert recorded[0]["call_id"] is None


def test_a_failed_provider_call_is_still_recorded(recorded):
    """The whole point. A row with ok=false and zero quantity is how a
    degrading provider becomes visible; no row at all is how it stays hidden."""
    with meter.for_call("call-1"), meter.metered("campaignbot", "messages") as m:
        m.failed(0, "HTTP 502")
    assert recorded[0]["ok"] is False
    assert recorded[0]["quantity"] == 0
    assert "502" in recorded[0]["error"]


def test_an_exception_is_recorded_and_then_re_raised(recorded):
    """A provider call that blew up still took time and still says something
    about that provider. Recorded on the way out, not swallowed."""
    with pytest.raises(RuntimeError):
        with meter.for_call("call-1"), meter.metered("sarvam", "characters"):
            raise RuntimeError("connection reset")
    assert recorded[0]["ok"] is False
    assert "connection reset" in recorded[0]["error"]


def test_a_block_that_sets_nothing_records_a_failure(recorded):
    """The honest default. A block that returned without claiming success did
    attempt the call, and recording that as a silent success would hide exactly
    the case worth seeing."""
    with meter.for_call("call-1"), meter.metered("sarvam", "characters"):
        pass
    assert recorded[0]["ok"] is False


def test_each_metered_call_gets_its_own_dedup_ref(recorded):
    """Several translations of one lead must each get a row rather than
    overwriting one another on the ledger's primary key."""
    with meter.for_call("call-1"):
        for _ in range(3):
            with meter.metered("sarvam", "characters", operation="translate") as m:
                m.succeeded(10)
    assert len({row["ref"] for row in recorded}) == 3


def test_the_call_scope_is_restored_afterwards(recorded):
    with meter.for_call("outer"):
        with meter.for_call("inner"):
            assert meter.current_call() == "inner"
        assert meter.current_call() == "outer"
    assert meter.current_call() is None

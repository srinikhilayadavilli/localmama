"""Pricing, against a real Postgres.

The arithmetic that turns units into money lives in SQL — in the
`localmama.usage_priced` view, which resolves an effective-dated rate card by
specificity. That is the right home for it (one implementation, and a
correction re-prices history), and it means a mock cannot test it. So these run
against a real database or not at all.

    createdb localmama_test
    TEST_DATABASE_URL=postgresql:///localmama_test make test

Skipped without `TEST_DATABASE_URL`, so the default suite stays hermetic.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

TEST_DSN = os.environ.get("TEST_DATABASE_URL", "")

pytestmark = pytest.mark.skipif(
    not TEST_DSN,
    reason="set TEST_DATABASE_URL to a scratch Postgres to exercise pricing",
)


@pytest.fixture()
def priced(monkeypatch):
    """A migrated, empty schema plus the `costing` module bound to it."""
    import psycopg

    from backend import costing, db
    from backend.config import settings
    from backend.migrate import available_migrations

    object.__setattr__(settings, "database_url", TEST_DSN)
    db.close()

    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("DROP SCHEMA IF EXISTS localmama CASCADE")
        for _, sql in available_migrations():
            conn.execute(sql)

    yield costing
    db.close()


def _insert(sql: str, params: tuple = ()) -> None:
    import psycopg

    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute(sql, params)


def _lead(call_id: str, *, service=None, handoff="pending", status="completed",
          ended_minutes_ago: int = 5) -> None:
    ended = datetime.now(timezone.utc) - timedelta(minutes=ended_minutes_ago)
    _insert(
        "INSERT INTO localmama.leads (call_id, agent_id, caller_phone, service,"
        " status, handoff_status, started_at, ended_at, processed_at)"
        " VALUES (%s, 'localmama', '+91999', %s, %s, %s, %s, %s, now())",
        (call_id, service, status, handoff,
         ended - timedelta(minutes=3), ended),
    )


def _usage(call_id: str, provider: str, unit: str, quantity: float, *,
           model="", operation="", estimated=False, ref="session",
           source="agent", ok=None) -> None:
    _insert(
        "INSERT INTO localmama.usage (call_id, ref, source, provider, model,"
        " operation, unit, quantity, estimated, ok)"
        " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
        (call_id, ref, source, provider, model, operation, unit, quantity,
         estimated, ok),
    )


# ── the rate card resolves by specificity ───────────────────────────────────


@pytest.mark.parametrize(
    ("model", "expected_per_million"),
    [
        ("gpt-realtime", 64.0),
        ("gpt-realtime-2.1", 64.0),
        ("gpt-realtime-2.1-mini", 20.0),
        # A dated snapshot must inherit its own base, not a shorter prefix.
        # Getting this wrong prices the mini model at three times its rate.
        ("gpt-realtime-2.1-mini-2026-01-15", 20.0),
        ("gpt-realtime-2.1-2026-01-15", 64.0),
    ],
)
def test_a_model_is_priced_by_the_most_specific_matching_rate(
    priced, model, expected_per_million,
):
    _lead("c1", service="plumber")
    _usage("c1", "openai", "audio_output_tokens", 1_000_000,
           model=model, operation="realtime")
    assert priced.for_call("c1")["cost_usd"] == pytest.approx(expected_per_million)


def test_cached_audio_is_priced_at_the_cached_rate(priced):
    """The 80x gap, end to end. 48k cached and 12k fresh must come to
    $0.4032 — not the $1.92 that billing all 60k as fresh would give."""
    _lead("c1", service="plumber")
    _usage("c1", "openai", "audio_input_tokens", 12_000,
           model="gpt-realtime", operation="realtime")
    _usage("c1", "openai", "audio_input_cached_tokens", 48_000,
           model="gpt-realtime", operation="realtime")
    assert priced.for_call("c1")["cost_usd"] == pytest.approx(
        12_000 / 1e6 * 32.0 + 48_000 / 1e6 * 0.40
    )


def test_the_session_and_the_phone_line_are_billed_separately(priced):
    """Same provider, same empty model, same unit — distinguished only by
    operation. Without operation in the key one silently overwrites the other
    and every call is billed half its phone time."""
    _lead("c1", service="plumber")
    _usage("c1", "livekit", "seconds", 180, operation="session")
    _usage("c1", "livekit", "seconds", 180, operation="telephony")
    assert priced.for_call("c1")["cost_usd"] == pytest.approx(2 * 180 * 0.01 / 60)


# ── coverage: the failures that look like cheapness ─────────────────────────


def test_an_unknown_provider_is_unpriced_rather_than_free(priced):
    """The failure this whole subsystem exists to prevent: a provider with no
    rate contributing exactly zero to every chart, indistinguishable from a
    provider that is genuinely free."""
    _lead("c1", service="plumber")
    _usage("c1", "elevenlabs", "characters", 900, model="flash-v2", operation="tts")
    detail = priced.for_call("c1")
    assert detail["unpriced_units"] == 1
    assert detail["units"][0]["cost_usd"] is None

    cover = priced.coverage()
    assert cover["unpriced_units"] == 1
    assert {"provider": "elevenlabs", "model": "flash-v2",
            "unit": "characters"} in cover["unpriced"]


def test_a_call_with_no_usage_reports_unmetered_rather_than_zero(priced):
    """A call whose `call.ended` never landed has no ledger. It must not read
    as a free call — the totals around it are a floor, and saying so is the
    difference between a number and a guess."""
    _lead("c1", service="plumber")  # no usage rows at all
    assert priced.for_call("c1")["metered"] is False
    assert priced.coverage()["unmetered_calls"] == 1


def test_a_placeholder_rate_is_counted_against_confidence(priced):
    """CampaignBot ships with a knowingly-wrong zero. Priced, so it is not
    "unpriced", but excluded from confidence so nobody reads the total as
    complete."""
    _lead("c1", service="plumber")
    _usage("c1", "campaignbot", "messages", 1, operation="handoff",
           ref="w1", source="backend", ok=True)
    cover = priced.coverage()
    assert cover["placeholder_units"] == 1
    assert cover["confidence_pct"] == 0.0


def test_an_estimated_quantity_is_flagged_but_still_priced(priced):
    _lead("c1", service="plumber")
    _usage("c1", "openai", "seconds", 180, model="gpt-4o-transcribe",
           operation="transcription", estimated=True)
    detail = priced.for_call("c1")
    assert detail["estimated_units"] == 1
    assert detail["cost_usd"] == pytest.approx(180 * 0.006 / 60)


# ── unit economics ──────────────────────────────────────────────────────────


def test_cost_per_delivered_lead_is_higher_than_cost_per_call(priced):
    """The distinction the page is built around. Three calls, one delivered:
    a lead costs three times what the per-call average suggests, and quoting
    the wrong one understates the business by the whole conversion loss."""
    for i, (service, handoff) in enumerate(
        [("plumber", "sent"), ("salon", "pending"), (None, "pending")]
    ):
        _lead(f"c{i}", service=service, handoff=handoff)
        _usage(f"c{i}", "livekit", "seconds", 600, operation="session")

    economics = priced.unit_economics()
    assert economics["calls"] == 3
    assert economics["delivered"] == 1
    assert economics["cost_per_delivered_usd"] == pytest.approx(
        3 * economics["cost_per_call_usd"]
    )


def test_delivering_nothing_reports_none_rather_than_zero(priced):
    """An hour that delivered no leads must not render as "leads are free"."""
    _lead("c1", service="plumber", handoff="pending")
    _usage("c1", "livekit", "seconds", 120, operation="session")
    assert priced.unit_economics()["cost_per_delivered_usd"] is None


def test_a_call_outside_the_window_is_excluded(priced):
    _lead("old", service="plumber", handoff="sent", ended_minutes_ago=60 * 40)
    _usage("old", "livekit", "seconds", 600, operation="session")
    assert priced.unit_economics(hours=24)["calls"] == 0
    assert priced.unit_economics(hours=24 * 3)["calls"] == 1


def test_spend_is_grouped_by_provider_and_failures_are_visible(priced):
    """A failed provider call is recorded with zero quantity and ok=false, so
    an outage shows up as failures rather than as an absence of rows."""
    _lead("c1", service="plumber")
    _usage("c1", "sarvam", "characters", 40, operation="translate",
           ref="t1", source="backend", ok=True)
    _usage("c1", "sarvam", "characters", 0, operation="translate",
           ref="t2", source="backend", ok=False)
    sarvam = next(r for r in priced.by_provider() if r["provider"] == "sarvam")
    assert sarvam["failed"] == 1
    assert sarvam["cost_usd"] == pytest.approx(40 / 10_000 * 0.23)


def test_the_rate_card_reads_back_without_choking_on_infinity(priced):
    """`-infinity` means "always", and psycopg raises rather than converting
    it. Reported as null, which is what a reader means by "no start date"."""
    card = priced.rate_card()
    assert card
    assert all(r["effective_from"] is None for r in card)
    assert any(r["placeholder"] for r in card)


def test_a_rate_change_reprices_only_calls_after_it(priced):
    """The whole reason the card is effective-dated. Correcting a price must
    not silently restate last quarter at today's number."""
    changed_at = datetime.now(timezone.utc) - timedelta(minutes=30)
    _insert("UPDATE localmama.rate_card SET effective_to = %s"
            " WHERE provider = 'livekit' AND operation = 'session'", (changed_at,))
    _insert(
        "INSERT INTO localmama.rate_card (provider, model, operation, unit,"
        " per_units, unit_price_usd, effective_from, source)"
        " VALUES ('livekit', '', 'session', 'seconds', 1, 0.001, %s, 'test')",
        (changed_at,),
    )
    _lead("before", service="plumber", ended_minutes_ago=90)
    _lead("after", service="plumber", ended_minutes_ago=5)
    _insert("INSERT INTO localmama.usage (call_id, source, provider, operation,"
            " unit, quantity, at) VALUES ('before','agent','livekit','session',"
            " 'seconds', 100, %s)", (datetime.now(timezone.utc) - timedelta(minutes=90),))
    _usage("after", "livekit", "seconds", 100, operation="session")

    assert priced.for_call("before")["cost_usd"] == pytest.approx(100 * 0.01 / 60)
    assert priced.for_call("after")["cost_usd"] == pytest.approx(100 * 0.001)


# ── the turn series ─────────────────────────────────────────────────────────


def test_latency_reports_the_gap_and_the_models_share_of_it(priced):
    _lead("c1", service="plumber")
    _insert("UPDATE localmama.leads SET turn_gaps = '[0.9,1.4,2.2,0.7]'::jsonb"
            " WHERE call_id = 'c1'")
    for i, ttft in enumerate([0.6, 0.8, 1.4]):
        _insert(
            "INSERT INTO localmama.call_turns (call_id, ref, at, ttft, duration,"
            " input_tokens, cached_tokens, output_tokens)"
            " VALUES ('c1', %s, %s, %s, 1.8, 900, 100, 300)",
            (f"r{i}", float(i * 5), ttft),
        )
    lag = priced.latency()
    assert lag["turn_gap"]["samples"] == 4
    assert lag["ttft"]["p50"] == pytest.approx(0.8)


def test_a_response_with_no_audio_is_excluded_from_ttft(priced):
    """-1 is "no audio token was sent", not "instant". Averaged in it would
    drag the percentile down and hide a real regression."""
    _lead("c1", service="plumber")
    for i, ttft in enumerate([-1.0, 2.0, 2.0]):
        _insert(
            "INSERT INTO localmama.call_turns (call_id, ref, at, ttft, duration,"
            " input_tokens, cached_tokens, output_tokens)"
            " VALUES ('c1', %s, %s, %s, 1.0, 100, 0, 50)", (f"r{i}", float(i), ttft),
        )
    lag = priced.latency()
    assert lag["ttft"]["samples"] == 2
    assert lag["ttft"]["p50"] == pytest.approx(2.0)

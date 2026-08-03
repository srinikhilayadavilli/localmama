"""What calls cost, and how much of that number to believe.

Reads. Everything here queries the ledger written by `store` and priced by the
`localmama.usage_priced` view; nothing here writes, and nothing here computes a
price in Python. That is deliberate: the rate card is effective-dated in SQL so
that a correction re-prices history, and a second implementation of the same
arithmetic up here is how the dashboard and the invoice come to disagree.

Three groups of questions:

  **What did it cost** — per call, per provider, per model, over a window.
  **What is it costing us per outcome** — the number the business actually runs
  on, which is not cost per call but cost per lead that reached a customer. A
  call that captured nothing still cost money.
  **How much of this is real** — unpriced units, estimated units, placeholder
  rates, and calls that were never metered at all. A costing system that cannot
  report its own coverage is a costing system that quietly rounds down.

Every function degrades to an empty result rather than raising. A dashboard
that cannot render is worse than one showing a gap, and none of this is on the
path of a call or a lead.
"""

from __future__ import annotations

import time
from typing import Any

from . import db
from .config import settings
from .logger import get_logger

logger = get_logger("localmama.costing")

#: Default window for every aggregate, in hours.
DEFAULT_WINDOW_HOURS = 24

#: How long to stop trying after the database proves unreachable.
#:
#: A summary is a dozen queries and the pool waits ten seconds before giving up
#: on a connection, so an unreachable Neon turned one scrape into a two-minute
#: hang — at exactly the moment the monitoring is meant to be telling somebody
#: that something is wrong. A scraper times out, the series goes blank, and the
#: outage looks like an absence of data rather than an outage.
#:
#: So the first connection failure short-circuits the rest of that request and
#: the next few seconds of them. Deliberately narrow: only failures that mean
#: "no database" trip it. A malformed query is this module's own bug and must
#: not blank out every other panel on the page.
_BREAKER_COOLDOWN = 15.0
_breaker_open_until = 0.0


def _is_unreachable(exc: BaseException) -> bool:
    """Whether this failure means the database is away rather than the SQL bad."""
    import psycopg
    import psycopg_pool

    return isinstance(exc, (psycopg.OperationalError, psycopg_pool.PoolTimeout,
                            psycopg.InterfaceError))


def _rows(sql: str, params: tuple = (), what: str = "query") -> list[tuple]:
    """Run a read. Returns [] on any failure, loudly."""
    global _breaker_open_until
    if not db.available():
        return []
    if time.monotonic() < _breaker_open_until:
        return []
    try:
        with db.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        _breaker_open_until = 0.0
        return rows
    except Exception as exc:  # noqa: BLE001 - a dashboard must never 500 on SQL
        if _is_unreachable(exc):
            _breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN
            logger.warning("costing: database unreachable (%s); pausing reads for %.0fs",
                           exc, _BREAKER_COOLDOWN)
        else:
            logger.warning("costing: %s failed: %s", what, exc)
        return []


def _dicts(sql: str, params: tuple, columns: list[str], what: str) -> list[dict]:
    return [dict(zip(columns, row)) for row in _rows(sql, params, what)]


def in_rupees(usd: float | None) -> float:
    """USD to INR, for display. See `settings.inr_per_usd`."""
    return round((usd or 0.0) * settings.inr_per_usd, 4)


# ── per call ────────────────────────────────────────────────────────────────


def for_call(call_id: str) -> dict:
    """The full cost breakdown for one call, with its caveats.

    `units` is every metered quantity with the rate that applied, which is what
    makes a surprising total explainable rather than merely surprising — the
    answer to "why did that call cost 60 cents" is almost always one row of it.
    """
    total = _dicts(
        "SELECT call_id, cost_usd::float8, unpriced_units::int, estimated_units::int,"
        " placeholder_units::int, inherited_units::int, by_provider"
        " FROM localmama.call_cost WHERE call_id = %s",
        (call_id,),
        ["call_id", "cost_usd", "unpriced_units", "estimated_units",
         "placeholder_units", "inherited_units", "by_provider"],
        "for_call total",
    )
    units = _dicts(
        "SELECT source, provider, model, operation, unit, quantity, estimated,"
        " unit_price_usd::float8, per_units,"
        " round(cost_usd::numeric, 8)::float8 AS cost_usd,"
        " unpriced, rate_inherited, placeholder, ok, latency_ms, error"
        " FROM localmama.usage_priced WHERE call_id = %s"
        " ORDER BY cost_usd DESC NULLS LAST, provider, unit",
        (call_id,),
        ["source", "provider", "model", "operation", "unit", "quantity",
         "estimated", "unit_price_usd", "per_units", "cost_usd", "unpriced",
         "rate_inherited", "placeholder", "ok", "latency_ms", "error"],
        "for_call units",
    )
    head = total[0] if total else {"call_id": call_id, "cost_usd": 0.0}
    return {
        **head,
        "cost_inr": in_rupees(float(head.get("cost_usd") or 0.0)),
        "units": units,
        # No ledger rows at all. Distinct from a call that cost nothing: it
        # means `call.ended` never arrived, or arrived from an agent that
        # cannot count. Either way the total below is a floor, not a figure.
        "metered": bool(units),
    }


def turns_for_call(call_id: str) -> list[dict]:
    """The per-response series: which turn made this call expensive.

    Ordered by time rather than by size on purpose. The shape that matters is
    the trend — input tokens climbing turn over turn is a context that is not
    being trimmed, and it reads as a slope rather than as a maximum.
    """
    return _dicts(
        "SELECT ref, at, ttft, duration, input_tokens, cached_tokens,"
        " output_tokens, cancelled FROM localmama.call_turns"
        " WHERE call_id = %s ORDER BY at",
        (call_id,),
        ["ref", "at", "ttft", "duration", "input_tokens", "cached_tokens",
         "output_tokens", "cancelled"],
        "turns_for_call",
    )


# ── aggregates ──────────────────────────────────────────────────────────────


def by_provider(hours: int = DEFAULT_WINDOW_HOURS) -> list[dict]:
    """Spend by provider over the window, most expensive first.

    This is the chart that answers "where does the money go", and on this
    deployment the answer is usually one row — the realtime model — which is
    worth seeing plainly before anyone optimises a Sarvam call that costs a
    hundredth of a cent.
    """
    return _dicts(
        "SELECT provider,"
        "       round(coalesce(sum(cost_usd), 0)::numeric, 6)::float8 AS cost_usd,"
        "       count(DISTINCT call_id)::int                       AS calls,"
        "       count(*) FILTER (WHERE unpriced)::int              AS unpriced,"
        "       count(*) FILTER (WHERE ok IS FALSE)::int           AS failed"
        "  FROM localmama.usage_priced"
        " WHERE at > now() - make_interval(hours => %s)"
        " GROUP BY provider ORDER BY 2 DESC",
        (hours,),
        ["provider", "cost_usd", "calls", "unpriced", "failed"],
        "by_provider",
    )


def by_model(hours: int = DEFAULT_WINDOW_HOURS) -> list[dict]:
    """Spend and volume by model, with the cost of an average call on each.

    The comparison that makes a model switch a decision rather than a guess: if
    the mini model handles the same calls at a third of the price, this is
    where that shows up — and if it takes more turns to get there, this is
    where that shows up too.
    """
    return _dicts(
        "SELECT provider, model,"
        "       count(DISTINCT call_id)::int                       AS calls,"
        "       round(coalesce(sum(cost_usd), 0)::numeric, 6)::float8 AS cost_usd,"
        "       round((coalesce(sum(cost_usd), 0)"
        "              / greatest(count(DISTINCT call_id), 1))::numeric, 6)::float8 AS cost_per_call_usd"
        "  FROM localmama.usage_priced"
        " WHERE at > now() - make_interval(hours => %s) AND model <> ''"
        " GROUP BY provider, model ORDER BY 4 DESC",
        (hours,),
        ["provider", "model", "calls", "cost_usd", "cost_per_call_usd"],
        "by_model",
    )


def expensive_calls(hours: int = DEFAULT_WINDOW_HOURS, limit: int = 20) -> list[dict]:
    """The costliest calls in the window, with what they produced.

    Cost beside outcome, because neither means much alone. An expensive call
    that delivered a lead is the system working; an expensive call that
    captured nothing is a model that talked for four minutes to no purpose,
    and only the pairing tells them apart.
    """
    return _dicts(
        "SELECT c.call_id,"
        "       round(c.cost_usd, 6)::float8      AS cost_usd,"
        "       l.status, l.service, l.city, l.needs_review, l.whatsapp_status,"
        "       l.confirmed,"
        "       extract(epoch FROM (l.ended_at - l.started_at))::float8 AS duration,"
        "       (SELECT count(*) FROM localmama.call_turns t"
        "         WHERE t.call_id = c.call_id)::int AS turns"
        "  FROM localmama.call_cost c"
        "  JOIN localmama.leads l ON l.call_id = c.call_id"
        " WHERE l.ended_at > now() - make_interval(hours => %s)"
        " ORDER BY c.cost_usd DESC LIMIT %s",
        (hours, limit),
        ["call_id", "cost_usd", "status", "service", "city", "needs_review",
         "whatsapp_status", "confirmed", "duration", "turns"],
        "expensive_calls",
    )


def unit_economics(hours: int = DEFAULT_WINDOW_HOURS) -> dict:
    """Cost per call, and cost per outcome. The number the business runs on.

    Cost per *call* is the easy number and the misleading one: it treats a
    wrong number and a delivered lead as the same unit of production. What a
    lead actually costs is total spend divided by the leads that reached a
    customer — which on a funnel that converts half its calls is twice the
    figure anyone quotes from the per-call average.

    Both are reported, along with the funnel between them, so the gap is
    visible rather than a matter of which number someone happened to pick up.
    """
    rows = _rows(
        "WITH window_calls AS ("
        "    SELECT l.call_id, l.status, l.service, l.whatsapp_status, l.needs_review"
        "      FROM localmama.leads l"
        "     WHERE l.ended_at > now() - make_interval(hours => %s)"
        "), spend AS ("
        "    SELECT coalesce(sum(c.cost_usd), 0) AS total"
        "      FROM localmama.call_cost c"
        "      JOIN window_calls w ON w.call_id = c.call_id"
        ")"
        " SELECT (SELECT total FROM spend)                                       AS cost_usd,"
        "        count(*)                                                        AS calls,"
        "        count(*) FILTER (WHERE service IS NOT NULL)                     AS captured,"
        "        count(*) FILTER (WHERE whatsapp_status = 'sent')                AS delivered,"
        "        count(*) FILTER (WHERE needs_review)                            AS flagged,"
        "        count(*) FILTER (WHERE status = 'completed')                    AS completed"
        "   FROM window_calls",
        (hours,),
        "unit_economics",
    )
    if not rows:
        return {}
    cost, calls, captured, delivered, flagged, completed = rows[0]
    cost = float(cost or 0.0)

    def per(n: int) -> float | None:
        # None rather than zero when the denominator is zero. "We delivered
        # nothing this hour" must not render as "leads are free".
        return round(cost / n, 6) if n else None

    return {
        "window_hours": hours,
        "cost_usd": round(cost, 6),
        "cost_inr": in_rupees(cost),
        "calls": calls,
        "captured": captured,
        "delivered": delivered,
        "flagged": flagged,
        "completed": completed,
        "cost_per_call_usd": per(calls),
        "cost_per_captured_usd": per(captured),
        #: The real one.
        "cost_per_delivered_usd": per(delivered),
        "cost_per_delivered_inr": in_rupees(per(delivered) or 0.0) if delivered else None,
        "capture_rate": round(100.0 * captured / calls, 1) if calls else None,
        "delivery_rate": round(100.0 * delivered / captured, 1) if captured else None,
    }


def latency(hours: int = DEFAULT_WINDOW_HOURS) -> dict:
    """The two latency numbers that mean different things.

    **turn_gap** is the silence the caller actually sat through between
    finishing their sentence and hearing a reply. It is measured in the agent,
    end to end, and it is the only one a caller could describe.

    **ttft** is how long the model took to start producing audio. It is a
    component of the gap, not a substitute for it: turn detection, the network
    and playback all sit in between, and a deployment where ttft is healthy and
    the gap is not has a turn-detection problem rather than a model problem.
    Reported side by side so that distinction is available at a glance.
    """
    gaps = _rows(
        "SELECT count(*)::int,"
        "       round(percentile_cont(0.5) WITHIN GROUP (ORDER BY g)::numeric, 3)::float8,"
        "       round(percentile_cont(0.95) WITHIN GROUP (ORDER BY g)::numeric, 3)::float8"
        "  FROM localmama.leads l"
        " CROSS JOIN LATERAL jsonb_array_elements_text(l.turn_gaps) AS e(v)"
        " CROSS JOIN LATERAL (SELECT e.v::float8 AS g) AS x"
        " WHERE l.ended_at > now() - make_interval(hours => %s)",
        (hours,), "latency gaps",
    )
    ttfts = _rows(
        "SELECT count(*)::int,"
        "       round(percentile_cont(0.5) WITHIN GROUP (ORDER BY ttft)::numeric, 3)::float8,"
        "       round(percentile_cont(0.95) WITHIN GROUP (ORDER BY ttft)::numeric, 3)::float8,"
        "       count(*) FILTER (WHERE cancelled)::int"
        "  FROM localmama.call_turns t"
        "  JOIN localmama.leads l ON l.call_id = t.call_id"
        # -1 means the response carried no audio at all. Excluded rather than
        # counted as instant, which is what averaging it in would do.
        " WHERE t.ttft >= 0 AND l.ended_at > now() - make_interval(hours => %s)",
        (hours,), "latency ttft",
    )
    gap_n, gap_p50, gap_p95 = gaps[0] if gaps else (0, None, None)
    ttft_n, ttft_p50, ttft_p95, cancelled = ttfts[0] if ttfts else (0, None, None, 0)
    return {
        "window_hours": hours,
        "turn_gap": {"samples": gap_n, "p50": gap_p50, "p95": gap_p95},
        "ttft": {"samples": ttft_n, "p50": ttft_p50, "p95": ttft_p95},
        # Barge-ins. A rising count is turn detection firing on noise, and it
        # shows up here before anyone thinks to describe the calls as choppy.
        "cancelled_responses": cancelled,
    }


def backlog() -> dict:
    """Work the system owes but has not done. The other half of health.

    Cost and latency describe calls that finished. These are the ones that did
    not finish properly — a lead the pipeline never processed, a message still
    owed to a customer. Both are silent by nature: nothing errors, somebody
    simply never hears from us.
    """
    rows = _rows(
        "SELECT count(*) FILTER (WHERE ended_at IS NOT NULL AND processed_at IS NULL)::int,"
        "       count(*) FILTER (WHERE whatsapp_status IN ('pending','sending')"
        "                          AND caller_phone IS NOT NULL"
        "                          AND processed_at IS NOT NULL)::int,"
        "       count(*) FILTER (WHERE needs_review)::int,"
        "       count(*) FILTER (WHERE status = 'in_progress'"
        "                          AND created_at < now() - interval '1 hour')::int"
        "  FROM localmama.leads",
        (), "backlog",
    )
    unprocessed, owed, review, stuck = rows[0] if rows else (0, 0, 0, 0)
    return {
        "unprocessed_leads": unprocessed,
        "owed_handoffs": owed,
        "needs_review": review,
        # A call that started, never ended, and is an hour old. The agent
        # crashed mid-call, or `call.ended` never landed — which is also
        # exactly the population that has no usage and no cost.
        "never_closed": stuck,
    }


# ── how much of this is real ────────────────────────────────────────────────


def coverage(hours: int = DEFAULT_WINDOW_HOURS) -> dict:
    """What the cost numbers do not know, stated plainly.

    A costing system's worst failure is not being wrong — it is being wrong
    quietly, in the direction of cheap. Every mechanism here fails that way if
    unwatched: a model with no rate card row prices at nothing, a placeholder
    rate prices at nothing, a call whose `call.ended` never arrived has no
    ledger at all and reads as a free call. Each is counted so the totals can
    be read with the right amount of confidence.
    """
    unmetered = _rows(
        "SELECT count(*) FROM localmama.leads l"
        " WHERE l.ended_at > now() - make_interval(hours => %s)"
        "   AND NOT EXISTS (SELECT 1 FROM localmama.usage u"
        "                    WHERE u.call_id = l.call_id)",
        (hours,), "coverage unmetered",
    )
    gaps = _rows(
        "SELECT count(*) FILTER (WHERE unpriced)                    AS unpriced,"
        "       count(*) FILTER (WHERE placeholder)                 AS placeholder,"
        "       count(*) FILTER (WHERE estimated AND quantity > 0)  AS estimated,"
        "       count(*) FILTER (WHERE rate_inherited)              AS inherited,"
        "       count(*)                                            AS total"
        "  FROM localmama.usage_priced"
        " WHERE at > now() - make_interval(hours => %s)",
        (hours,), "coverage gaps",
    )
    unpriced_detail = _dicts(
        "SELECT DISTINCT provider, model, unit FROM localmama.usage_priced"
        " WHERE unpriced AND at > now() - make_interval(hours => %s)"
        " ORDER BY provider, model, unit",
        (hours,), ["provider", "model", "unit"], "coverage detail",
    )
    unpriced, placeholder, estimated, inherited, total = (
        gaps[0] if gaps else (0, 0, 0, 0, 0)
    )
    return {
        "window_hours": hours,
        "unmetered_calls": unmetered[0][0] if unmetered else 0,
        "unpriced_units": unpriced,
        "placeholder_units": placeholder,
        "estimated_units": estimated,
        "inherited_units": inherited,
        "total_units": total,
        # The single number to put on the dashboard: what fraction of metered
        # quantities carry a rate somebody actually confirmed.
        "confidence_pct": round(
            100.0 * (total - unpriced - placeholder) / total, 1
        ) if total else None,
        "unpriced": unpriced_detail,
    }


def rate_card() -> list[dict]:
    """The rates currently in force, so a total can be traced to a number."""
    return _dicts(
        "SELECT provider, model, operation, unit, per_units, unit_price_usd::float8,"
        " placeholder, source, note,"
        # -infinity means 'always', and psycopg cannot map it to a datetime at
        # all — it raises rather than returning something odd. Reported as
        # null, which is what 'no start date' means to a reader anyway.
        " nullif(effective_from, '-infinity') AS effective_from"
        "  FROM localmama.rate_card"
        " WHERE effective_to IS NULL OR effective_to > now()"
        " ORDER BY provider, model, operation, unit",
        (),
        ["provider", "model", "operation", "unit", "per_units", "unit_price_usd",
         "placeholder", "source", "note", "effective_from"],
        "rate_card",
    )


def prune_turns() -> int:
    """Drop turn series past the retention window. Returns rows removed.

    Only the turn series. The usage ledger is the billing record and is kept —
    it is small, and a cost history that thins out is not a cost history.
    """
    days = settings.turn_retention_days
    if days <= 0:
        return 0
    try:
        with db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM localmama.call_turns t USING localmama.leads l"
                    " WHERE l.call_id = t.call_id"
                    "   AND l.ended_at < now() - make_interval(days => %s)",
                    (days,),
                )
                removed = cur.rowcount
            conn.commit()
        if removed:
            logger.info("pruned %d turn row(s) past %d-day retention", removed, days)
        return removed
    except Exception as exc:  # noqa: BLE001 - a sweep must never kill the worker
        logger.warning("turn pruning failed: %s", exc)
        return 0


def summary(hours: int = DEFAULT_WINDOW_HOURS) -> dict[str, Any]:
    """Everything the dashboard needs, in one round of queries."""
    return {
        "economics": unit_economics(hours),
        "latency": latency(hours),
        "backlog": backlog(),
        "by_provider": by_provider(hours),
        "by_model": by_model(hours),
        "coverage": coverage(hours),
        "expensive": expensive_calls(hours),
    }

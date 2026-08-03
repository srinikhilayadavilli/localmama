"""The operator-facing surface: what is happening, and what it is costing.

Kept apart from `api.py`, which is the agent's surface and has exactly one
client. These endpoints have different readers (a scraper, a dashboard, a
person with curl), a different secret, and different failure expectations — a
500 here is a missing graph, while a 500 there is a lost lead. Mixing them
would put a reporting query on the same router as the one thing that must
never be slow.

    GET /metrics              Prometheus exposition
    GET /v1/ops/summary       everything the dashboard needs, as JSON
    GET /v1/ops/calls/{id}    one call: cost breakdown and turn series
    GET /v1/ops/rates         the rate card currently in force

Everything is read-only and every handler degrades to an empty result rather
than raising, because `costing` already swallows its own failures. What that
buys is a dashboard that shows a gap instead of a stack trace when Neon is
briefly unreachable.
"""

from __future__ import annotations

import hmac

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import PlainTextResponse

from . import costing
from .config import settings

router = APIRouter()

#: Ceiling on the reporting window. Not a performance guard — these queries are
#: indexed on `at` — but a reminder that this is an operational view and not an
#: analytics warehouse. A year of unit economics is a question for SQL.
MAX_WINDOW_HOURS = 24 * 90


async def authorise_ops(authorization: str = Header(default="")) -> None:
    """The ops token, compared in constant time.

    An unset token refuses everything rather than allowing everything, the same
    rule the agent surface follows and for the same reason: a deployment that
    silently publishes its cost per lead does not look misconfigured from the
    outside.
    """
    if not settings.ops_token:
        raise HTTPException(503, "OPS_TOKEN is not configured")
    presented = authorization.removeprefix("Bearer ").strip()
    if not hmac.compare_digest(presented, settings.ops_token):
        raise HTTPException(401, "bad token")


def _window(hours: int) -> int:
    return max(1, min(int(hours), MAX_WINDOW_HOURS))


@router.get("/v1/ops/summary", dependencies=[Depends(authorise_ops)])
async def summary(hours: int = Query(costing.DEFAULT_WINDOW_HOURS, ge=1)) -> dict:
    """Volume, funnel, latency, backlog, spend, and the coverage caveats."""
    return costing.summary(_window(hours))


@router.get("/v1/ops/calls/{call_id}", dependencies=[Depends(authorise_ops)])
async def call_detail(call_id: str) -> dict:
    """One call, priced line by line, with its per-response series.

    The pair is the point: the total says a call was expensive and the series
    says which turn made it so. On a realtime model that is almost always a
    context that grew without being trimmed, and it is visible as a slope in
    `input_tokens` rather than as any single bad number.
    """
    return {
        "cost": costing.for_call(call_id),
        "turns": costing.turns_for_call(call_id),
    }


@router.get("/v1/ops/rates", dependencies=[Depends(authorise_ops)])
async def rates() -> dict:
    """The rates in force, so any total can be traced back to a published number."""
    card = costing.rate_card()
    return {
        "rates": card,
        "placeholders": [r for r in card if r.get("placeholder")],
        "inr_per_usd": settings.inr_per_usd,
    }


# ── Prometheus ──────────────────────────────────────────────────────────────


def _escape(value: str) -> str:
    return str(value).replace("\\", r"\\").replace('"', r"\"").replace("\n", " ")


class _Exposition:
    """A minimal Prometheus text builder.

    Hand-rolled rather than pulling in a client library. The exposition format
    is a dozen lines of string formatting, and this service's whole argument is
    that it carries four dependencies and no model weights — adding one so a
    scrape endpoint can exist would be a poor trade. There is no registry and
    no process-global state here on purpose: every scrape renders from a fresh
    query, so two workers cannot report each other's numbers.
    """

    def __init__(self) -> None:
        # Samples are bucketed by metric name rather than appended to one list,
        # because the exposition format requires every sample of a family to
        # appear as a single contiguous group under one HELP/TYPE pair. Writing
        # them in call order interleaves families the moment two metrics are
        # emitted inside the same loop — which reads fine and which a strict
        # parser rejects outright.
        self._samples: dict[str, list[str]] = {}
        self._meta: dict[str, tuple[str, str]] = {}

    def add(self, name: str, value, *, help_: str = "", kind: str = "gauge",
            **labels) -> None:
        """One sample. Silently skips a None, which is how `costing` reports
        'the denominator was zero' — and a zero would be a lie about it."""
        if value is None:
            return
        self._meta.setdefault(name, (help_, kind))
        rendered = ",".join(f'{k}="{_escape(v)}"' for k, v in sorted(labels.items()))
        suffix = f"{{{rendered}}}" if rendered else ""
        self._samples.setdefault(name, []).append(f"{name}{suffix} {float(value)}")

    def render(self) -> str:
        lines: list[str] = []
        for name, samples in self._samples.items():  # insertion-ordered
            help_, kind = self._meta[name]
            if help_:
                lines.append(f"# HELP {name} {_escape(help_)}")
            lines.append(f"# TYPE {name} {kind}")
            lines.extend(samples)
        return "\n".join(lines) + "\n"


def _exposition(hours: int) -> str:
    """Render the whole picture as Prometheus text.

    The metric set is chosen so that the three questions an operator actually
    pages on are each one query: is it working (funnel and backlog), is it fast
    (latency), and is it affordable (cost per delivered lead). The coverage
    gauges are here for a fourth that is easy to forget — whether the cost
    numbers still mean anything.
    """
    out = _Exposition()
    economics = costing.unit_economics(hours) or {}
    lag = costing.latency(hours) or {}
    gaps = costing.backlog() or {}
    cover = costing.coverage(hours) or {}

    out.add("localmama_calls", economics.get("calls"), kind="gauge",
            help_="Calls that ended in the window.")
    for stage in ("captured", "delivered", "completed", "flagged"):
        out.add("localmama_calls_stage", economics.get(stage), stage=stage,
                help_="Calls reaching each stage of the funnel.")

    out.add("localmama_cost_usd", economics.get("cost_usd"),
            help_="Total metered spend in the window, USD.")
    out.add("localmama_cost_per_call_usd", economics.get("cost_per_call_usd"),
            help_="Spend divided by calls. The easy number.")
    out.add("localmama_cost_per_delivered_usd", economics.get("cost_per_delivered_usd"),
            help_="Spend divided by leads that reached a customer. The real one.")

    for row in costing.by_provider(hours):
        out.add("localmama_provider_cost_usd", row["cost_usd"],
                provider=row["provider"], help_="Spend by provider, USD.")
        out.add("localmama_provider_failed_calls", row["failed"],
                provider=row["provider"],
                help_="Backend provider calls that did not succeed.")

    for row in costing.by_model(hours):
        out.add("localmama_model_cost_per_call_usd", row["cost_per_call_usd"],
                provider=row["provider"], model=row["model"],
                help_="Average cost of a call on each model, USD.")

    for metric, description in (
        ("turn_gap", "Silence the caller sat through before hearing a reply."),
        ("ttft", "Model's share of that gap: time to its first audio token."),
    ):
        for quantile in ("p50", "p95"):
            out.add(f"localmama_{metric}_seconds", (lag.get(metric) or {}).get(quantile),
                    quantile=quantile, help_=description)
    out.add("localmama_cancelled_responses", lag.get("cancelled_responses"),
            help_="Barged-in responses. Rising means turn detection firing on noise.")

    for name, value in gaps.items():
        out.add(f"localmama_backlog_{name}", value,
                help_="Work owed but not done. Silent by nature — nothing errors.")

    # Whether the numbers above still mean anything. A provider with no rate
    # card row costs nothing on every chart, which is the failure mode this
    # whole subsystem is built to make loud.
    for name in ("unmetered_calls", "unpriced_units", "placeholder_units",
                 "estimated_units", "inherited_units"):
        out.add(f"localmama_coverage_{name}", cover.get(name),
                help_="Gaps in cost coverage. Non-zero means the totals understate.")
    out.add("localmama_coverage_confidence_pct", cover.get("confidence_pct"),
            help_="Share of metered quantities carrying a confirmed rate.")
    return out.render()


@router.get("/metrics", dependencies=[Depends(authorise_ops)],
            response_class=PlainTextResponse)
async def metrics(hours: int = Query(1, ge=1)) -> PlainTextResponse:
    """Prometheus exposition.

    A one-hour default window rather than `costing`'s twenty-four: a scrape is
    a question about now, and a day-wide average moves so slowly that an alert
    built on it fires long after anyone could have acted.
    """
    return PlainTextResponse(
        _exposition(_window(hours)),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )

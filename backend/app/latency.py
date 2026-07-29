"""Where the caller's wait actually goes.

    python -m backend.app.latency

Reads the per-turn metrics collected during real calls and prints the
distribution per stage. The point is the p95 column: tuning has so far been
driven by whichever call someone happened to listen to, which finds a stall bad
enough to hear and misses "most calls are fine, some are slow" entirely.

Every number here comes from LiveKit's own instrumentation of real traffic, not
from a stopwatch around a synthetic request.
"""

from __future__ import annotations

from .logger import setup_logging
from .services import metrics_store

#: What to do about a slow stage. Keeping the remedy next to the measurement is
#: what makes this a loop rather than a dashboard.
LEVERS = {
    "EOUMetrics": "ENDPOINT_MAX_DELAY / ENDPOINTING_MODE — how long we wait before answering",
    "RealtimeModelMetrics": "system prompt size, or OPENAI_REALTIME_MODEL",
    "LLMMetrics": "PHRASING_MODEL, or prefetch hit rate (see conversation_manager)",
    "TTSMetrics": "TTS_PROVIDER / SARVAM_* — Sarvam streams, OpenAI does not",
    "STTMetrics": "OPENAI_STT_MODEL — mini is faster and weaker on Indic",
    "InterruptionMetrics": "INTERRUPT_MIN_WORDS / BACKCHANNEL_BOUNDARY — noise cutting Mami off",
    "TurnGap": "end to end, and it INCLUDES Mami speaking. If this is slow while ttft is fast, the reply is too long — shorten what she says before touching any timing setting",
}


def main() -> None:
    setup_logging()
    if not metrics_store.available():
        print("DATABASE_URL is not set, so no metrics have been collected.")
        return

    rows = metrics_store.summary()
    if not rows:
        print(
            "No metrics recorded yet.\n"
            "They are written when a call ends, so make one call and re-run this."
        )
        return

    print(f"{'stage':22} {'n':>5} {'avg':>7} {'p50':>7} {'p95':>7} {'max':>7}")
    print("-" * 62)
    for r in rows:
        print(
            f"{r['stage']:22} {r['samples']:>5} {r['avg']:>7.2f} "
            f"{r['p50']:>7.2f} {r['p95']:>7.2f} {r['max']:>7.2f}"
        )

    print("\nwhat each stage measures, and the lever that moves it:")
    for r in rows:
        print(f"  {r['stage']}")
        print(f"      measures: {r['measures']}")
        print(f"      lever   : {LEVERS.get(r['stage'], '—')}")

    worst = max(
        (r for r in rows if r["stage"] != "InterruptionMetrics"),
        key=lambda r: r["p95"], default=None,
    )
    if worst:
        print(
            f"\nSlowest at p95: {worst['stage']} at {worst['p95']:.2f}s "
            f"(median {worst['p50']:.2f}s).\n"
            f"Start there: {LEVERS.get(worst['stage'], '—')}"
        )
        if worst["p95"] > worst["p50"] * 2:
            print(
                "  Note the gap between p50 and p95 — this stage is *inconsistent* "
                "rather than uniformly slow, which is what an intermittent delay "
                "looks like from the inside."
            )


if __name__ == "__main__":
    main()

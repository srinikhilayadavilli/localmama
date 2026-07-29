"""Per-turn call metrics, persisted for analysis.

Tuning this pipeline has so far meant reading one call's logs and guessing
which stage was slow. That works for a stall obvious enough to hear, and not at
all for "most calls are fine but some are slow" — the interesting cases are
distributional, and a log tail cannot show a distribution.

LiveKit already computes the numbers. `AgentSession` emits a metrics event per
turn carrying, depending on the stage:

  EOUMetrics           end_of_utterance_delay — how long after the caller stops
                       speaking before the turn is considered over. This is the
                       endpointing setting, measured rather than assumed.
  RealtimeModelMetrics ttft — the model's time to first token. On the
                       speech-to-speech path this IS the caller's wait.
  TTSMetrics           ttfb — time to first audio byte.
  InterruptionMetrics  num_interruptions / num_backchannels — how often Mami
                       was cut off, and how much of that was mere backchannel.

Metrics are buffered in memory for the duration of the call and written once at
the end. Writing per turn would put a database round trip inside the audio path
to measure the audio path, which is self-defeating.

Nothing here may affect a call: every failure is logged and swallowed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any

from ..config import settings
from ..logger import get_logger

logger = get_logger("localmama.metrics")

_DDL = """
CREATE SCHEMA IF NOT EXISTS localmama;
CREATE TABLE IF NOT EXISTS localmama.turn_metrics (
    id          BIGSERIAL PRIMARY KEY,
    session_id  TEXT NOT NULL,
    room        TEXT,
    agent_id    TEXT NOT NULL,
    kind        TEXT NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- The fields worth querying are lifted out; the rest stays in `raw` so a
    -- new SDK field is captured without a migration.
    eou_delay          DOUBLE PRECISION,
    transcription_delay DOUBLE PRECISION,
    ttft               DOUBLE PRECISION,
    ttfb               DOUBLE PRECISION,
    duration           DOUBLE PRECISION,
    audio_duration     DOUBLE PRECISION,
    num_interruptions  INTEGER,
    num_backchannels   INTEGER,
    raw         JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_turn_metrics_session ON localmama.turn_metrics(session_id);
CREATE INDEX IF NOT EXISTS idx_turn_metrics_kind    ON localmama.turn_metrics(agent_id, kind, recorded_at DESC);
"""

_schema_ready = False


def available() -> bool:
    return bool(settings.database_url)


def _as_dict(metric: Any) -> dict:
    """LiveKit metrics are dataclasses in some versions and pydantic in others."""
    for attempt in (
        lambda: asdict(metric) if is_dataclass(metric) else None,
        lambda: metric.model_dump(mode="json"),
        lambda: dict(vars(metric)),
    ):
        try:
            got = attempt()
            if got:
                return {k: v for k, v in got.items() if not k.startswith("_")}
        except Exception:  # noqa: BLE001
            continue
    return {}


def to_row(metric: Any) -> dict:
    """One metrics event flattened into the columns we query on.

    Negative values are dropped rather than stored. LiveKit uses -1 as "not
    measurable" — a cancelled or interrupted response has no time-to-first-
    token — and averaging that as if it were a duration produced a median ttft
    of MINUS one second, which is how the first run of `make latency` reported
    a stage as faster than instantaneous.
    """
    data = _as_dict(metric)

    def num(key: str):
        value = data.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
        return value if value >= 0 else None
    return {
        "kind": type(metric).__name__,
        "eou_delay": num("end_of_utterance_delay"),
        "transcription_delay": num("transcription_delay"),
        "ttft": num("ttft"),
        "ttfb": num("ttfb"),
        "duration": num("duration"),
        "audio_duration": num("audio_duration"),
        "num_interruptions": num("num_interruptions"),
        "num_backchannels": num("num_backchannels"),
        "raw": data,
    }


def flush(session_id: str, room: str, rows: list[dict]) -> int:
    """Write a call's buffered metrics. Returns how many landed."""
    if not available() or not rows:
        return 0
    global _schema_ready
    try:
        import psycopg

        with psycopg.connect(settings.database_url, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                if not _schema_ready:
                    cur.execute(_DDL)
                    conn.commit()
                    _schema_ready = True
                cur.executemany(
                    "INSERT INTO localmama.turn_metrics (session_id, room, agent_id,"
                    " kind, eou_delay, transcription_delay, ttft, ttfb, duration,"
                    " audio_duration, num_interruptions, num_backchannels, raw)"
                    " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    [
                        (
                            session_id, room, settings.brain_agent_id, r["kind"],
                            r["eou_delay"], r["transcription_delay"], r["ttft"],
                            r["ttfb"], r["duration"], r["audio_duration"],
                            r["num_interruptions"], r["num_backchannels"],
                            json.dumps(r["raw"], default=str, ensure_ascii=False),
                        )
                        for r in rows
                    ],
                )
            conn.commit()
        logger.info("stored %d metrics for session=%s", len(rows), session_id[:8])
        return len(rows)
    except Exception as exc:  # noqa: BLE001 - measurement must never break a call
        logger.warning("could not store metrics: %s", exc)
        return 0


#: Which column carries the number worth reading, per metric kind.
_FOCUS = {
    "EOUMetrics": ("eou_delay", "caller stops speaking -> turn ends"),
    "RealtimeModelMetrics": ("ttft", "model time to first token"),
    "LLMMetrics": ("ttft", "LLM time to first token"),
    "TTSMetrics": ("ttfb", "time to first audio byte"),
    "STTMetrics": ("duration", "transcription duration"),
    # Synthetic, and the only one that measures what the caller experiences:
    # everything else times a component. Stored in the ttfb column because it
    # is, in the end, a time to first audible byte.
    # Measured to the assistant item COMPLETING, which includes speaking the
    # reply — so a long answer inflates it. Read it as "how long the caller
    # waited plus how long Mami talked", not as latency alone: a 22s gap on a
    # read-back that recited a full sentence back was mostly the reciting.
    "TurnGap": ("ttfb", "caller stops speaking -> Mami finishes replying (incl. speaking time)"),
}


def summary(limit_calls: int = 50) -> list[dict]:
    """p50/p95 per stage over recent calls — the distribution a log cannot show."""
    if not available():
        return []
    out: list[dict] = []
    try:
        import psycopg

        with psycopg.connect(settings.database_url, connect_timeout=5) as conn, conn.cursor() as cur:
            cur.execute(_DDL)
            conn.commit()
            for kind, (col, meaning) in _FOCUS.items():
                cur.execute(
                    f"SELECT count(*), avg({col}),"
                    f" percentile_cont(0.5) WITHIN GROUP (ORDER BY {col}),"
                    f" percentile_cont(0.95) WITHIN GROUP (ORDER BY {col}),"
                    f" max({col})"
                    " FROM localmama.turn_metrics"
                    f" WHERE agent_id = %s AND kind = %s AND {col} IS NOT NULL",
                    (settings.brain_agent_id, kind),
                )
                n, avg, p50, p95, mx = cur.fetchone()
                if n:
                    out.append({
                        "stage": kind, "measures": meaning, "samples": n,
                        "avg": round(avg or 0, 2), "p50": round(p50 or 0, 2),
                        "p95": round(p95 or 0, 2), "max": round(mx or 0, 2),
                    })
            cur.execute(
                "SELECT sum(num_interruptions), sum(num_backchannels)"
                " FROM localmama.turn_metrics WHERE agent_id = %s",
                (settings.brain_agent_id,),
            )
            interruptions, backchannels = cur.fetchone()
            if interruptions or backchannels:
                out.append({
                    "stage": "InterruptionMetrics",
                    "measures": "times Mami was cut off / times it was only a backchannel",
                    "samples": int(interruptions or 0),
                    "avg": 0, "p50": 0, "p95": 0, "max": int(backchannels or 0),
                })
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not summarise metrics: %s", exc)
    return out

"""What the call consumed, counted while it happens.

The agent already had every number needed to price a call and was throwing all
of them away. `livekit-agents` reports the provider's own `response.usage` —
exact token counts, split by modality and by cached-versus-fresh — and
`worker.py` never subscribed to any of it. Cost was left to be guessed from
call duration, which on a realtime model is the one thing that does not
predict it.

This module accumulates, and hands `call.ended` two things:

  * a **usage ledger** — totals per (provider, model, unit), which the backend
    prices against an effective-dated rate card. Units, never money: see
    `contract.UsageRecord` for why.
  * a **turn series** — one row per model response, which is what tells you
    *which* turn made a call expensive rather than merely that it was.

Nothing here is on the caller's clock: every entry point is a dictionary
update on an event the framework already emits, and every one is wrapped. A
metering bug must cost a cost number, never a phone call.

## Two sources, deliberately

The ledger is built from `session_usage_updated`, which carries the framework's
own cumulative `AgentSessionUsage`. The turn series is built from
`metrics_collected`, which is the only place a *per-response* token count and
time-to-first-token exist.

They are split because `metrics_collected` is deprecated on `AgentSession` in
1.6.7 while `session_usage_updated` is its replacement for usage. Reading
billing off the deprecated one would mean an SDK upgrade silently zeroing every
cost on the dashboard. Reading it off the supported one means an upgrade can at
worst cost the diagnostic series, and `Meter.usage()` keeps returning the truth.
If `metrics_collected` stops firing, `turns()` returns empty and nothing else
changes.

## The subtraction that matters

A realtime model re-bills the whole conversation on every response. Most of
that is served from cache, and cached audio is priced at $0.40 per million
against $32 for fresh — 80x. Every provider, and the framework's own
`ModelUsageCollector`, reports the *total* with the cached count as a nested
subset of it. Adding the two and billing each at its own rate overstates a call
by up to eighty times, so `_ledger` subtracts. This is the single piece of
arithmetic that decides whether the numbers on the dashboard are real.
"""

from __future__ import annotations

import time

from contract import TurnMetric, Unit, UsageRecord

from .logger import get_logger

logger = get_logger("localmama.metering")

#: Cap on the per-response series shipped with `call.ended`. A call is a few
#: dozen turns; a model stuck in a loop is thousands, and the payload that
#: closes the record must not become the thing that fails to send. Totals are
#: unaffected — the ledger comes from a different source and keeps counting.
MAX_TURNS = 500

#: LiveKit bills the agent session and the SIP leg per minute, and neither
#: appears in any usage event — they are facts about the room, not the model.
_LIVEKIT = "livekit"

#: How many input-audio tokens a realtime model charges for one second of the
#: caller talking. OpenAI encodes inbound audio at one token per 100ms.
#: Used only to size the transcription estimate below.
_INPUT_AUDIO_TOKENS_PER_SECOND = 10.0


#: What the rate card calls each provider, keyed by what the framework reports.
#:
#: `livekit-agents` fills `provider` from the client's base URL for some
#: plugins, so the realtime model arrives as "api.openai.com" while the
#: transcription path — which sets it explicitly — arrives as "openai". The
#: rate card prices "openai". A hostname prices nothing, and an unpriced unit
#: is a real cost reported as zero: call 5467f3fb showed gpt-realtime at
#: $0.0000 with 613 audio output tokens on it.
_CANONICAL_PROVIDER = {
    "api.openai.com": "openai",
    "api.groq.com": "groq",
    "api.deepgram.com": "deepgram",
    "api.cartesia.ai": "cartesia",
    "api.elevenlabs.io": "elevenlabs",
    "api.sarvam.ai": "sarvam",
    "generativelanguage.googleapis.com": "google",
}


def _canonical_provider(raw: str | None) -> str:
    """The rate card's name for a provider, not the SDK's.

    Deliberately a lookup rather than a rule. "Strip the leading api. and the
    TLD" turns generativelanguage.googleapis.com into "googleapis", which is
    not a provider anybody prices. An unmapped hostname is left alone and
    logged: it will read as unpriced on the coverage line, which is the point —
    a costing system that quietly renames things is worse than one that admits
    it does not recognise a name.
    """
    name = (raw or "unknown").strip().lower()[:40]
    if name in _CANONICAL_PROVIDER:
        return _CANONICAL_PROVIDER[name]
    if "." in name and name != "unknown":
        logger.warning(
            "meter: %r looks like a hostname and is not a known provider; its "
            "units will price as unpriced until the map or the rate card names it",
            name,
        )
    return name


def _nonneg(value: float, what: str) -> float:
    """Clamp at zero, loudly.

    Only reachable if a provider reports a cached subset larger than the total
    it is a subset of. That has not been observed; it is guarded because the
    alternative is a negative quantity reaching a rate card, and a call that
    appears to earn money is a number nobody checks twice.
    """
    if value < 0:
        logger.warning("meter: %s came out negative (%s); clamping to 0", what, value)
        return 0.0
    return float(value)


class Meter:
    """One call's consumption. Not thread-safe; the session is single-tasked.

    Built even when there is nothing to meter. An unmetered call reports empty
    lists rather than absent ones, so the backend can tell "this call consumed
    nothing" from "this agent cannot count" — different operational facts, and
    only one of them is a bug.
    """

    def __init__(self, started_mono: float, transcription_model: str = "") -> None:
        self.started_mono = started_mono
        #: The second-decode model, named so the estimate below can be labelled
        #: and priced. Passed in rather than read from settings so this class
        #: stays testable without an environment.
        self.transcription_model = (transcription_model or "").strip()[:80]
        #: (provider, model, operation) -> {unit: quantity}. Replaced wholesale
        #: from each cumulative snapshot, never accumulated — see `on_usage`.
        self._model_units: dict[tuple[str, str, str], dict[Unit, float]] = {}
        #: Room-clock units, which no usage event reports. Kept apart so a
        #: snapshot replacing the model totals cannot wipe them.
        self._room_units: dict[tuple[str, str, str], dict[Unit, float]] = {}
        self._turns: list[TurnMetric] = []
        self._turns_dropped = 0
        #: Responses seen, including cancelled ones. The denominator for
        #: "cost per turn", and cheap to keep.
        self.responses = 0

    # -- the ledger ------------------------------------------------------

    def on_usage(self, usage) -> None:  # noqa: ANN001 - livekit AgentSessionUsage
        """Fold in a cumulative usage snapshot. Never raises.

        The framework emits this after every metric, carrying running totals
        for the whole session — so this REPLACES what it knows rather than
        adding to it. Handling it as a delta would multiply every call by the
        number of turns in it.
        """
        try:
            fresh: dict[tuple[str, str, str], dict[Unit, float]] = {}
            for model_usage in getattr(usage, "model_usage", None) or []:
                self._ledger(model_usage, fresh)
            self._model_units = fresh
        except Exception as exc:  # noqa: BLE001 - a cost number is never worth a call
            logger.warning("meter: could not read a usage snapshot: %s", exc)

    def _ledger(self, u, into: dict) -> None:  # noqa: ANN001 - livekit ModelUsage union
        """One provider/model's totals, converted into billable units.

        Dispatched on the usage object's own `type` discriminator rather than
        by `isinstance`, so a plugin reporting a usage class this version does
        not know about is skipped instead of crashing a live call.
        """
        kind = getattr(u, "type", "")
        provider = _canonical_provider(getattr(u, "provider", ""))
        model = (getattr(u, "model", "") or "").strip()[:80]

        if kind == "llm_usage":
            # Covers the realtime model too: the framework files speech-to-
            # speech usage under LLM, which is why `operation` is derived from
            # whether there is any audio rather than from the type.
            audio = _nonneg(u.input_audio_tokens - u.input_cached_audio_tokens, "fresh audio in")
            text = _nonneg(u.input_text_tokens - u.input_cached_text_tokens, "fresh text in")
            operation = "realtime" if (u.input_audio_tokens or u.output_audio_tokens) else "llm"
            units = {
                Unit.AUDIO_INPUT_TOKENS: audio,
                Unit.AUDIO_INPUT_CACHED_TOKENS: _nonneg(u.input_cached_audio_tokens, "cached audio"),
                Unit.TEXT_INPUT_TOKENS: text,
                Unit.TEXT_INPUT_CACHED_TOKENS: _nonneg(u.input_cached_text_tokens, "cached text"),
                Unit.AUDIO_OUTPUT_TOKENS: _nonneg(u.output_audio_tokens, "audio out"),
                Unit.TEXT_OUTPUT_TOKENS: _nonneg(u.output_text_tokens, "text out"),
            }
            # A provider that reports totals without the modality breakdown
            # would otherwise meter as free. Bill the remainder as text, which
            # is the cheaper guess, and say so — silently pricing unknown
            # tokens as audio would inflate the whole dashboard.
            split = sum(units.values())
            if u.input_tokens and not split:
                units[Unit.TEXT_INPUT_TOKENS] = _nonneg(
                    u.input_tokens - u.input_cached_tokens, "fresh input"
                )
                units[Unit.TEXT_INPUT_CACHED_TOKENS] = _nonneg(u.input_cached_tokens, "cached in")
                units[Unit.TEXT_OUTPUT_TOKENS] = _nonneg(u.output_tokens, "output")
                logger.info(
                    "meter: %s/%s reports no modality split; billing %d token(s) as text",
                    provider, model or "?", u.input_tokens + u.output_tokens,
                )
            # Session-based billing (xAI and friends). Zero on OpenAI and
            # Gemini, and `add` drops zeros, so this costs nothing to carry.
            units[Unit.SECONDS] = _nonneg(getattr(u, "session_duration", 0.0), "llm session")

        elif kind == "stt_usage":
            # The independent transcription decode. Easy to forget, because no
            # part of the conversation waits on it — it exists so the backend's
            # audit has a second opinion on what the caller said. It is still
            # billed on every call, and omitting it understates every number
            # here by a fixed margin. Duration and tokens are both recorded
            # because providers differ on which they charge for; the rate card
            # prices whichever one it knows.
            operation = "transcription"
            units = {
                Unit.SECONDS: _nonneg(u.audio_duration, "stt duration"),
                Unit.AUDIO_INPUT_TOKENS: _nonneg(u.input_tokens, "stt in"),
                Unit.TEXT_OUTPUT_TOKENS: _nonneg(u.output_tokens, "stt out"),
            }

        elif kind == "tts_usage":
            # Not on the realtime path. Present so that moving to a cascaded
            # pipeline does not silently stop metering half the bill.
            operation = "tts"
            units = {
                Unit.CHARACTERS: _nonneg(u.characters_count, "tts characters"),
                Unit.SECONDS: _nonneg(u.audio_duration, "tts duration"),
                Unit.TEXT_INPUT_TOKENS: _nonneg(u.input_tokens, "tts in"),
                Unit.AUDIO_OUTPUT_TOKENS: _nonneg(u.output_tokens, "tts out"),
            }
        else:
            # Turn detection and interruption models report request counts.
            # LiveKit does not bill them separately today, so they are not
            # metered — but they are worth knowing about if that changes.
            return

        bucket = into.setdefault((provider, model, operation), {})
        for unit, quantity in units.items():
            if quantity > 0:
                bucket[unit] = bucket.get(unit, 0.0) + quantity

    def note_session(self, seconds: float) -> None:
        """The metered wall clock: the LiveKit agent session and the SIP leg.

        Both are billed per minute by the same vendor and both run for exactly
        as long as the room does, so one duration produces both. Recorded as
        two operations rather than one because a deployment that moves its
        telephony elsewhere re-rates only half of it — and because "the model
        was cheap and the phone line was not" is a conclusion worth being able
        to reach.
        """
        seconds = _nonneg(seconds, "session duration")
        if seconds <= 0:
            return
        for operation in ("session", "telephony"):
            self._room_units[(_LIVEKIT, "", operation)] = {Unit.SECONDS: seconds}

    # -- the turn series -------------------------------------------------

    def on_metrics(self, metrics) -> None:  # noqa: ANN001 - livekit AgentMetrics union
        """Record one model response. Never raises.

        Only `realtime_model_metrics` is kept: it is the sole carrier of a
        per-response token count and time-to-first-token, which is what makes
        the series worth having. Everything else on this channel is already
        counted, better, by `on_usage`.
        """
        try:
            if getattr(metrics, "type", "") != "realtime_model_metrics":
                return
            self.responses += 1
            inp = metrics.input_token_details
            self._add_turn(TurnMetric(
                ref=str(metrics.request_id or f"response-{self.responses}")[:80],
                at=max(0.0, time.monotonic() - self.started_mono),
                # -1 means the response carried no audio. Passed through
                # rather than zeroed: averaging it in as zero would make a
                # text-only turn look like the fastest one on the call.
                ttft=float(metrics.ttft),
                duration=_nonneg(metrics.duration, "response duration"),
                input_tokens=int(_nonneg(metrics.input_tokens, "input tokens")),
                cached_tokens=int(_nonneg(inp.cached_tokens, "cached tokens")),
                output_tokens=int(_nonneg(metrics.output_tokens, "output tokens")),
                cancelled=bool(metrics.cancelled),
            ))
        except Exception as exc:  # noqa: BLE001 - a diagnostic is never worth a call
            logger.warning("meter: could not record a turn: %s", exc)

    def _add_turn(self, turn: TurnMetric) -> None:
        if len(self._turns) >= MAX_TURNS:
            self._turns_dropped += 1
            return
        self._turns.append(turn)

    # -- output ----------------------------------------------------------

    def _estimated_transcription(self) -> list[UsageRecord]:
        """The second decode, which is billed and which nothing reports.

        `input_audio_transcription` on a realtime session is a separate charge
        — a whole extra model running over the caller's audio — and OpenAI does
        report it, on `conversation.item.input_audio_transcription.completed`.
        The LiveKit plugin drops that event's `usage` field on the floor
        (`realtime/realtime_model.py`, the completed handler), so it reaches no
        metric, no usage snapshot, and no dashboard. Left alone it is a charge
        on every single call that appears nowhere.

        Under-reporting a real cost is worse than estimating it and saying so,
        so it is derived: inbound audio is encoded at a fixed ten tokens per
        second, and the *fresh* (uncached) audio input tokens across the call
        are the caller's audio counted exactly once — cached repeats of the
        same audio in later responses are excluded by the subtraction in
        `_ledger`. That makes fresh-tokens/10 the seconds of caller speech,
        which is what transcription bills on.

        Marked `estimated=True` and never mixed silently into a measured total.
        Two things retire it: the plugin forwarding that usage field upstream,
        or a real `stt_usage` appearing — which is why this yields nothing when
        one already has.
        """
        if not self.transcription_model:
            return []
        if any(op == "transcription" for _, _, op in self._model_units):
            return []  # something measured it; never estimate over evidence

        fresh_audio = sum(
            units.get(Unit.AUDIO_INPUT_TOKENS, 0.0)
            for (_, _, operation), units in self._model_units.items()
            if operation == "realtime"
        )
        if fresh_audio <= 0:
            return []
        seconds = fresh_audio / _INPUT_AUDIO_TOKENS_PER_SECOND
        return [UsageRecord(
            provider="openai", model=self.transcription_model,
            operation="transcription", unit=Unit.SECONDS,
            quantity=round(seconds, 4), ref="session", estimated=True,
        )]

    def usage(self) -> list[UsageRecord]:
        """The ledger, as whole-call totals. Safe to call more than once.

        `ref` is "session" throughout: these are totals, not deltas, so a
        retried `call.ended` replaces them rather than doubling them.
        """
        records = []
        for source in (self._model_units, self._room_units):
            for (provider, model, operation), units in source.items():
                for unit, quantity in units.items():
                    records.append(UsageRecord(
                        provider=provider, model=model, operation=operation,
                        unit=unit, quantity=round(quantity, 4), ref="session",
                    ))
        return records + self._estimated_transcription()

    def turns(self) -> list[TurnMetric]:
        if self._turns_dropped:
            logger.warning(
                "meter: %d turn(s) past the %d cap were not recorded; the usage "
                "totals are unaffected", self._turns_dropped, MAX_TURNS,
            )
        return list(self._turns)

    def summary(self) -> str:
        """One line for the log, so a call's cost drivers are visible without
        a database.

        Tokens, not money. The agent has no rate card and should not grow one:
        a price is a fact about the day the call happened and has to stay
        correctable after the fact, which it cannot be once it is printed here.
        """
        parts = []
        for (provider, model, operation), units in (self._model_units | self._room_units).items():
            body = " ".join(
                f"{unit.value.removesuffix('_tokens')}={quantity:,.0f}"
                for unit, quantity in sorted(units.items(), key=lambda kv: kv[0].value)
            )
            parts.append(f"{model or provider}/{operation}[{body}]")
        if not parts:
            return "nothing metered"
        return f"{self.responses} response(s)  " + "  ".join(parts)

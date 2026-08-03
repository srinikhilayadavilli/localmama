"""The meter, and the one subtraction that decides whether it is honest.

The expensive mistake this guards against is not a crash. It is a call that
reports eighty times its real audio-input cost because the cached tokens — a
*subset* of the total, priced at $0.40 against $32 per million — were counted
again at the fresh rate. Nothing about that failure looks wrong on a dashboard;
the number is just large. So most of what is below is arithmetic.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.metering import Meter
from contract import Unit


def _llm_usage(**kw):
    """One `LLMModelUsage` as the framework reports it, cached counts nested
    inside the totals they are part of."""
    base = dict(
        type="llm_usage", provider="openai", model="gpt-realtime",
        input_tokens=0, input_cached_tokens=0,
        input_audio_tokens=0, input_cached_audio_tokens=0,
        input_text_tokens=0, input_cached_text_tokens=0,
        output_tokens=0, output_audio_tokens=0, output_text_tokens=0,
        session_duration=0.0,
    )
    return SimpleNamespace(**{**base, **kw})


def _snapshot(*usages):
    return SimpleNamespace(model_usage=list(usages))


def _units(meter) -> dict:
    """(operation, unit) -> quantity, for the assertions below."""
    return {(r.operation, r.unit): r.quantity for r in meter.usage()}


def test_cached_audio_is_subtracted_from_the_fresh_total():
    """The 80x mistake. `input_audio_tokens` is the whole charge and
    `input_cached_audio_tokens` is the part of it that was cached — billing
    both at their own rates counts the cached portion twice, once at eighty
    times its price."""
    meter = Meter(started_mono=0.0)
    meter.on_usage(_snapshot(_llm_usage(
        input_audio_tokens=60_000, input_cached_audio_tokens=48_000,
    )))
    units = _units(meter)
    assert units[("realtime", Unit.AUDIO_INPUT_TOKENS)] == 12_000
    assert units[("realtime", Unit.AUDIO_INPUT_CACHED_TOKENS)] == 48_000
    # The two must reconstruct the provider's own total, or the invoice and
    # the dashboard are describing different calls.
    total = (units[("realtime", Unit.AUDIO_INPUT_TOKENS)]
             + units[("realtime", Unit.AUDIO_INPUT_CACHED_TOKENS)])
    assert total == 60_000


def test_cached_text_is_subtracted_too():
    """Text input is the system prompt, re-billed on every response and almost
    entirely cached after the first — so the same subtraction applies, and the
    same 10x rate gap ($4 against $0.40) rides on it."""
    meter = Meter(started_mono=0.0)
    meter.on_usage(_snapshot(_llm_usage(
        input_audio_tokens=500,  # any audio at all makes this a realtime turn
        input_text_tokens=9_000, input_cached_text_tokens=8_000,
    )))
    units = _units(meter)
    assert units[("realtime", Unit.TEXT_INPUT_TOKENS)] == 1_000
    assert units[("realtime", Unit.TEXT_INPUT_CACHED_TOKENS)] == 8_000


def test_a_text_only_session_is_not_labelled_realtime():
    """The operation is inferred from whether any audio moved, and it decides
    which rate-card row applies. A cascaded text turn must not be priced as
    speech-to-speech."""
    meter = Meter(started_mono=0.0)
    meter.on_usage(_snapshot(_llm_usage(input_text_tokens=400, output_text_tokens=90)))
    assert {r.operation for r in meter.usage()} == {"llm"}


def test_a_snapshot_replaces_rather_than_accumulates():
    """The framework emits running totals after every metric. Treating them as
    deltas would multiply a call by the number of turns in it — which is a
    plausible-looking number, and wrong by an order of magnitude on a long
    call and by nothing at all on a one-turn one."""
    meter = Meter(started_mono=0.0)
    for total in (1_000, 5_000, 12_000):
        meter.on_usage(_snapshot(_llm_usage(input_audio_tokens=total)))
    assert _units(meter)[("realtime", Unit.AUDIO_INPUT_TOKENS)] == 12_000


def test_a_provider_reporting_no_modality_split_still_bills():
    """A provider that fills in only the totals would otherwise meter as free.
    The remainder is billed as text — the cheaper guess, deliberately, since
    silently pricing unknown tokens as audio would inflate every chart."""
    meter = Meter(started_mono=0.0)
    meter.on_usage(_snapshot(_llm_usage(
        provider="someone", model="mystery-1",
        input_tokens=5_000, input_cached_tokens=1_000, output_tokens=800,
    )))
    units = _units(meter)
    assert units[("llm", Unit.TEXT_INPUT_TOKENS)] == 4_000
    assert units[("llm", Unit.TEXT_INPUT_CACHED_TOKENS)] == 1_000
    assert units[("llm", Unit.TEXT_OUTPUT_TOKENS)] == 800


def test_the_room_clock_bills_the_session_and_the_phone_line_separately():
    """One duration, two charges — LiveKit meters the agent session and the SIP
    leg apart, and a deployment that moves its telephony must be able to
    re-rate only the half that moved."""
    meter = Meter(started_mono=0.0)
    meter.note_session(184.0)
    units = _units(meter)
    assert units[("session", Unit.SECONDS)] == 184.0
    assert units[("telephony", Unit.SECONDS)] == 184.0


def test_a_usage_snapshot_does_not_wipe_the_room_clock():
    """The two live in different stores precisely so a snapshot arriving after
    the room clock was noted cannot replace it with nothing."""
    meter = Meter(started_mono=0.0)
    meter.note_session(120.0)
    meter.on_usage(_snapshot(_llm_usage(input_audio_tokens=100)))
    assert _units(meter)[("session", Unit.SECONDS)] == 120.0


def test_transcription_is_derived_because_nothing_reports_it():
    """OpenAI bills the second decode and the LiveKit plugin discards the usage
    it sends. Derived from fresh audio at ten tokens a second, and flagged, so
    a total that is part inference never passes as wholly measured."""
    meter = Meter(started_mono=0.0, transcription_model="gpt-4o-transcribe")
    meter.on_usage(_snapshot(_llm_usage(
        input_audio_tokens=2_400, input_cached_audio_tokens=600,
    )))
    derived = [r for r in meter.usage() if r.operation == "transcription"]
    assert len(derived) == 1
    # Fresh audio only — 1,800 tokens — because the cached repeats are the same
    # caller audio counted again, and it is transcribed once.
    assert derived[0].quantity == pytest.approx(180.0)
    assert derived[0].estimated is True
    assert derived[0].model == "gpt-4o-transcribe"


def test_a_measured_transcription_suppresses_the_estimate():
    """Never estimate over evidence: a real STT node reporting its own duration
    must win, or the call is billed for transcription twice."""
    meter = Meter(started_mono=0.0, transcription_model="gpt-4o-transcribe")
    meter.on_usage(_snapshot(
        _llm_usage(input_audio_tokens=2_400),
        SimpleNamespace(type="stt_usage", provider="openai",
                        model="gpt-4o-transcribe", input_tokens=0,
                        output_tokens=40, audio_duration=95.0),
    ))
    transcription = [r for r in meter.usage() if r.operation == "transcription"]
    assert [r.quantity for r in transcription if r.unit == Unit.SECONDS] == [95.0]
    assert not any(r.estimated for r in transcription)


def test_nothing_is_estimated_when_no_transcriber_is_configured():
    meter = Meter(started_mono=0.0, transcription_model="")
    meter.on_usage(_snapshot(_llm_usage(input_audio_tokens=2_400)))
    assert not any(r.estimated for r in meter.usage())


def test_a_contradictory_provider_cannot_produce_a_negative_charge():
    """A cached subset larger than its own total has not been observed. If it
    ever is, the quantity must clamp — a negative unit reaching a rate card is
    a call that appears to earn money, which is a number nobody re-checks."""
    meter = Meter(started_mono=0.0)
    meter.on_usage(_snapshot(_llm_usage(
        input_audio_tokens=1_000, input_cached_audio_tokens=4_000,
    )))
    assert all(r.quantity >= 0 for r in meter.usage())


def test_a_broken_metric_never_escapes_into_the_call():
    """Metering is not worth a phone call. A usage object missing the fields
    this expects must be dropped, not raised."""
    meter = Meter(started_mono=0.0)
    meter.on_usage(SimpleNamespace(model_usage=[SimpleNamespace(type="llm_usage")]))
    meter.on_usage(None)
    meter.on_metrics(SimpleNamespace(type="realtime_model_metrics"))
    assert meter.usage() == []


def test_an_unknown_usage_type_is_skipped_rather_than_guessed():
    meter = Meter(started_mono=0.0)
    meter.on_usage(_snapshot(SimpleNamespace(
        type="eot_usage", provider="livekit", model="turn-detector-v1",
        total_requests=12,
    )))
    assert meter.usage() == []


# ── the turn series ─────────────────────────────────────────────────────────


def _response(request_id="resp_1", ttft=0.6, cached=0, total_in=1000, out=300,
              cancelled=False):
    return SimpleNamespace(
        type="realtime_model_metrics", request_id=request_id, ttft=ttft,
        duration=1.8, input_tokens=total_in, output_tokens=out,
        cancelled=cancelled,
        input_token_details=SimpleNamespace(cached_tokens=cached),
    )


def test_the_turn_series_records_context_growth():
    meter = Meter(started_mono=0.0)
    for i, (total, cached) in enumerate([(900, 0), (4200, 3800), (9800, 9100)]):
        meter.on_metrics(_response(f"resp_{i}", total_in=total, cached=cached))
    turns = meter.turns()
    assert [t.input_tokens for t in turns] == [900, 4200, 9800]
    assert [t.cached_tokens for t in turns] == [0, 3800, 9100]
    assert meter.responses == 3


def test_a_response_without_audio_keeps_its_negative_ttft():
    """-1 means "no audio token was sent", which is not zero. Averaged in as
    zero it would make a text-only turn look like the fastest on the call."""
    meter = Meter(started_mono=0.0)
    meter.on_metrics(_response(ttft=-1.0))
    assert meter.turns()[0].ttft == -1.0


def test_a_barged_into_response_is_still_recorded():
    """It was still generated and still billed, and a call full of them is a
    turn-detection problem showing up as a cost problem."""
    meter = Meter(started_mono=0.0)
    meter.on_metrics(_response(cancelled=True))
    assert meter.turns()[0].cancelled is True


def test_the_series_is_capped_without_affecting_the_ledger():
    """A looping model must not produce a `call.ended` too large to send. The
    totals come from a different source and are untouched by the cap."""
    from agent.metering import MAX_TURNS

    meter = Meter(started_mono=0.0)
    for i in range(MAX_TURNS + 25):
        meter.on_metrics(_response(f"resp_{i}"))
    assert len(meter.turns()) == MAX_TURNS
    assert meter.responses == MAX_TURNS + 25


def test_metrics_that_are_not_realtime_responses_are_left_to_the_ledger():
    """Everything else on that channel is already counted, better, by
    `on_usage` — recording it here as well would double it."""
    meter = Meter(started_mono=0.0)
    meter.on_metrics(SimpleNamespace(type="stt_metrics", audio_duration=12.0))
    assert meter.turns() == []
    assert meter.usage() == []

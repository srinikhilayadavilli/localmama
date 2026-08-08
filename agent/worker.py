"""The voice agent. Deployed to LiveKit; talks, and nothing else.

This process answers a phone call, runs a speech-to-speech model, follows
Mami's flow, and captures four things in the caller's own words. Every one of
those is something the caller hears the latency of.

Everything it does *not* do is the point. There is no database here, no vendor
catalogue, no translation, no WhatsApp, no outbox, no migrations, no embedding
model. Captured values leave as raw text over one HTTP client
(`backend_client`) and the backend makes a lead of them after the caller has
hung up. Tools are dictionary writes; the only network call anyone waits on is
a vendor lookup, and that one is hard-bounded and degrades to an apology.

Configuration is fetched from the backend at the start of each call, overlapped
with the wait for the caller to join so it costs nothing. If it has not landed
by then the environment's values are used — a slow config lookup must never be
heard as dead air.

Run:  python -m agent.worker start
"""

from __future__ import annotations

import asyncio
import time

from livekit import agents
from livekit.agents import Agent, AgentServer, AgentSession, JobContext

from contract import CallStatus

from . import telephony
from .backend_client import EventQueue
from .config import settings, startup_problems
from .logger import get_logger, setup_logging
from .metering import Meter
from .prompts.flow import instructions
from .prompts.voice_style import GREETING_INSTRUCTION, GREETING_STT_PROMPT
from .tools import Recorder

logger = get_logger("localmama.agent")

AGENT_VERSION = "2.0"


def _build_openai_model():
    """OpenAI Realtime — the default. Returns None if it cannot be built."""
    if not settings.openai_api_key:
        logger.error("OPENAI_API_KEY is not set — the OpenAI Realtime API cannot start.")
        return None

    from livekit.plugins.openai import realtime
    from openai.types import realtime as realtime_types
    from openai.types.realtime import realtime_audio_input_turn_detection as turn_types

    logger.info(
        "OpenAI Realtime model=%s voice=%s eagerness=%s noise_reduction=%s "
        "speed=%.2f interruptible=%s",
        settings.openai_realtime_model, settings.openai_realtime_voice,
        settings.openai_realtime_eagerness, settings.openai_realtime_noise_reduction,
        settings.openai_realtime_speed, settings.openai_realtime_interrupt,
    )
    try:
        return realtime.RealtimeModel(
            model=settings.openai_realtime_model,
            voice=settings.openai_realtime_voice,
            speed=settings.openai_realtime_speed,
            # Semantic VAD, not plain server VAD: it judges whether the caller
            # has finished the thought rather than merely gone quiet, which is
            # what stops an Indian speaker pausing between clauses from being
            # cut off mid-sentence.
            turn_detection=turn_types.SemanticVad(
                type="semantic_vad",
                eagerness=settings.openai_realtime_eagerness,
                create_response=True,
                interrupt_response=settings.openai_realtime_interrupt,
            ),
            input_audio_noise_reduction=settings.openai_realtime_noise_reduction,
            # A second, independent decode of the same audio the model hears.
            # This is the evidence the backend's audit runs on — two decoders
            # agree on a name the caller said and disagree on an invented one.
            # No `language` is set, so the opening turn can arrive in any of the
            # six.
            input_audio_transcription=realtime_types.AudioTranscription(
                model=settings.openai_stt_model,
                prompt=GREETING_STT_PROMPT,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("could not build the OpenAI Realtime model: %s", exc)
        return None


def _build_gemini_model():
    """Gemini Live — kept for latency comparison."""
    if not settings.google_api_key:
        logger.error("GOOGLE_API_KEY is not set — Gemini Live cannot start.")
        return None

    from google.genai import types as genai_types
    from livekit.plugins.google import realtime

    logger.info(
        "Gemini Live model=%s voice=%s language=%s  silence=%dms padding=%dms end=%s",
        settings.gemini_realtime_model or "(plugin default)", settings.gemini_voice,
        settings.gemini_language or "(auto)", settings.gemini_silence_ms,
        settings.gemini_prefix_padding_ms, settings.gemini_end_sensitivity,
    )
    # Gemini runs its own voice-activity detection server-side. Left at the
    # defaults it waits a long time in silence before answering.
    end_sensitivity = (
        genai_types.EndSensitivity.END_SENSITIVITY_HIGH
        if settings.gemini_end_sensitivity == "high"
        else genai_types.EndSensitivity.END_SENSITIVITY_LOW
    )
    kwargs: dict = {
        "instructions": instructions(),
        "realtime_input_config": genai_types.RealtimeInputConfig(
            automatic_activity_detection=genai_types.AutomaticActivityDetection(
                silence_duration_ms=settings.gemini_silence_ms,
                prefix_padding_ms=settings.gemini_prefix_padding_ms,
                end_of_speech_sensitivity=end_sensitivity,
            )
        ),
        "api_key": settings.google_api_key,
        "voice": settings.gemini_voice,
        # Transcribe both sides — without this a speech-to-speech session is a
        # black box, and the backend has no evidence to audit against.
        "input_audio_transcription": {},
        "output_audio_transcription": {},
    }
    if settings.gemini_realtime_model:
        kwargs["model"] = settings.gemini_realtime_model
    if settings.gemini_language:
        kwargs["language"] = settings.gemini_language

    try:
        return realtime.RealtimeModel(**kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.error("could not build Gemini Live model: %s", exc)
        return None


def build_realtime_model():
    if settings.realtime_provider == "gemini":
        return _build_gemini_model()
    if settings.realtime_provider != "openai":
        logger.warning("unknown REALTIME_PROVIDER=%r, falling back to openai",
                       settings.realtime_provider)
    return _build_openai_model()


async def run_session(ctx: JobContext, participant) -> None:  # noqa: ANN001
    """Run one call."""
    started_mono = time.monotonic()
    events = EventQueue()
    events.start()

    meter = Meter(started_mono, transcription_model=settings.openai_stt_model)
    recorder = Recorder(
        events,
        caller_phone=telephony.caller_id(participant) or None,
        dialled=telephony.dialled_number(participant) or None,
        started_mono=started_mono,
        meter=meter,
    )
    s = recorder.session
    recorder.announce_start(AGENT_VERSION)
    logger.info("call=%s  room=%s  caller=%s", s.short(), ctx.room.name,
                telephony.mask(s.caller_phone or "") or "(anonymous)")

    model = build_realtime_model()
    if model is None:
        recorder.announce_end(CallStatus.ABANDONED)
        await events.drain(settings.drain_seconds)
        return

    # A realtime model handles audio in and out itself: no STT, no TTS.
    # The tools are the ONLY route from speech to a lead.
    session = AgentSession(llm=model)
    agent = Agent(instructions=instructions(), tools=recorder.build_tools())

    # What the call costs. Both handlers are dictionary writes on events the
    # framework already emits, and both swallow their own errors — see
    # `metering.py` for why the ledger and the turn series come from two
    # different channels rather than one.
    @session.on("session_usage_updated")
    def on_usage(event) -> None:  # noqa: ANN001
        meter.on_usage(event.usage)

    # Subscribing to this logs a deprecation warning, once per call. That is
    # expected and the trade is deliberate: this is the only channel carrying a
    # per-response token count and TTFT, which is what identifies WHICH turn
    # made a call expensive. Nothing billable depends on it — if a future SDK
    # drops it, the turn series goes empty and every cost stays correct.
    @session.on("metrics_collected")
    def on_metrics(event) -> None:  # noqa: ANN001
        meter.on_metrics(event.metrics)

    # The most recent utterance the agent began, so the hangup can wait for the
    # *audio* to finish rather than for a state machine to look idle.
    #
    # `agent_state` flickers while a realtime model is still streaming audio
    # out, and polling it hung up four words into the goodbye — "Thank you for
    # choosing" — with the room deleted 1.5s later. `SpeechHandle` is the
    # framework's own answer: `wait_for_playout()` resolves when the caller has
    # actually heard it.
    latest_speech: dict = {}

    @session.on("speech_created")
    def on_speech(event) -> None:  # noqa: ANN001
        handle = getattr(event, "speech_handle", None)
        if handle is not None:
            latest_speech["handle"] = handle
            latest_speech["at"] = time.monotonic()

    last_user_turn: dict = {}

    @session.on("user_input_transcribed")
    def on_user(event) -> None:  # noqa: ANN001
        if event.is_final and event.transcript.strip():
            last_user_turn["at"] = time.monotonic()
            recorder.note_caller_speech(event.transcript)
            logger.info("caller: %s", event.transcript)

    logged_items: set[str] = set()

    @session.on("conversation_item_added")
    def on_item(event) -> None:  # noqa: ANN001
        item = getattr(event, "item", None)
        if item is None or getattr(item, "role", None) != "assistant":
            return
        text = (getattr(item, "text_content", None) or "").strip()
        if not text:
            return
        # The same item fires repeatedly as it streams; log it once.
        item_id = getattr(item, "id", None) or text[:40]
        if item_id in logged_items:
            return
        logged_items.add(item_id)
        recorder.note_agent_speech(text)
        started_at = last_user_turn.pop("at", None)
        if started_at is not None:
            gap = time.monotonic() - started_at
            s.turn_gaps.append(round(gap, 3))
            logger.info("turn gap %.2fs (caller finished -> Mami replied)", gap)
        logger.info("mami: %s", text)
        logger.info("  captured so far: %s", recorder.snapshot())

    # Timed: a live call showed 16s between the session starting and the
    # greeting being spoken, with the caller talking into silence twice while
    # waiting. Which half is slow is not something the vendor's logs answer.
    _t0 = time.monotonic()
    await session.start(agent=agent, room=ctx.room)
    logger.info("session.start took %.2fs", time.monotonic() - _t0)

    # Neither vendor's realtime model speaks first on its own — without this the
    # caller joins to silence. Gemini's latency-optimised models reject
    # generate_reply entirely, so there the caller has to speak first.
    try:
        _t1 = time.monotonic()
        await session.generate_reply(instructions=GREETING_INSTRUCTION)
        logger.info("greeting took %.2fs", time.monotonic() - _t1)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "this model does not support an agent-initiated greeting (%s); "
            "the caller must speak first.", exc,
        )

    async def on_shutdown() -> None:
        # Close the record whatever happened. A caller who gave a name and a
        # service and then hung up is still a lead — arguably the most
        # interesting kind — and the backend cannot process one it was never
        # told had ended.
        recorder.announce_end(
            CallStatus.COMPLETED if s.saved else CallStatus.ABANDONED
        )
        remaining = await events.drain(settings.drain_seconds)
        logger.info(
            "call=%s  ended %s; %d event(s) undelivered",
            s.short(), "complete" if s.saved else "incomplete", remaining,
        )

    ctx.add_shutdown_callback(on_shutdown)

    async def hang_up_when_finished() -> None:
        # The wall-clock ceiling is a backstop for everything that is not a
        # clean finish: a model that loops, a caller who goes silent without
        # hanging up. Both the SIP leg and the realtime session bill by the
        # minute, and neither ends on its own.
        hard_deadline = started_mono + settings.max_call_seconds
        while not s.saved:
            if time.monotonic() >= hard_deadline:
                logger.warning("call=%s  exceeded MAX_CALL_SECONDS=%.0fs; ending it",
                               s.short(), settings.max_call_seconds)
                await _end_room(ctx, s.short())
                return
            await asyncio.sleep(0.5)

        started = await wait_for_outro(lambda: session.agent_state, latest_speech)

        logger.info("call=%s  outro %s; hanging up after %.1fs tail", s.short(),
                    "finished" if started else "never started (timed out)",
                    settings.hangup_grace_seconds)
        # A short tail so the last audio frames reach the caller before the room
        # disappears — ending on the final syllable sounds like a cut-off.
        await asyncio.sleep(settings.hangup_grace_seconds)
        await _end_room(ctx, s.short())

    # Held in a local so the task is not garbage-collected mid-await.
    hangup_task = asyncio.create_task(hang_up_when_finished())

    async def _cancel_hangup() -> None:
        # Must be a coroutine: LiveKit awaits shutdown callbacks, and
        # `task.cancel()` returns a bool — awaiting that raised
        # "TypeError: object bool can't be used in 'await' expression"
        # on every single call, after the caller had already hung up.
        hangup_task.cancel()

    ctx.add_shutdown_callback(_cancel_hangup)


#: How long to wait for the goodbye to begin once the read-back has finished.
#: Not a setting: it is the length of one pause in a conversation, not
#: something a deployment tunes. A model that has already said everything it
#: intends to should not hold a metered line for the whole backstop.
OUTRO_START_WAIT = 4.0


async def wait_for_outro(agent_state, latest_speech: dict, poll: float = 0.2) -> bool:
    """Wait for the closing line to actually be heard. True if it was.

    `save_lead` is called *inside the same turn as the read-back* — the model
    says "...I'll send the details to your WhatsApp" and calls the tool as part
    of the same utterance. So at save time the agent is usually already
    speaking, and that utterance is **not** the goodbye.

    So: let whatever is in flight finish, wait for a *new* utterance to start,
    and wait for that one to play out. A fixed delay cannot work — the same
    goodbye takes longer in Telugu than the English it was timed on.

    **Playout, not `agent_state`.** This has now cut the goodbye off twice.
    First by treating the read-back as the outro; a caller heard two words.
    Then, after that was fixed, by polling `agent_state` for the end of the
    outro: on the realtime path that state flickers while audio is still
    streaming, so the poll after "speaking" saw something else and returned
    instantly. Call 5467f3fb logged "outro finished" at the exact millisecond
    the outro *began*, and the caller heard "Thank you for choosing" before the
    room disappeared.

    `SpeechHandle.wait_for_playout()` is the framework's own answer and it is a
    fact rather than a sample: it resolves when the audio has left. The state
    poll survives only as the fallback for a session that never hands us a
    handle, where being timing-lucky beats hanging up immediately.
    """
    deadline = time.monotonic() + settings.hangup_max_wait_seconds
    at_save = latest_speech.get("at", 0.0)

    # 1. Whatever was in flight at save time — the read-back — finishes.
    in_flight = latest_speech.get("handle")
    if in_flight is not None:
        await _played_out(in_flight, deadline)
    else:
        while time.monotonic() < deadline and agent_state() == "speaking":
            await asyncio.sleep(poll)

    # 2. The goodbye begins: a *new* handle, not the one we just waited on.
    #    Bounded tighter than the backstop — if the model is not going to speak
    #    again, waiting 25s is 25s of dead air on a metered line.
    start_by = min(deadline, time.monotonic() + OUTRO_START_WAIT)
    while time.monotonic() < start_by and latest_speech.get("at", 0.0) <= at_save:
        await asyncio.sleep(poll)
    outro = latest_speech.get("handle")
    if latest_speech.get("at", 0.0) <= at_save or outro is None:
        # No new utterance. Fall back to the old signal in case this session
        # never emits handles at all, rather than assuming silence.
        if agent_state() != "speaking":
            return False
        while time.monotonic() < deadline and agent_state() == "speaking":
            await asyncio.sleep(poll)
        return True

    # 3. And is heard.
    await _played_out(outro, deadline)
    return True


async def _played_out(handle, deadline: float) -> None:
    """Wait for one utterance to reach the caller, bounded by the backstop.

    Never raises: a handle that errors or a framework that changes shape must
    cost the tail of a goodbye, not a call that never hangs up.
    """
    remaining = max(0.0, deadline - time.monotonic())
    try:
        await asyncio.wait_for(handle.wait_for_playout(), timeout=remaining)
    except asyncio.TimeoutError:
        logger.warning("outro still playing after the backstop; ending anyway")
    except Exception as exc:  # noqa: BLE001 - fall through to the hangup
        logger.warning("could not wait for playout (%s); ending anyway", exc)


async def _end_room(ctx: JobContext, sid: str) -> None:
    try:
        await ctx.delete_room()
        logger.info("call=%s  ended by the agent", sid)
    except Exception as exc:  # noqa: BLE001 - the caller can always hang up
        logger.warning("could not end the call: %s", exc)


server = AgentServer()


@server.rtc_session(agent_name=settings.livekit_agent_name)
async def entrypoint(ctx: JobContext) -> None:
    setup_logging()
    await ctx.connect()
    participant = await ctx.wait_for_participant()
    await run_session(ctx, participant)


def main() -> None:
    setup_logging()
    for problem in startup_problems():
        logger.error("STARTUP CHECK: %s", problem)
    agents.cli.run_app(server)


if __name__ == "__main__":
    main()

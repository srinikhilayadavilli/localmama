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
from .prompts.flow import instructions
from .prompts.voice_style import GREETING_STT_PROMPT
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
        "OpenAI Realtime model=%s voice=%s eagerness=%s noise_reduction=%s speed=%.2f",
        settings.openai_realtime_model, settings.openai_realtime_voice,
        settings.openai_realtime_eagerness, settings.openai_realtime_noise_reduction,
        settings.openai_realtime_speed,
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
                interrupt_response=True,
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

    recorder = Recorder(
        events,
        caller_phone=telephony.caller_id(participant) or None,
        dialled=telephony.dialled_number(participant) or None,
        started_mono=started_mono,
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

    # The gap the caller actually feels. Nothing else measures it: the model
    # does its own turn detection and audio, so LiveKit emits no EOU or TTS
    # metrics and the one number we do get — time to first token — says only
    # that the model started promptly, not when the caller heard anything.
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
        await session.generate_reply(instructions="Greet the caller and begin.")
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

        # Wait for the outro, rather than guessing how long it takes.
        #
        # A fixed delay cut the closing line off mid-sentence: save_lead fires
        # BEFORE the model speaks its goodbye, so counting from there means
        # racing a sentence whose length is not known — and in Telugu or Tamil
        # the same words take longer than the English the guess was tuned on.
        deadline = time.monotonic() + settings.hangup_max_wait_seconds
        while time.monotonic() < deadline and session.agent_state != "speaking":
            await asyncio.sleep(0.2)
        started = session.agent_state == "speaking"
        while time.monotonic() < deadline and session.agent_state == "speaking":
            await asyncio.sleep(0.2)

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

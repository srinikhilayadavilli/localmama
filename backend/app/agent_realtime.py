"""Speech-to-speech entrypoint — an EXPERIMENT, not the product.

Runs on the **OpenAI Realtime API** by default (`REALTIME_PROVIDER=openai`),
with Gemini Live kept as an alternative because its latency numbers are
documented in the README and worth keeping comparable.

This is a deliberately separate worker from `agent.py`. Nothing here is shared
with the deterministic pipeline, so the working agent is untouched and you can
run either.

**What this trades away.** `agent.py` speaks only text produced by a state
machine, which is what guarantees that no mandatory field is skipped, that no
promise is made before confirmation, that prompt injection cannot move the
workflow, and that a mistranscription gets caught by the read-back. A realtime
model generates speech directly from audio: fluent and fast, but the model —
not the state machine — decides what to say and when the call is done. Every
one of those guarantees is gone here.

The model is bridged to the engine by function tools (`realtime_tools.py`).
Those tools are the ONLY path from speech to a saved lead, and each one
sanitises and normalises its value the same way the typed pipeline does.
`save_lead` refuses while any mandatory field is missing, so the model cannot
declare a call finished early.

Still weaker than `agent.py`: the *order* of questions and the read-back are
requested in the prompt, not enforced. The model can ask things out of order,
or skip the read-back, and nothing stops it.

The accent is not left to chance either — the persona instructions carry the
same Indian-accent block the TTS path uses (`prompts/voice_style.py`), because
neither vendor ships an Indian voice.

Run:  python -m backend.app.agent_realtime dev
"""

from __future__ import annotations

import asyncio
import time

from livekit import agents
from livekit.agents import Agent, AgentServer, AgentSession, JobContext

from . import telephony
from .config import settings
from .logger import get_logger, setup_logging
from .prompts.voice_style import GREETING_STT_PROMPT, realtime_accent_instructions
from .realtime_tools import LeadRecorder, prefill_from_profile
from .services import metrics_store
from .services.conversation_manager import abandon

logger = get_logger("localmama.realtime")

server = AgentServer()



#: Gemini Live voices. "Puck" is the plugin default; these are worth comparing
#: by ear for an Indian audience.
DEFAULT_VOICE = "Puck"

WORKFLOW_INSTRUCTIONS = """You are Mami, the voice of Local Mama, a service that connects \
people in India with trusted local service providers.

Speak warmly and briefly, like a helpful neighbour — never a call-centre bot. \
One or two short sentences per turn. You are on a phone call, so never use \
markdown, bullet points, or emoji.

Run this conversation in order, one question at a time:
1. Greet: "Welcome to Local Mama! You can call me Mami." Then ask which Indian \
language they would like to speak — English, Hindi, Bengali, Telugu, Tamil, or \
Kannada.
2. When they answer, call set_language, then speak ONLY that language for the \
rest of the call.
3. Ask their name. When they answer, call set_name.
4. Ask what service they need. When they answer, call set_service with their \
own words — do not translate it yourself. Then call lookup_services to check \
we cover it, and mention a matching business by name if one comes back.
5. Ask which city they are in. When they answer, call set_city.
6. Read all three details back and ask if they are correct. If they say no, fix \
the wrong one by calling that tool again, then read back again.
7. ONLY after they confirm, call save_lead. If it reports something missing, \
ask the caller for it and try again.
8. When save_lead succeeds, tell them you will send the details to their \
WhatsApp, thank them for choosing Local Mama, and wish them a good day.

Rules:
- Record every detail by calling the matching tool. A detail you do not record \
is lost — speaking it aloud is not enough.
- Every tool result ends with HELD and STILL NEEDED. That is the truth about \
what has been captured — trust it over your own memory of the conversation. \
NEVER ask again for something listed under HELD; ask only for what is listed \
under STILL NEEDED.
- Correcting one detail changes only that detail. If the caller fixes their \
name, the service and city they already gave still stand — do not re-collect \
them.
- Never invent a name, service, city, price, or phone number. Only use what the \
caller actually said.
- Keep speaking the language the caller chose. Background noise, a stray word, \
or one syllable you are unsure of is NEVER a reason to change language. Only \
change it if the caller clearly asks to, and confirm with them first.
- If you did not clearly hear the caller, say so and ask them to repeat. Do not \
guess, and do not answer noise.
- Do not promise to send anything until save_lead has succeeded.
- If the caller chats or asks something unrelated, answer warmly in one short \
clause, then return to the question you were on."""


def instructions() -> str:
    """Workflow rules plus the Indian-accent block.

    The accent goes in the same system prompt as the workflow because a
    speech-to-speech model has nowhere else to put it: there is no TTS object
    to retune when the caller switches language.
    """
    return f"{WORKFLOW_INSTRUCTIONS}\n\n{realtime_accent_instructions()}"


def _build_openai_model():
    """OpenAI Realtime — the default. Returns None if it cannot be built."""
    if not settings.openai_api_key:
        logger.error(
            "OPENAI_API_KEY is not set — the OpenAI Realtime API cannot start."
        )
        return None

    from livekit.plugins.openai import realtime
    from openai.types import realtime as realtime_types
    from openai.types.realtime import realtime_audio_input_turn_detection as turn_types

    logger.info(
        "OpenAI Realtime model=%s voice=%s eagerness=%s noise_reduction=%s speed=%.2f",
        settings.openai_realtime_model,
        settings.openai_realtime_voice,
        settings.openai_realtime_eagerness,
        settings.openai_realtime_noise_reduction,
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
            # The model hears the audio directly; this transcript is for the
            # logs, without which a speech-to-speech call is a black box. The
            # prompt biases it toward Indian names and the service vocabulary,
            # and no `language` is set so the opening turn can arrive in any of
            # the six.
            input_audio_transcription=realtime_types.AudioTranscription(
                model=settings.openai_stt_model,
                prompt=GREETING_STT_PROMPT,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("could not build the OpenAI Realtime model: %s", exc)
        return None


def _build_gemini_model():
    """Gemini Live — the earlier experiment, kept for comparison."""
    if not settings.google_api_key:
        logger.error(
            "GOOGLE_API_KEY is not set — Gemini Live cannot start. "
            "Get a key at https://aistudio.google.com/"
        )
        return None

    from google.genai import types as genai_types
    from livekit.plugins.google import realtime

    logger.info(
        "Gemini Live model=%s voice=%s language=%s  "
        "turn-taking: silence=%dms padding=%dms end_sensitivity=%s",
        settings.gemini_realtime_model or "(plugin default)",
        settings.gemini_voice,
        settings.gemini_language or "(auto)",
        settings.gemini_silence_ms,
        settings.gemini_prefix_padding_ms,
        settings.gemini_end_sensitivity,
    )

    # Gemini runs its own voice-activity detection server-side. Left at the
    # defaults it waits a long time in silence before answering, which is the
    # "huge gap between turns" — the LiveKit endpointing settings used by
    # agent.py are ignored on this path entirely.
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
        # Transcribe both sides so the conversation is visible in the logs —
        # without this a speech-to-speech session is a black box.
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
    """The realtime model for the configured provider, or None."""
    if settings.realtime_provider == "gemini":
        return _build_gemini_model()
    if settings.realtime_provider != "openai":
        logger.warning(
            "unknown REALTIME_PROVIDER=%r, falling back to openai "
            "(available: openai, gemini)",
            settings.realtime_provider,
        )
    return _build_openai_model()


@server.rtc_session(agent_name=settings.livekit_agent_name)
async def entrypoint(ctx: JobContext) -> None:
    setup_logging()
    await ctx.connect()

    participant = await ctx.wait_for_participant()
    phone = telephony.caller_id(participant)
    recorder = LeadRecorder(caller_id=phone or None)
    prefilled = prefill_from_profile(recorder)
    if prefilled:
        logger.info("session=%s  returning caller, prefilled %s",
                    recorder.session.session_id[:8], ", ".join(prefilled))

    logger.info("session room=%s  provider=%s", ctx.room.name, settings.realtime_provider)
    model = build_realtime_model()
    if model is None:
        return

    # A realtime model handles audio in and out itself: no STT, no TTS, and no
    # deterministic controller in the loop.
    # The tools are the ONLY route from speech to a saved lead.
    session = AgentSession(llm=model)
    agent = Agent(instructions=instructions(), tools=recorder.build_tools())

    # Buffered, not written per turn: a database round trip inside the audio
    # path to measure the audio path would distort the thing being measured.
    turn_metrics: list[dict] = []

    @session.on("metrics_collected")
    def on_metrics(event) -> None:  # noqa: ANN001 - LiveKit event object
        try:
            turn_metrics.append(metrics_store.to_row(getattr(event, "metrics", event)))
        except Exception as exc:  # noqa: BLE001 - never let measurement break a call
            logger.debug("could not record metrics: %s", exc)

    @session.on("user_input_transcribed")
    def on_user(event) -> None:  # noqa: ANN001
        if event.is_final and event.transcript.strip():
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
        logger.info("mami: %s", text)
        logger.info("  captured so far: %s", recorder.snapshot())

    # Timed: a live call showed 16s between the session starting and the
    # greeting being spoken, with the caller talking into silence twice while
    # waiting. Which half is slow — connecting the realtime session, or the
    # model's first response to a ~1200-token system prompt — is not something
    # the vendor's logs answer, so measure both.
    _t0 = time.monotonic()
    await session.start(agent=agent, room=ctx.room)
    logger.info("session.start took %.2fs", time.monotonic() - _t0)

    # Neither vendor's realtime model speaks first on its own — without this
    # the caller joins to silence. (The "Welcome...Welcome..." stutter that
    # looked like a double greeting was a logging artifact, fixed in the
    # handler above.)
    #
    # Gemini's latency-optimised models reject generate_reply entirely, so the
    # greeting has to be skipped there and the caller speaks first — a real
    # trade: much faster turns, no opening line. gpt-realtime accepts it, so
    # the OpenAI path keeps both.
    try:
        _t1 = time.monotonic()
        await session.generate_reply(instructions="Greet the caller and begin.")
        logger.info("greeting generate_reply took %.2fs", time.monotonic() - _t1)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "this model does not support an agent-initiated greeting (%s); "
            "the caller must speak first. Use a native-audio model if you need "
            "Mami to open the call.",
            exc,
        )

    async def on_shutdown() -> None:
        # Written off the event loop, at the end, in one batch.
        if turn_metrics:
            await asyncio.to_thread(
                metrics_store.flush,
                recorder.session.session_id, ctx.room.name, turn_metrics,
            )

        # A call that ends before save_lead still leaves what we learned.
        if not recorder.saved:
            abandon(recorder.session)
            logger.info("session=%s  call ended without save_lead; partial lead stored",
                        recorder.session.session_id[:8])

    ctx.add_shutdown_callback(on_shutdown)

    # Hang up once the lead is saved.
    #
    # Nothing here used to end the call: after the closing line the agent simply
    # sat there, and on a real call the caller waited, said "Thank you", got an
    # unprompted extra reply, and hung up themselves. On a phone line that is
    # both rude and billable — the SIP leg stays up and metered until someone
    # acts.
    #
    # `save_lead` is the only signal that the workflow is genuinely finished,
    # and the grace period lets the model speak its closing before the room
    # disappears mid-sentence.
    async def hang_up_when_finished() -> None:
        while not recorder.saved:
            await asyncio.sleep(0.5)

        # Wait for the outro, rather than guessing how long it takes.
        #
        # A fixed delay cut the closing line off mid-sentence: save_lead fires
        # BEFORE the model speaks its goodbye, so counting from there means
        # racing a sentence whose length is not known — and in Telugu or Tamil
        # the same words take longer than the English the guess was tuned on.
        #
        # So: wait for the goodbye to start, then wait for it to end. The
        # deadline is a backstop for the case where it never speaks at all,
        # which must not leave the caller on a live line forever.
        sid = recorder.session.session_id[:8]
        deadline = time.monotonic() + settings.hangup_max_wait_seconds

        while time.monotonic() < deadline and session.agent_state != "speaking":
            await asyncio.sleep(0.2)
        started = session.agent_state == "speaking"

        while time.monotonic() < deadline and session.agent_state == "speaking":
            await asyncio.sleep(0.2)

        logger.info(
            "session=%s  outro %s; hanging up after %.1fs tail",
            sid,
            "finished" if started else "never started (timed out)",
            settings.hangup_grace_seconds,
        )
        # A short tail so the last audio frames reach the caller before the
        # room disappears — ending on the final syllable sounds like a cut-off.
        await asyncio.sleep(settings.hangup_grace_seconds)
        try:
            await ctx.delete_room()
            logger.info("session=%s  call ended by the agent", recorder.session.session_id[:8])
        except Exception as exc:  # noqa: BLE001 - the caller can always hang up
            logger.warning("could not end the call: %s", exc)

    # Held in a local so the task is not garbage-collected mid-await.
    hangup_task = asyncio.create_task(hang_up_when_finished())

    async def _cancel_hangup() -> None:
        # Must be a coroutine: LiveKit awaits shutdown callbacks, and
        # `task.cancel()` returns a bool — awaiting that raised
        # "TypeError: object bool can't be used in 'await' expression"
        # on every single call, after the caller had already hung up.
        hangup_task.cancel()

    ctx.add_shutdown_callback(_cancel_hangup)


def main() -> None:
    setup_logging()

    if not settings.livekit_configured:
        logger.warning("LiveKit credentials are not all set; the worker cannot connect.")
    if settings.realtime_provider == "gemini":
        if not settings.google_api_key:
            logger.warning("GOOGLE_API_KEY is not set; Gemini Live will not start.")
    elif not settings.openai_api_key:
        logger.warning("OPENAI_API_KEY is not set; the OpenAI Realtime API will not start.")
    agents.cli.run_app(server)


if __name__ == "__main__":
    main()

"""LiveKit Agents entrypoint — the realtime voice path.

Architecture note: `AgentSession` is created *without* an LLM. LiveKit's own
source skips reply generation entirely when `llm is None`, so the session acts
as a pure STT + TTS pipeline and every spoken word comes from our deterministic
controller via `session.say()`. The conversation cannot drift off-script.

Verified against livekit-agents 1.6.7.

Run:  python -m backend.app.agent dev
"""

from __future__ import annotations

import asyncio

from livekit import agents
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    TurnHandlingOptions,
)

from . import session_store, telephony
from .config import settings
from .languages import LOCALE, Language, iso639_1
from .logger import get_logger, setup_logging
from .prompts.agent_instructions import GREETING_INSTRUCTIONS, instructions_for
from .prompts.voice_style import stt_prompt, tts_instructions
from .providers.livekit_plugins import build_stt, build_tts, build_vad
from .services.conversation_manager import abandon

logger = get_logger("localmama.agent")

server = AgentServer()




def _apply_language_to_stt(stt, language: Language) -> None:
    """Pin the recogniser to the language the caller has chosen.

    Only meaningful for the OpenAI path, which starts in auto-detect mode so
    the opening turn can arrive in any language. Detecting on every subsequent
    turn is strictly worse once the language is known: a short "haan" or "sare"
    gives the detector almost nothing to go on and it guesses.

    Providers whose STT cannot retune mid-call (Sarvam) simply have no
    `update_options`, and this is a no-op for them.
    """
    if stt is None or not hasattr(stt, "update_options"):
        return
    try:
        stt.update_options(
            language=iso639_1(language),
            prompt=stt_prompt(language),
        )
        logger.info("STT pinned to %s", language.value)
    except TypeError:
        # Plugin takes neither of those names — leave it as configured.
        logger.info("STT provider cannot switch language at runtime")
    except Exception as exc:  # noqa: BLE001 - a bad value must not end the call
        logger.warning("STT language switch failed: %s", exc)


def _apply_language_to_tts(tts, language: Language) -> None:
    """Best-effort switch of the TTS voice to the caller's language.

    LiveKit cannot replace the TTS object on a live AgentSession, but most
    plugins expose `update_options`. The parameter name is not standardised —
    OpenAI has no language parameter at all and takes free-text `instructions`,
    Sarvam takes `target_language_code`, others take `language` (as a locale or
    a bare ISO-639-1 code), Google-style plugins take `language_code`. We probe
    the known shapes and keep the first that is accepted.

    Verified against livekit-plugins-openai 1.6.7 (TTS accepts `instructions`,
    STT accepts `language` + `prompt`) and livekit-plugins-sarvam 1.6.7 (TTS
    accepts `target_language_code`, STT has no `update_options` at all).

    If nothing is accepted we keep the original voice: audible as a wrong
    accent, not a broken call.
    """
    if tts is None or not hasattr(tts, "update_options"):
        logger.info("TTS provider cannot switch language at runtime")
        return

    # OpenAI is handled separately rather than as another probe candidate: a
    # plugin that swallows unknown kwargs would accept `instructions`, do
    # nothing with it, and stop us from trying the name it actually wants.
    if settings.tts_provider == "openai":
        if settings.openai_tts_instructions:
            logger.info(
                "OPENAI_TTS_INSTRUCTIONS is set, so the voice stays as "
                "configured for %s",
                language.value,
            )
            return
        try:
            tts.update_options(instructions=tts_instructions(language))
            logger.info("TTS switched to %s via instructions", language.value)
        except Exception as exc:  # noqa: BLE001
            logger.warning("TTS language switch failed (instructions): %s", exc)
        return

    locale = LOCALE[language]              # e.g. "te-IN"
    iso2 = iso639_1(language)              # e.g. "te"
    candidates = (
        {"target_language_code": locale},  # sarvam
        {"language": iso2},                # cartesia, elevenlabs
        {"language": locale},
        {"language_code": locale},         # google-style
    )
    for kwargs in candidates:
        try:
            tts.update_options(**kwargs)
            logger.info("TTS switched to %s via %s", language.value, next(iter(kwargs)))
            return
        except TypeError:
            continue  # wrong parameter name for this plugin — try the next
        except Exception as exc:  # noqa: BLE001 - a bad value must not end the call
            logger.warning("TTS language switch failed (%s): %s", next(iter(kwargs)), exc)
            continue
    logger.warning(
        "Could not switch TTS to %s — no known parameter accepted. The call "
        "continues in the startup voice.",
        language.value,
    )


@server.rtc_session(agent_name=settings.livekit_agent_name)
async def entrypoint(ctx: JobContext) -> None:
    setup_logging()
    await ctx.connect()

    # Wait for the caller before building anything: a SIP call carries the
    # caller's number on the participant, and that is what enables the WhatsApp
    # handoff and returning-caller memory. Browser callers simply have none.
    participant = await ctx.wait_for_participant()
    phone = telephony.caller_id(participant)

    manager = session_store.create(caller_id=phone or None)
    session_id = manager.session.session_id
    logger.info(
        "session=%s  room=%s  caller=%s  starting",
        session_id[:8], ctx.room.name, telephony.mask(phone) or "(anonymous)",
    )

    default_language = Language.ENGLISH
    stt = build_stt(default_language)
    tts = build_tts(default_language)
    vad = build_vad()

    if tts is None:
        logger.error(
            "No TTS provider available — the agent cannot speak. "
            "Set TTS_PROVIDER in .env and install the matching plugin."
        )
        return

    session_kwargs: dict = {}
    if stt is not None:
        session_kwargs["stt"] = stt
    if vad is not None:
        session_kwargs["vad"] = vad

    # Natural, uninterrupted turn-taking.
    #
    # Plain VAD ends a turn on silence, which cuts people off mid-sentence —
    # especially in Indian languages, where speakers pause between clauses and
    # code-switch into English mid-utterance. A semantic end-of-turn model
    # decides whether the caller has actually finished, not merely gone quiet.
    # It runs via LiveKit Cloud inference, so it needs LiveKit credentials;
    # without them we fall back to VAD rather than failing the call.
    #
    # These are grouped into TurnHandlingOptions rather than passed as loose
    # AgentSession kwargs, which are deprecated and removed in agents v2.0.
    try:
        from livekit.agents import inference
        from livekit.agents.voice.turn import EndpointingOptions, InterruptionOptions

        session_kwargs["turn_handling"] = TurnHandlingOptions(
            turn_detection=inference.TurnDetector(),
            endpointing=EndpointingOptions(
                mode=settings.endpointing_mode,
                min_delay=settings.endpoint_min_delay,
                max_delay=settings.endpoint_max_delay,
            ),
            interruption=InterruptionOptions(
                enabled=True,     # the caller can talk over Mami
                min_duration=settings.interrupt_min_duration,
                # Without this the count is 0 and duration alone decides, so
                # any 0.4s of sound stops her — including her own audio echoing
                # back off the caller's speakers.
                min_words=settings.interrupt_min_words,
                # A "haan" while she is still talking is agreement, not a
                # request to stop.
                backchannel_boundary=(
                    settings.backchannel_boundary,
                    settings.backchannel_boundary,
                ),
                # If the interruption produced no speech after all, pick the
                # sentence back up instead of leaving the caller in silence
                # wondering whether she is still there.
                resume_false_interruption=True,
            ),
        )
        logger.info(
            "semantic turn detection enabled (endpointing %s %.2f-%.2fs, "
            "interrupt after %.2fs and %d word(s), backchannel window %.1fs)",
            settings.endpointing_mode,
            settings.endpoint_min_delay,
            settings.endpoint_max_delay,
            settings.interrupt_min_duration,
            settings.interrupt_min_words,
            settings.backchannel_boundary,
        )
    except Exception as exc:  # noqa: BLE001 - degrade to VAD endpointing
        logger.warning("semantic turn detection unavailable (%s); using VAD", exc)

    # Intentionally no `llm=`: the workflow is deterministic.
    session = AgentSession(tts=tts, **session_kwargs)

    agent = Agent(instructions=GREETING_INSTRUCTIONS)

    # Serialise turn handling so two rapid finals cannot interleave and commit
    # fields out of order.
    turn_lock = asyncio.Lock()
    last_language = default_language

    async def handle_utterance(text: str) -> None:
        nonlocal last_language
        async with turn_lock:
            result = await manager.handle(text)

            if result.language is not last_language:
                _apply_language_to_stt(stt, result.language)
                _apply_language_to_tts(tts, result.language)
                last_language = result.language
                await agent.update_instructions(instructions_for(result.language))

            await session.say(result.reply)

            if result.is_final:
                logger.info("session=%s  call complete", session_id[:8])
                session_store.drop(session_id)

    #: Strong references to in-flight turn tasks.
    #:
    #: asyncio only keeps a *weak* reference to a task, so one created and left
    #: unreferenced can be garbage-collected mid-await. The turn then stops
    #: wherever it happened to be — after the state machine has committed the
    #: field and logged the transition, but before `session.say()` — and the
    #: caller hears nothing at all. Nothing is raised and nothing is logged,
    #: which is what made "after saying the name nothing happens" so hard to
    #: see: the logs show a reply being produced and no error anywhere.
    pending_turns: set[asyncio.Task] = set()

    def _turn_finished(task: asyncio.Task) -> None:
        pending_turns.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            # Previously this exception went nowhere: the task result was never
            # retrieved, so the caller just got silence.
            logger.exception(
                "session=%s  turn failed; caller heard nothing",
                session_id[:8],
                exc_info=exc,
            )

    @session.on("user_input_transcribed")
    def on_transcript(event) -> None:  # noqa: ANN001 - LiveKit event object
        if not event.is_final or not event.transcript.strip():
            return
        logger.info("session=%s  user: %s", session_id[:8], event.transcript)
        task = asyncio.create_task(handle_utterance(event.transcript))
        pending_turns.add(task)
        task.add_done_callback(_turn_finished)

    await session.start(agent=agent, room=ctx.room)

    # Speak the greeting only after the session is live, so the caller hears it.
    opening = manager.start()
    await session.say(opening.reply)

    async def on_shutdown() -> None:
        abandon(manager.session)
        session_store.drop(session_id)

    ctx.add_shutdown_callback(on_shutdown)


def main() -> None:
    setup_logging()

    if not settings.livekit_configured:
        logger.warning(
            "LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET are not all set — "
            "the worker will fail to connect. See README 'LiveKit voice path'."
        )

    # Prove the voice works before taking calls. Without this the worker starts
    # happily, registers, accepts a job, and only then discovers it has no TTS —
    # so every caller joins and hears silence, and the only clue is one line
    # buried in the per-call logs. A provider named in .env but absent from the
    # image (a plugin left out of requirements.txt) fails exactly this way.
    if build_tts(Language.ENGLISH) is None:
        logger.error(
            "STARTUP CHECK FAILED: TTS_PROVIDER=%r produced no usable TTS, so "
            "this worker would answer every call with silence. Check that its "
            "plugin is installed in this environment and its API key is set.",
            settings.tts_provider,
        )

    agents.cli.run_app(server)


if __name__ == "__main__":
    main()

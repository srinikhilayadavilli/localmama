"""When the agent is allowed to hang up.

Written because a real caller lost their goodbye — twice, for two different
reasons, which is why this waits on audio now rather than on a state.

**First** (`save_lead` is called inside the same turn as the read-back, so the
agent is already speaking when the hang-up logic starts watching): the original
latched onto that utterance, waited for it to end, and killed the room a second
and a half later. The caller heard two words of the outro.

**Then**, after that was fixed by waiting for a *new* utterance: the end of the
outro was detected by polling `agent_state`, which flickers on the realtime
path while audio is still streaming out. Call `5467f3fb` logged
"outro finished" at the exact millisecond the outro began, and the caller heard
"Thank you for choosing" before the line dropped.

A state you sample can be sampled at the wrong moment. Playout is a fact.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from agent import worker


class Speech:
    """A scripted agent_state, advancing one step per poll."""

    def __init__(self, *states: str) -> None:
        self.states = list(states)
        self.polls = 0

    def __call__(self) -> str:
        state = self.states[min(self.polls, len(self.states) - 1)]
        self.polls += 1
        return state


class Handle:
    """A SpeechHandle whose audio takes `plays_for` seconds to reach the caller."""

    def __init__(self, plays_for: float = 0.05) -> None:
        self.plays_for = plays_for
        self.awaited = False

    async def wait_for_playout(self) -> None:
        self.awaited = True
        await asyncio.sleep(self.plays_for)


def _speech(handle, at: float | None = None) -> dict:
    return {"handle": handle, "at": at if at is not None else time.monotonic()}


# --- the regression that prompted this ------------------------------------


@pytest.mark.asyncio
async def test_the_outro_is_waited_for_even_when_the_state_says_idle():
    """The bug, exactly. `agent_state` reports "listening" from the first poll
    while the goodbye is still playing. Polling it returns instantly and the
    room is deleted four words in; waiting on playout does not."""
    readback, outro = Handle(0.01), Handle(0.12)
    latest = _speech(readback)

    async def _new_utterance() -> None:
        await asyncio.sleep(0.02)
        latest.update(_speech(outro))

    asyncio.create_task(_new_utterance())
    began = time.monotonic()
    # Never reports "speaking" — the state is useless here, which is the point.
    assert await worker.wait_for_outro(Speech("listening"), latest, poll=0.005) is True

    assert outro.awaited, "the goodbye's audio was never waited for"
    assert time.monotonic() - began >= 0.12, "hung up before the outro finished"


@pytest.mark.asyncio
async def test_the_read_back_is_not_mistaken_for_the_goodbye():
    """The first bug. The utterance in flight at save time is the read-back;
    the goodbye is the *next* one, and it must be a different handle."""
    readback, outro = Handle(0.02), Handle(0.05)
    latest = _speech(readback)

    async def _new_utterance() -> None:
        await asyncio.sleep(0.04)
        latest.update(_speech(outro))

    asyncio.create_task(_new_utterance())
    assert await worker.wait_for_outro(Speech("listening"), latest, poll=0.005) is True

    assert readback.awaited and outro.awaited, "both utterances must be waited for"


# --- the line is never held open ------------------------------------------


@pytest.mark.asyncio
async def test_a_model_that_never_says_goodbye_does_not_hold_the_line(monkeypatch):
    """The backstop. A caller must not be left on a live, metered line because
    the model decided it had finished talking."""
    monkeypatch.setattr(worker, "OUTRO_START_WAIT", 0.05)
    latest = _speech(Handle(0.0))

    assert await worker.wait_for_outro(Speech("listening"), latest, poll=0.005) is False


@pytest.mark.asyncio
async def test_a_goodbye_that_never_ends_is_cut_off_eventually(monkeypatch):
    """A stuck utterance holds the line too. `hangup_max_wait_seconds` bounds
    the wait, and a handle that never resolves must not outlive it."""
    from agent.config import settings

    monkeypatch.setattr(worker, "OUTRO_START_WAIT", 1.0)
    object.__setattr__(settings, "hangup_max_wait_seconds", 0.08)
    try:
        latest = _speech(Handle(30.0))   # would play for half a minute
        began = time.monotonic()
        assert await worker.wait_for_outro(Speech("speaking"), latest, poll=0.005) in (True, False)
        assert time.monotonic() - began < 1.0, "the backstop did not bound the wait"
    finally:
        object.__setattr__(settings, "hangup_max_wait_seconds", 25.0)


@pytest.mark.asyncio
async def test_a_handle_that_raises_still_hangs_up(monkeypatch):
    """A framework that changes shape must cost the tail of a goodbye, never a
    call that stays open on a metered line."""
    monkeypatch.setattr(worker, "OUTRO_START_WAIT", 0.05)

    class Broken:
        async def wait_for_playout(self):
            raise RuntimeError("SpeechHandle changed shape")

    latest = _speech(Broken())
    assert await worker.wait_for_outro(Speech("listening"), latest, poll=0.005) is False


# --- the session that emits no handles at all ------------------------------


@pytest.mark.asyncio
async def test_without_handles_it_falls_back_to_the_state():
    """A session that never emits `speech_created` still has to hang up. Being
    timing-lucky beats hanging up immediately."""
    speech = Speech(
        "speaking", "speaking", "speaking",   # read-back, still going
        "listening",                          # pause
        "speaking", "speaking",               # the outro
        "listening",                          # done
    )
    assert await worker.wait_for_outro(speech, {}, poll=0.0) is True
    assert speech.polls > 4


@pytest.mark.asyncio
async def test_without_handles_a_silent_model_still_returns_false(monkeypatch):
    monkeypatch.setattr(worker, "OUTRO_START_WAIT", 0.05)
    assert await worker.wait_for_outro(Speech("listening"), {}, poll=0.0) is False

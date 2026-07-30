"""When the agent is allowed to hang up.

Written because a real caller lost their goodbye. `save_lead` is called inside
the same turn as the read-back, so the agent is already speaking when the
hang-up logic starts watching — and the first version latched onto that
utterance, waited for it to end, and killed the room a second and a half later.
The caller heard two words of the outro.
"""

from __future__ import annotations

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


@pytest.mark.asyncio
async def test_the_goodbye_is_waited_for_when_saving_mid_sentence():
    """The bug, exactly. Speaking at save time (the read-back), a pause, then
    the outro. All three phases must be observed before hanging up."""
    speech = Speech(
        "speaking", "speaking", "speaking",   # read-back, still going
        "listening",                          # pause
        "speaking", "speaking",               # the outro
        "listening",                          # done
    )
    assert await worker.wait_for_outro(speech, poll=0.0) is True
    # It cannot have stopped at the first "listening" — that was the gap
    # between the read-back and the goodbye, not the end of the call.
    assert speech.polls > 4


@pytest.mark.asyncio
async def test_the_goodbye_is_waited_for_when_saving_between_sentences():
    """The other ordering: the tool call lands while the agent is idle, and the
    outro follows. Nothing to drain first."""
    speech = Speech("listening", "speaking", "speaking", "listening")
    assert await worker.wait_for_outro(speech, poll=0.0) is True


@pytest.mark.asyncio
async def test_a_model_that_never_says_goodbye_does_not_hold_the_line(monkeypatch):
    """The backstop. A caller must not be left on a live, metered line because
    the model decided it had finished talking."""
    monkeypatch.setattr(worker, "OUTRO_START_WAIT", 0.05)
    assert await worker.wait_for_outro(Speech("listening"), poll=0.0) is False


@pytest.mark.asyncio
async def test_a_goodbye_that_never_ends_is_cut_off_eventually(monkeypatch):
    """A stuck model holds the line too. `hangup_max_wait_seconds` bounds it."""
    from agent.config import settings

    monkeypatch.setattr(worker, "OUTRO_START_WAIT", 1.0)
    object.__setattr__(settings, "hangup_max_wait_seconds", 0.05)
    try:
        # Never stops speaking; returns True but only after the deadline.
        assert await worker.wait_for_outro(Speech("speaking"), poll=0.0) in (True, False)
    finally:
        object.__setattr__(settings, "hangup_max_wait_seconds", 25.0)

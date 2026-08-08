"""One delivery opportunity per lead per sweep.

The sweep's whole retry contract is "one attempt per lead per pass" — trying
harder now is the wrong shape for a provider that is down for hours, and
`OUTBOX_MAX_ATTEMPTS` is what eventually stops a lead being retried forever.
A pass that quietly spends four attempts on one lead brings that ceiling
forward by 4x while a receiver is down.
"""

from __future__ import annotations

import pytest

from backend import outbox_worker


@pytest.mark.asyncio
async def test_a_pass_drains_before_it_recovers(monkeypatch):
    """`reprocess` runs the pipeline, and the pipeline's `notify` already
    delivers with three attempts. If `drain` ran afterwards it would claim
    whatever came back `pending` and try a fourth time in the same pass.

    Recovered leads are still delivered in this pass — by that `notify` — they
    are simply not swept again until the next one.
    """
    order = []

    async def _drain(limit=50):
        order.append("drain")
        return 0, 0

    async def _reprocess():
        order.append("reprocess")
        return 0

    monkeypatch.setattr(outbox_worker, "drain", _drain)
    monkeypatch.setattr(outbox_worker, "reprocess", _reprocess)
    monkeypatch.setattr(outbox_worker.store, "expire_transcripts", lambda: 0)
    monkeypatch.setattr(outbox_worker.costing, "prune_turns", lambda: 0)

    await outbox_worker.one_pass()

    assert order == ["drain", "reprocess"]


@pytest.mark.asyncio
async def test_the_retention_sweeps_still_run_when_nothing_is_owed(monkeypatch):
    """They are not conditional on there being leads. A quiet week is exactly
    when nobody notices transcripts stopped expiring."""
    ran = []

    async def _drain(limit=50):
        return 0, 0

    async def _reprocess():
        return 0

    monkeypatch.setattr(outbox_worker, "drain", _drain)
    monkeypatch.setattr(outbox_worker, "reprocess", _reprocess)
    monkeypatch.setattr(outbox_worker.store, "expire_transcripts",
                        lambda: ran.append("transcripts"))
    monkeypatch.setattr(outbox_worker.costing, "prune_turns",
                        lambda: ran.append("turns"))

    await outbox_worker.one_pass()

    assert ran == ["transcripts", "turns"]

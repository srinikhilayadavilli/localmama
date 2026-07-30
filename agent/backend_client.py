"""The agent's only link to the outside world.

Everything the agent knows about the business — vendors, leads, WhatsApp,
Postgres — is behind this one HTTP client. That is the whole point of the
split: this process holds a phone call, and a phone call is a hard real-time
obligation that a database round trip has no business being inside.

Two very different kinds of call live here, and they get very different
treatment.

**Events are fire-and-forget.** A capture is queued and flushed by a background
task; no tool ever awaits one. If the backend is slow, unreachable, or
restarting, the queue holds and retries — the caller hears nothing either way.
Events are idempotent by `event_id`, so retrying is always safe and the client
never has to reason about whether a send landed.

**A vendor lookup is the one thing a caller waits on.** It gets a hard, short
timeout and degrades to "I can't look that up right now" rather than to dead
air. Better to admit it than to leave someone listening to silence.

Nothing here raises into a call. A backend that is entirely down costs the lead
its enrichment, not the conversation.
"""

from __future__ import annotations

import asyncio
from collections import deque

import httpx

from contract import SCHEMA_VERSION, Event, Language, MatchKind, VendorReply

from .config import settings
from .logger import get_logger

logger = get_logger("localmama.backend")

#: How long the caller may wait on a directory lookup before we give up and say
#: so. The realtime model is holding the turn open for the whole of this.
VENDOR_TIMEOUT = 0.8

#: Events are not on anyone's clock, so this is about not stacking up
#: connections rather than about latency.
EVENT_TIMEOUT = 10.0

#: How long the flusher waits before retrying a batch that failed. Backs off to
#: avoid hammering a backend that is restarting, and caps so a long deploy does
#: not push the retry past the end of the call.
_RETRY_BASE = 1.0
_RETRY_MAX = 8.0

#: A call is a few dozen events. Well past that means the backend has been
#: unreachable for the whole call and the queue is not going to drain — keep
#: the newest, because a lost `call.ended` is worse than a lost mid-call
#: capture that the read-back will restate anyway.
MAX_QUEUE = 500


def configured() -> bool:
    return bool(settings.backend_url and settings.backend_token)


def _headers() -> dict:
    return {"Authorization": f"Bearer {settings.backend_token}"}


class EventQueue:
    """Buffers capture events and flushes them in the background.

    One per call. `send()` never blocks and never raises; `drain()` is awaited
    once at shutdown, with a deadline, so a lead is not lost to a process
    exiting a few hundred milliseconds too early.
    """

    def __init__(self) -> None:
        self._pending: deque[Event] = deque(maxlen=MAX_QUEUE)
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._closed = False
        self.dropped = 0

    def start(self) -> None:
        if self._task is None and configured():
            self._task = asyncio.create_task(self._run())

    def send(self, event: Event) -> None:
        """Queue an event. Returns immediately."""
        if not configured():
            logger.debug("no backend configured; dropping %s", event.type)
            return
        if len(self._pending) == self._pending.maxlen:
            self.dropped += 1
            logger.warning("event queue full; dropped the oldest (total %d)", self.dropped)
        self._pending.append(event)
        self._wake.set()

    async def _run(self) -> None:
        delay = _RETRY_BASE
        while not self._closed or self._pending:
            if not self._pending:
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
            batch = list(self._pending)[:100]
            if not batch:
                continue
            if await self._post(batch):
                for _ in batch:
                    self._pending.popleft()
                delay = _RETRY_BASE
            else:
                # Left on the queue deliberately. Every event is idempotent, so
                # a retry that turns out to have landed the first time is a
                # no-op on the backend rather than a duplicate lead.
                await asyncio.sleep(delay)
                delay = min(delay * 2, _RETRY_MAX)

    async def _post(self, batch: list[Event]) -> bool:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "events": [e.model_dump(mode="json") for e in batch],
        }
        try:
            async with httpx.AsyncClient(timeout=EVENT_TIMEOUT) as client:
                res = await client.post(
                    f"{settings.backend_url}/v1/events",
                    json=payload, headers=_headers(),
                )
            if res.status_code == 202:
                body = res.json()
                if body.get("rejected"):
                    # The backend will never accept these, however many times we
                    # try. Loud, because it means the two sides disagree about
                    # the contract — which is a deploy problem, not a call one.
                    logger.error("backend rejected %d event(s): %s",
                                 len(body["rejected"]), body["rejected"])
                return True
            if res.status_code == 422:
                logger.error("backend refused a malformed batch (422): %s",
                             res.text[:300])
                return True          # retrying will not help; drop it
            logger.warning("backend returned %d for %d event(s); will retry",
                           res.status_code, len(batch))
            return False
        except Exception as exc:  # noqa: BLE001 - never raise into a call
            logger.warning("could not reach the backend (%s); will retry", exc)
            return False

    async def drain(self, timeout: float) -> int:
        """Flush what is left. Returns how many events did not make it.

        Awaited at shutdown. Anything still queued when the deadline passes is
        genuinely lost — which is why `call.captured` is sent as it happens
        rather than batched up for the end of the call.
        """
        self._closed = True
        self._wake.set()
        if self._task is None:
            return len(self._pending)
        try:
            await asyncio.wait_for(asyncio.shield(self._task), timeout)
        except asyncio.TimeoutError:
            self._task.cancel()
        except Exception as exc:  # noqa: BLE001
            logger.warning("event flush ended badly: %s", exc)
        remaining = len(self._pending)
        if remaining:
            logger.error(
                "%d event(s) never reached the backend; this call's lead is "
                "incomplete", remaining,
            )
        return remaining


async def lookup_vendor(
    name: str, city: str | None, language: Language, call_id: str = ""
) -> VendorReply:
    """Ask the directory about a business. Never raises.

    Degrades to an honest "I can't look that up" — a caller told that can ask
    for something else, while a caller listening to silence hangs up.

    `call_id` ties the question to the call so the business reaches their
    WhatsApp too. A number read aloud on a phone line is easily lost; the one
    they explicitly asked for is the last thing that should be missing from
    the written handoff.
    """
    unavailable = VendorReply(
        match=MatchKind.NONE,
        say="I can't look that up right now.",
        instruction="Say so, and offer to have the team follow up.",
    )
    if not configured():
        return unavailable
    try:
        async with httpx.AsyncClient(timeout=VENDOR_TIMEOUT) as client:
            res = await client.get(
                f"{settings.backend_url}/v1/vendors",
                params={"name": name, "city": city or "",
                        "language": language.value, "call_id": call_id},
                headers=_headers(),
            )
        res.raise_for_status()
        return VendorReply.model_validate(res.json())
    except Exception as exc:  # noqa: BLE001 - a lookup must never end a call
        logger.warning("vendor lookup failed (%s); telling the caller so", exc)
        return unavailable

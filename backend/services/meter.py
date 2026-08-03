"""Metering for the provider calls the backend makes on a lead's behalf.

The agent meters itself, because only it can see the model's usage. Everything
after the caller hangs up — the Sarvam round trips that turn "కార్ వాష్" into
"car wash", the WhatsApp send — is billed too, and happens here, out of the
agent's sight. Left unmetered, the per-call number on the dashboard would be
the agent's half of the bill presented as the whole of it.

One decorator-free helper, `metered`, wraps a provider call and records what it
cost and how it went. It is deliberately a context manager rather than a
wrapper function: the quantity usually is not known until the call has been
made — a translation bills on the characters actually sent, and a send that
fails on a bad number sends nothing — so the body sets it.

Nothing here can fail its caller. The lead is durable long before any of this
runs, and a metering bug must never become a lead that did not go out.
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from ..logger import get_logger
from ..store import record_service_call

logger = get_logger("localmama.meter")

#: Which call the work in flight belongs to.
#:
#: A context variable rather than an argument threaded through every signature.
#: `translate.english_name` takes a string and returns a string, and it should
#: go on doing that — a lead's identity is not something a transliterator needs
#: to know about, and adding it to six signatures to satisfy the accountant
#: would put metering into the shape of the code it is measuring.
#:
#: Correct under asyncio: each task gets a copy of the context at creation, so
#: two leads processed concurrently cannot see each other's call_id. Unset
#: outside a `for_call` block, and unattributed work is simply not metered.
_current_call: ContextVar[str | None] = ContextVar("localmama_call_id", default=None)


@contextmanager
def for_call(call_id: str | None):
    """Attribute every metered call made inside this block to `call_id`."""
    token = _current_call.set(call_id)
    try:
        yield
    finally:
        _current_call.reset(token)


def current_call() -> str | None:
    return _current_call.get()


@dataclass
class Measurement:
    """The mutable handle a metered block fills in.

    Starts at zero and not-ok, which is the honest default: a block that raises
    before setting anything did attempt the call, and recording that attempt as
    a silent success would hide exactly the failure worth seeing.
    """

    quantity: float = 0.0
    ok: bool = False
    error: str = ""

    def succeeded(self, quantity: float) -> None:
        self.quantity, self.ok, self.error = float(quantity), True, ""

    def failed(self, quantity: float, error: str) -> None:
        self.quantity, self.ok, self.error = float(quantity), False, str(error)


@contextmanager
def metered(
    provider: str, unit: str,
    *, model: str = "", operation: str = "", call_id: str | None = None,
):
    """Record one provider call: its units, its latency, and whether it worked.

    Yields a `Measurement` for the body to fill in. The row is written on the
    way out whatever happened, including on an exception — a provider call that
    blew up still took time and still says something about that provider's
    health, and an exception path that writes nothing is how an outage comes to
    look like an idle afternoon.

    A fresh `ref` per call, so several translations of one lead each get a row
    rather than overwriting one another.

    `call_id` defaults to whatever `for_call` is scoping; work outside such a
    block is not attributable to a call and is therefore not metered.
    """
    call_id = call_id or current_call()
    measurement = Measurement()
    started = time.monotonic()
    try:
        yield measurement
    except Exception as exc:  # noqa: BLE001 - re-raised below; recorded first
        measurement.failed(measurement.quantity, repr(exc))
        raise
    finally:
        try:
            record_service_call(
                call_id, ref=f"{operation or provider}-{uuid.uuid4().hex[:12]}",
                provider=provider, unit=unit, quantity=measurement.quantity,
                model=model, operation=operation, ok=measurement.ok,
                latency_ms=int((time.monotonic() - started) * 1000),
                error=measurement.error,
            )
        except Exception as exc:  # noqa: BLE001 - never surface out of a finally
            logger.warning("could not meter a %s call: %s", provider, exc)

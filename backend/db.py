"""One connection pool for the process.

The agent opened a fresh `psycopg.connect()` for every query — a full TLS
handshake to Neon per lookup, and a vendor lookup could need four to six of
them while a caller was on the line. Nothing here is on a caller's clock any
more, but a pool is still the difference between a lead landing in 40ms and in
half a second, and it is what keeps Neon's connection ceiling out of reach when
the outbox sweeps.

Opened lazily so importing this module never touches the network: the test
suite, the migration runner and `--help` all import it without a database.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager

from .config import settings
from .logger import get_logger

logger = get_logger("localmama.db")

_pool = None
_lock = threading.Lock()


def available() -> bool:
    return bool(settings.database_url)


def pool():
    """The process-wide pool, opened on first use."""
    global _pool
    if _pool is not None:
        return _pool
    with _lock:
        if _pool is None:
            from psycopg_pool import ConnectionPool

            logger.info(
                "opening connection pool (min=%d max=%d)",
                settings.db_pool_min, settings.db_pool_max,
            )
            _pool = ConnectionPool(
                settings.database_url,
                min_size=settings.db_pool_min,
                max_size=settings.db_pool_max,
                # Fail a request rather than queue behind an exhausted pool.
                timeout=10.0,
                # A Neon connection that has been idle through a scale-to-zero
                # is dead but still looks open; this notices before a query does.
                max_idle=300.0,
                open=True,
            )
    return _pool


@contextmanager
def connection():
    """A pooled connection. Commits on clean exit, rolls back on an exception."""
    with pool().connection() as conn:
        yield conn


@contextmanager
def cursor():
    """A pooled cursor, for the common single-statement case."""
    with connection() as conn, conn.cursor() as cur:
        yield cur


def close() -> None:
    """Shut the pool down. Called on service shutdown; safe if never opened."""
    global _pool
    with _lock:
        if _pool is not None:
            _pool.close()
            _pool = None

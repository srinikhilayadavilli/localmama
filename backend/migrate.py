"""Apply pending migrations. Render runs this as a pre-deploy command.

    python -m backend.migrate           # apply what is pending
    python -m backend.migrate --status  # report without applying

Files in `migrations/` are applied in filename order, once each, tracked in
`localmama.schema_migrations`. Each runs in its own transaction, so a failure
leaves everything before it applied and everything after it pending.

This exists because the agent used to issue DDL lazily from inside a live
call's save path. Migrations belong at deploy time, run once, by one process.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys

from . import db
from .config import settings
from .logger import get_logger, setup_logging

logger = get_logger("localmama.migrate")

MIGRATIONS_DIR = pathlib.Path(__file__).parent / "migrations"

_TRACKING = """
CREATE SCHEMA IF NOT EXISTS localmama;
CREATE TABLE IF NOT EXISTS localmama.schema_migrations (
    filename   TEXT PRIMARY KEY,
    checksum   TEXT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def _checksum(sql: str) -> str:
    return hashlib.sha256(sql.encode("utf-8")).hexdigest()[:16]


def available_migrations() -> list[tuple[str, str]]:
    """(filename, sql) in application order."""
    return [
        (p.name, p.read_text(encoding="utf-8"))
        for p in sorted(MIGRATIONS_DIR.glob("*.sql"))
    ]


def applied() -> dict[str, str]:
    with db.cursor() as cur:
        cur.execute(_TRACKING)
        cur.connection.commit()
        cur.execute("SELECT filename, checksum FROM localmama.schema_migrations")
        return dict(cur.fetchall())


def run(dry_run: bool = False) -> int:
    """Apply pending migrations. Returns how many ran."""
    if not db.available():
        logger.error("DATABASE_URL is not set; nothing to migrate.")
        return -1

    done = applied()
    ran = 0
    for name, sql in available_migrations():
        if name in done:
            # A migration whose contents changed after being applied is an
            # editing accident, not a new migration. Loud, because the database
            # and the repository now disagree about what the schema is.
            if done[name] != _checksum(sql):
                logger.error(
                    "%s was applied with different contents (db=%s, file=%s). "
                    "Add a new migration instead of editing an applied one.",
                    name, done[name], _checksum(sql),
                )
            continue
        if dry_run:
            logger.info("PENDING  %s", name)
            ran += 1
            continue
        logger.info("applying %s", name)
        with db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                cur.execute(
                    "INSERT INTO localmama.schema_migrations (filename, checksum)"
                    " VALUES (%s, %s)",
                    (name, _checksum(sql)),
                )
            conn.commit()
        ran += 1

    logger.info("%s %d migration(s)", "pending:" if dry_run else "applied", ran)
    return ran


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply database migrations.")
    parser.add_argument("--status", action="store_true", help="report without applying")
    args = parser.parse_args()

    setup_logging()
    if not settings.database_url:
        print("DATABASE_URL is not set.")
        return 1
    ran = run(dry_run=args.status)
    db.close()
    return 0 if ran >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())

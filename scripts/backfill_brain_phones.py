"""Copy vendor phone numbers into the brain, making it the single source of truth.

Run with --apply to write; without it this is a dry run.

`utter.knowledge` was always designed to hold vendors — it has `phone`, `city`,
`kind` and `keywords` columns and its own docs describe "one Brain for every
kind of knowledge (vendors/faqs/docs)". The phones simply lived in the
tenant-owned `localmama.businesses` table instead, which meant two stores, two
matching strategies, and a WhatsApp message that named a business without a
number to ring.

Both tables hold the same 120 businesses, matched exactly by name, and the brain
rows already carry the keywords. So `phone` is copied, and `category` with it —
the directory categories were tidied from 59 to 50 and the brain had missed that,
so it still said "AutoMobiles" where the directory said "Automobiles".

Deliberately NOT changed:
  * `kind` stays `doc`. Setting it to `vendor` would be more honest but the
    table is shared with the Vaani bridge, and nothing here needs the filter.
  * `localmama.businesses` is left in place. Vaani's matcher reads it directly,
    so dropping it would break another product for our tidiness.
"""

from __future__ import annotations

import argparse
import os
import sys

import psycopg
from dotenv import load_dotenv

OWNER = os.environ.get("BRAIN_OWNER_ID", "localmama")
AGENT = os.environ.get("BRAIN_AGENT_ID", "localmama")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write the changes")
    args = parser.parse_args()

    load_dotenv(".env")
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        print("DATABASE_URL is not set")
        return 1

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM utter.knowledge"
            " WHERE owner_id=%s AND agent_id=%s AND phone IS NOT NULL",
            (OWNER, AGENT),
        )
        before = cur.fetchone()[0]

        cur.execute(
            "SELECT count(*) FROM utter.knowledge k JOIN localmama.businesses b"
            " ON k.title = b.name"
            " WHERE k.owner_id=%s AND k.agent_id=%s AND b.phone IS NOT NULL"
            "   AND (k.phone IS DISTINCT FROM b.phone"
            "        OR k.category IS DISTINCT FROM b.category)",
            (OWNER, AGENT),
        )
        to_change = cur.fetchone()[0]

        print(f"brain rows with a phone: {before}")
        print(f"rows to update:          {to_change}")

        if args.apply and to_change:
            cur.execute(
                "UPDATE utter.knowledge k SET phone = b.phone, category = b.category"
                " FROM localmama.businesses b"
                " WHERE k.title = b.name AND k.owner_id=%s AND k.agent_id=%s"
                "   AND b.phone IS NOT NULL",
                (OWNER, AGENT),
            )
            conn.commit()

        cur.execute(
            "SELECT count(*) FROM utter.knowledge"
            " WHERE owner_id=%s AND agent_id=%s AND phone IS NOT NULL",
            (OWNER, AGENT),
        )
        after = cur.fetchone()[0]

    print(f"brain rows with a phone now: {after}")
    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply to commit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

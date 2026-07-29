"""Merge duplicate, misspelled and mis-cased categories in the directory.

Run with --apply to write; without it this is a dry run and touches nothing.

Only unambiguous merges are listed. A category that needs a judgement about what
a business actually does is deliberately left alone — "Service Provider" covers
nine businesses whose keywords are boilerplate ("business services,
professional support, consulting"), so nothing in the data says what they are,
and inventing a category would be worse than an ugly one.

The table is shared with the Vaani dashboard, which owns the write path. This
touches `category` only: names, phones and keywords are untouched, and keywords
are what Vaani's matcher actually keys on.
"""

from __future__ import annotations

import argparse
import os
import sys

import psycopg
from dotenv import load_dotenv

#: old -> new. Every line here is a spelling, casing or plural variant of the
#: target, or a strict subset of it. Nothing here changes what a business is.
MERGES: dict[str, str] = {
    "AutoMobiles": "Automobiles",
    "Automobiles Services": "Automobiles",
    "Manafcuters & Traders": "Manufacturers & Traders",     # typo
    "Manufactures & Traders": "Manufacturers & Traders",    # typo
    "Service Provider": "Service Providers",                # plural only
    "HOME TUTORS": "Tutors",
    "Delivery LOGISTICS": "Delivery & Logistics",
    "Resturants & FOOD COURT": "Restaurants & Food Court",  # typo + caps
    "mobile shop repairs & repairs": "Mobile Shops",        # duplicated word
    "Clothing / Fashion": "Clothing",
    "Events & Entertainment": "Events",
}


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
        cur.execute("SELECT count(DISTINCT category) FROM localmama.businesses")
        before = cur.fetchone()[0]

        total = 0
        print(f"{'category':34} {'->':4} {'becomes':28} rows")
        print("-" * 76)
        for old, new in MERGES.items():
            cur.execute(
                "SELECT count(*) FROM localmama.businesses WHERE category = %s", (old,)
            )
            n = cur.fetchone()[0]
            if not n:
                continue
            total += n
            print(f"{old:34} {'->':4} {new:28} {n}")
            if args.apply:
                cur.execute(
                    "UPDATE localmama.businesses SET category = %s WHERE category = %s",
                    (new, old),
                )

        if args.apply:
            conn.commit()
        cur.execute("SELECT count(DISTINCT category) FROM localmama.businesses")
        after = cur.fetchone()[0]

    print("-" * 76)
    print(f"{total} businesses re-filed; {before} categories -> {after}")
    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply to commit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

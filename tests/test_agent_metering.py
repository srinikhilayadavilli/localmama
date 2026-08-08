"""What the meter records, and under whose name.

Offline: none of this needs a database, which matters because the costing
tests that do are skipped without `TEST_DATABASE_URL` — and a test that only
runs on someone's laptop is a test that does not run.
"""

from __future__ import annotations



def test_a_hostname_is_normalised_to_the_provider():
    """`livekit-agents` fills `provider` from the client's base URL for some
    plugins, so the realtime model arrived as `api.openai.com` while the rate
    card prices `openai`. Six units priced at zero on a real call, and only the
    coverage line said so."""
    from agent.metering import _canonical_provider as canonical

    assert canonical("api.openai.com") == "openai"
    assert canonical("openai") == "openai"
    assert canonical("api.sarvam.ai") == "sarvam"


def test_an_unknown_hostname_is_left_alone_rather_than_guessed():
    """A rule would be worse than a map: "strip api. and the TLD" turns
    generativelanguage.googleapis.com into "googleapis", which nobody prices.
    An unrecognised name stays as it is and reads as unpriced, which is the
    honest outcome — a costing system that quietly renames things is worse than
    one that admits it does not know."""
    from agent.metering import _canonical_provider as canonical

    assert canonical("generativelanguage.googleapis.com") == "google"   # mapped
    assert canonical("api.unknown-vendor.io") == "api.unknown-vendor.io"
    assert canonical("") == "unknown"

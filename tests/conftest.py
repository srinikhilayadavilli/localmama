"""Test defaults: no network, no LLM, no surprises.

Once a real `ANTHROPIC_API_KEY` is present in `.env`, both the LLM extraction
fallback and natural phrasing switch themselves on — which quietly turned the
suite into a live-API integration run: 0.35s became several minutes, and every
result depended on a network round trip.

So both are forced off for every test. Tests that need them opt in explicitly
by patching the same properties (see the `natural_on` fixture in
`test_natural_replies.py`), which keeps the suite fast, offline, deterministic,
and free.
"""

from __future__ import annotations

import pytest

from backend.app.config import Settings, settings


@pytest.fixture(autouse=True)
def _offline_by_default(monkeypatch):
    monkeypatch.setattr(Settings, "llm_available", property(lambda self: False))
    monkeypatch.setattr(
        Settings, "natural_replies_available", property(lambda self: False)
    )
    # And no database. `.env` carries a real DATABASE_URL, so without this the
    # suite wrote its fixture leads — "Ravi / plumber / Pune" — straight into
    # the shared production Neon instance, and every brain lookup in a test hit
    # the network. Blank means lead_store and brain both report unavailable and
    # degrade exactly as they do on a host with no database configured.
    #
    # Patched on the INSTANCE, unlike the two properties above. `database_url`
    # is a dataclass field, so it lives in the instance __dict__ and a
    # class-level patch is simply shadowed — which is how the first attempt at
    # this silently kept writing to production. `settings` is frozen, hence
    # object.__setattr__ rather than monkeypatch.
    original_dsn = settings.database_url
    object.__setattr__(settings, "database_url", "")
    # And no translation, for the same reason and by the same mechanism: `.env`
    # carries a real SARVAM_API_KEY, so a vendor lookup in a test would post
    # the fixture's service phrase to Sarvam and bill for it. Blank means
    # `translate.available()` is False and `to_english` returns its input, which
    # is also how it degrades on a host with no Sarvam key.
    original_sarvam = settings.sarvam_api_key
    object.__setattr__(settings, "sarvam_api_key", "")
    yield
    object.__setattr__(settings, "database_url", original_dsn)
    object.__setattr__(settings, "sarvam_api_key", original_sarvam)

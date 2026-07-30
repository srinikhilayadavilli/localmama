"""Matching a caller's words against an English catalogue.

The failure this guards against is silent. A caller asks for a car wash in
Telugu, the phrase is stored as "కార్ వాష్", and `category LIKE '%కార్ వాష్%'`
is false for all 50 categories — so the WhatsApp message says we are still
shortlisting, for a trade that is right there in the directory.
"""

from __future__ import annotations

import pytest

from backend.app.config import settings
from backend.app.services import translate


@pytest.mark.parametrize(
    "text,expected",
    [
        ("కార్ వాష్", True),          # Telugu
        ("मोबाइल की दुकान", True),     # Devanagari
        ("பிளம்பர்", True),            # Tamil
        ("car wash", False),           # already English
        ("naaku plumber kaavali", False),  # romanised: Latin, left alone
        ("", False),
    ],
)
def test_only_non_latin_is_sent_for_translation(text: str, expected: bool) -> None:
    """Romanised input is Latin already; a round trip could only make it worse."""
    assert translate.needs_translation(text) is expected


@pytest.mark.asyncio
async def test_english_is_returned_untouched_without_a_network_call() -> None:
    """The common case must not cost an API call."""
    assert await translate.to_english("car wash") == "car wash"


@pytest.mark.asyncio
async def test_disabled_translation_returns_the_caller_s_words() -> None:
    """conftest blanks SARVAM_API_KEY, so this is the no-config path."""
    assert translate.available() is False
    assert await translate.to_english("కార్ వాష్") == "కార్ వాష్"


@pytest.mark.asyncio
async def test_a_failed_translation_falls_back_rather_than_raising(monkeypatch) -> None:
    """The worst case must be the behaviour we already had, not a lost handoff."""
    object.__setattr__(settings, "sarvam_api_key", "test-key")
    try:
        class _Boom:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def post(self, *a, **kw):
                raise RuntimeError("sarvam is down")

        import httpx

        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: _Boom())
        assert await translate.to_english("కార్ వాష్") == "కార్ వాష్"
    finally:
        object.__setattr__(settings, "sarvam_api_key", "")


@pytest.mark.asyncio
async def test_a_translation_is_used_for_the_lookup(monkeypatch) -> None:
    object.__setattr__(settings, "sarvam_api_key", "test-key")
    try:
        class _Reply:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return {"translated_text": "Car Wash", "source_language_code": "te-IN"}

        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def post(self, url, **kw):
                assert kw["json"]["target_language_code"] == "en-IN"
                # Detected, not taken from the session: transcripts wander
                # across scripts mid-call and the two disagree.
                assert kw["json"]["source_language_code"] == "auto"
                return _Reply()

        import httpx

        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: _Client())
        assert await translate.to_english("కార్ వాష్") == "Car Wash"
    finally:
        object.__setattr__(settings, "sarvam_api_key", "")


def test_the_nearest_category_absorbs_spelling_differences(monkeypatch) -> None:
    """Translation returns American spelling; the catalogue is British.

    "Jewelry store" against "jewellery stores" is the case that made this
    necessary — a literal LIKE misses it by two letters.
    """
    from backend.app.services import brain

    monkeypatch.setattr(
        brain, "categories",
        lambda: {"jewellery stores", "mobile shops", "car wash", "bakery",
                 "professional services", "bike & car rentals"},
    )
    assert brain.closest_category("jewelry store") == "jewellery stores"
    assert brain.closest_category("mobile shop") == "mobile shops"
    assert brain.closest_category("bakery") == "bakery"


def test_an_absent_trade_stays_absent(monkeypatch) -> None:
    """Finding nothing is a valid answer, and better than a confident wrong one.

    There is no photographer in the catalogue. Its nearest neighbour is
    "professional services", which is not what the caller asked for.
    """
    from backend.app.services import brain

    monkeypatch.setattr(
        brain, "categories",
        lambda: {"jewellery stores", "mobile shops", "car wash", "bakery",
                 "professional services", "bike & car rentals"},
    )
    assert brain.closest_category("photographer") is None
    assert brain.closest_category("") is None


def test_car_wash_is_not_dragged_towards_car_rentals(monkeypatch) -> None:
    """Whole-phrase similarity, so a shared first word does not decide it."""
    from backend.app.services import brain

    monkeypatch.setattr(
        brain, "categories", lambda: {"bike & car rentals", "car wash"},
    )
    assert brain.closest_category("car wash") == "car wash"


def test_the_fuzzy_fallback_cannot_loop(monkeypatch) -> None:
    """A category that matches nothing must end the search, not re-enter it.

    `matches_for_service` retries itself with the nearest category. If that
    category also has no rows with a phone, the retry's own nearest match is
    itself — so the guard has to stop there rather than ask again forever.
    """
    from backend.app.services import brain

    calls: list = []

    monkeypatch.setattr(brain, "available", lambda: True)
    monkeypatch.setattr(brain, "categories", lambda: {"car wash"})

    class _Cur:
        def __enter__(self): return self
        def __exit__(self, *e): return False
        def execute(self, *a, **kw): calls.append(a[1] if len(a) > 1 else None)
        def fetchall(self): return []

    class _Conn:
        def __enter__(self): return self
        def __exit__(self, *e): return False
        def cursor(self): return _Cur()

    monkeypatch.setattr(brain, "_connect", lambda: _Conn())
    assert brain.matches_for_service("carwash") == []
    # Once for the literal miss, once for the nearest category, then stop.
    assert len(calls) == 2, f"expected 2 queries, got {len(calls)}"

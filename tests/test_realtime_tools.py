"""Gemini talks; these tools decide what is true.

The free-form path could hold a perfect conversation and save nothing — 37
leads on disk, zero from Gemini. These tools are the only route from speech to
a stored lead, and each one applies the same validation as the typed pipeline.
"""

from __future__ import annotations

import pytest

from backend.app.config import settings
from backend.app.languages import Language
from backend.app.realtime_tools import LeadRecorder


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(type(settings), "data_dir", property(lambda self: tmp_path),
                        raising=False)
    from backend.app import persistence
    monkeypatch.setattr(persistence.settings.__class__, "data_dir",
                        property(lambda self: tmp_path), raising=False)
    return tmp_path


def tools(rec: LeadRecorder) -> dict:
    return {t.info.name: t for t in rec.build_tools()}


def test_exactly_the_expected_tools_are_exposed():
    assert set(tools(LeadRecorder())) == {
        "set_language", "set_name", "set_service", "set_city",
        "lookup_vendor_contact", "save_lead"
    }


# --- save is gated --------------------------------------------------------


@pytest.mark.asyncio
async def test_save_refuses_when_nothing_captured():
    rec = LeadRecorder(); t = tools(rec)
    reply = await t["save_lead"]()
    assert "missing" in reply.lower()
    assert rec.saved is False


@pytest.mark.asyncio
async def test_save_refuses_with_one_field_outstanding():
    rec = LeadRecorder(); t = tools(rec)
    await t["set_language"](language="English")
    await t["set_name"](name="Ravi")
    await t["set_service"](service="plumber")
    reply = await t["save_lead"]()          # no area
    assert "city" in reply.lower()
    assert rec.saved is False


@pytest.mark.asyncio
async def test_save_succeeds_once_complete():
    rec = LeadRecorder(); t = tools(rec)
    await t["set_language"](language="English")
    await t["set_name"](name="Ravi")
    await t["set_service"](service="plumber")
    await t["set_city"](city="Pune")
    reply = await t["save_lead"]()
    assert "saved" in reply.lower()
    assert rec.saved is True


# --- values are validated, not trusted -----------------------------------


@pytest.mark.asyncio
async def test_unsupported_language_is_rejected():
    rec = LeadRecorder(); t = tools(rec)
    reply = await t["set_language"](language="Klingon")
    assert "not a supported language" in reply
    assert rec.session.selected_language is None


@pytest.mark.asyncio
async def test_telugu_sentence_is_normalised_to_a_trade():
    """The real call said 'నేను ఎలక్ట్రిషన్ కోసం వెతుకుతున్నాను'."""
    rec = LeadRecorder(); t = tools(rec)
    await t["set_service"](service="నేను ఎలక్ట్రిషన్ కోసం వెతుకుతున్నాను")
    assert rec.session.requested_service == "electrician"


@pytest.mark.asyncio
async def test_a_native_script_name_is_stored_in_english():
    """The language is still whatever they chose; the value is always English.

    Sarvam is unavailable in tests (see conftest), so this is the offline
    romanisation — which is the point: the guarantee cannot depend on a network
    call succeeding.
    """
    rec = LeadRecorder(); t = tools(rec)
    await t["set_language"](language="తెలుగు")
    await t["set_name"](name="ఫణి కుమార్")
    assert rec.session.selected_language is Language.TELUGU
    assert rec.session.user_name == "Phani Kumar"


@pytest.mark.asyncio
async def test_markup_is_stripped_from_a_name():
    rec = LeadRecorder(); t = tools(rec)
    await t["set_name"](name="Ravi<script>alert(1)</script>")
    assert "<script>" not in (rec.session.user_name or "")


@pytest.mark.asyncio
async def test_empty_values_are_refused():
    rec = LeadRecorder(); t = tools(rec)
    assert "again" in (await t["set_name"](name="   ")).lower()
    assert rec.session.user_name is None


# --- the lead actually lands on disk -------------------------------------


@pytest.mark.asyncio
async def test_a_saved_lead_is_written_and_readable(isolated):
    rec = LeadRecorder(); t = tools(rec)
    await t["set_language"](language="తెలుగు")
    await t["set_name"](name="ఫణి కుమార్")
    await t["set_service"](service="నేను ఎలక్ట్రిషన్ కోసం వెతుకుతున్నాను")
    await t["set_city"](city="రాజమండ్రి")
    await t["save_lead"]()

    files = list((isolated / "leads").glob("*.json"))
    assert len(files) == 1
    import json
    lead = json.loads(files[0].read_text(encoding="utf-8"))
    # Every field English, whatever script the call was held in: this record is
    # matched against an English catalogue, sent over WhatsApp, and read by
    # whoever actions it.
    assert lead["user_name"] == "Phani Kumar"
    assert lead["requested_service"] == "electrician"
    assert lead["city_or_area"] == "Rajamandri"
    assert lead["conversation_status"] == "completed"


@pytest.mark.asyncio
async def test_a_correction_overwrites_before_saving():
    rec = LeadRecorder(); t = tools(rec)
    await t["set_service"](service="plumber")
    await t["set_service"](service="electrician")   # caller corrected at read-back
    assert rec.session.requested_service == "electrician"


@pytest.mark.asyncio
async def test_a_call_that_captured_nothing_writes_no_lead(isolated):
    """A Gemini run that never got past the greeting wrote an all-null record."""
    from backend.app.services.conversation_manager import abandon

    rec = LeadRecorder()
    abandon(rec.session)
    assert not list((isolated / "leads").glob("*.json"))


@pytest.mark.asyncio
async def test_a_partial_call_still_writes_what_it_had(isolated):
    from backend.app.services.conversation_manager import abandon

    rec = LeadRecorder(); t = tools(rec)
    await t["set_name"](name="Ravi")
    abandon(rec.session)
    files = list((isolated / "leads").glob("*.json"))
    assert len(files) == 1
    import json
    assert json.loads(files[0].read_text())["user_name"] == "Ravi"


@pytest.mark.asyncio
async def test_save_lead_is_idempotent(isolated):
    """A second save must not re-fire the side effects.

    The lead file is keyed by session id so it merely rewrites, but the
    WhatsApp handoff and caller-profile update are not idempotent: the caller
    would receive a second message and their call_count would double.
    """
    rec = LeadRecorder()
    t = tools(rec)
    await t["set_language"](language="telugu")
    await t["set_name"](name="Ravi")
    await t["set_service"](service="plumber")
    await t["set_city"](city="Hyderabad")

    first = await t["save_lead"]()
    assert "saved" in first.lower()

    sent: list = []
    from backend.app.services import whatsapp
    original, whatsapp.fire = whatsapp.fire, lambda *a, **k: sent.append(a)
    try:
        second = await t["save_lead"]()
    finally:
        whatsapp.fire = original

    assert "already saved" in second.lower()
    assert sent == [], "a repeat save must not send another WhatsApp message"
    assert len(list((isolated / "leads").glob("*.json"))) == 1


# --- language is not flipped by noise ---------------------------------


@pytest.mark.asyncio
async def test_first_language_is_accepted_immediately():
    t = tools(LeadRecorder())
    reply = await t["set_language"](language="telugu")
    assert "telugu" in reply.lower()


@pytest.mark.asyncio
async def test_changing_language_needs_confirming_first():
    """A real call switched to Hindi because the caller said "ఏది?" — Telugu for
    "which?". One half-heard word must not carry the rest of the call."""
    rec = LeadRecorder()
    t = tools(rec)
    await t["set_language"](language="telugu")

    first = await t["set_language"](language="hindi")
    assert rec.session.selected_language is Language.TELUGU, "must not switch yet"
    assert "do not switch" in first.lower()

    second = await t["set_language"](language="hindi")
    assert rec.session.selected_language is Language.HINDI, "a confirmed change applies"


@pytest.mark.asyncio
async def test_a_different_second_guess_does_not_confirm_the_first():
    """Two different mishearings must not add up to a confirmation."""
    rec = LeadRecorder()
    t = tools(rec)
    await t["set_language"](language="telugu")
    await t["set_language"](language="hindi")
    await t["set_language"](language="tamil")
    assert rec.session.selected_language is Language.TELUGU


@pytest.mark.asyncio
async def test_restating_the_current_language_is_harmless():
    rec = LeadRecorder()
    t = tools(rec)
    await t["set_language"](language="telugu")
    await t["set_language"](language="telugu")
    assert rec.session.selected_language is Language.TELUGU


# --- the model must never be told less than it has --------------------


@pytest.mark.asyncio
async def test_correcting_a_name_reports_the_other_fields_intact():
    """Reported as "fixing one thing makes it forget the rest". The values were
    never lost — the tool reply mentioned only the field just written, so the
    model re-anchored on that and re-asked for the rest."""
    rec = LeadRecorder()
    t = tools(rec)
    await t["set_language"](language="telugu")
    await t["set_name"](name="Ravi")
    await t["set_service"](service="plumber")
    await t["set_city"](city="Hyderabad")

    reply = await t["set_name"](name="Ravi Kumar")
    assert "Ravi Kumar" in reply
    assert "plumber" in reply, "the service must still be reported"
    assert "Hyderabad" in reply, "the city must still be reported"
    assert "STILL NEEDED: nothing" in reply


@pytest.mark.asyncio
async def test_state_line_lists_what_is_outstanding():
    rec = LeadRecorder()
    t = tools(rec)
    reply = await t["set_name"](name="Ravi")
    assert "STILL NEEDED" in reply
    for field in ("service", "city"):
        assert field in reply.split("STILL NEEDED")[1]


@pytest.mark.asyncio
async def test_save_lead_does_not_make_the_caller_wait(isolated):
    """The caller sits in silence until this tool returns.

    Measured on a real call, doing the durable write, the knowledge-base lookup
    and the WhatsApp send inline cost 15.26s of dead air before Mami could say
    goodbye — a synchronous psycopg round trip, a cold embedding model, then
    three retries against a provider that is down. None of it has to finish
    before she speaks, and the lead is already on disk.
    """
    import asyncio
    import time

    rec = LeadRecorder(caller_id="+919739960092")
    t = tools(rec)
    await t["set_language"](language="telugu")
    await t["set_name"](name="Ravi")
    await t["set_service"](service="electrician")
    await t["set_city"](city="Hyderabad")

    started = time.perf_counter()
    reply = await t["save_lead"]()
    elapsed = time.perf_counter() - started

    assert "saved" in reply.lower()
    assert elapsed < 1.0, f"save_lead blocked the caller for {elapsed:.2f}s"
    # The remaining work must be scheduled, not skipped.
    assert rec.background, "post-call work was dropped rather than backgrounded"
    await asyncio.gather(*rec.background, return_exceptions=True)


@pytest.mark.asyncio
async def test_background_work_is_referenced_until_it_finishes(isolated):
    """asyncio keeps only a weak reference to a task; an unreferenced one can be
    collected mid-await and the lead would never reach Postgres."""
    import asyncio

    rec = LeadRecorder()
    t = tools(rec)
    await t["set_language"](language="english")
    await t["set_name"](name="Ravi")
    await t["set_service"](service="plumber")
    await t["set_city"](city="Pune")
    await t["save_lead"]()

    assert len(rec.background) == 1
    await asyncio.gather(*rec.background, return_exceptions=True)
    assert not rec.background, "the done callback must release the reference"

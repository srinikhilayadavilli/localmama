"""Every captured detail is English, whatever language the call was in.

The catalogue is English and matching is literal, so a Devanagari value matches
nothing; the WhatsApp message and the Postgres row are read by people who do
not read five scripts. So the conversion happens once, at capture, and what is
stored is what is matched.

Sarvam is disabled throughout (conftest blanks the key), which makes these the
offline path — deliberately. The guarantee is that a value comes out in Latin
letters, and a guarantee that depends on a network call succeeding is not one.
"""

from __future__ import annotations

import pytest

from backend.app.realtime_tools import LeadRecorder
from backend.app.services import translate, translit


def tools(rec: LeadRecorder) -> dict:
    return {t.info.name: t for t in rec.build_tools()}


# --- the offline romaniser -----------------------------------------------


@pytest.mark.parametrize(
    "spoken, expected",
    [
        ("राहुल", "rahul"),            # Devanagari, final schwa dropped
        ("रुचि", "ruchi"),
        ("मुंबई", "mumbai"),           # anusvara takes the following labial
        ("चंदन", "chandan"),           # and stays dental otherwise
        ("रवि", "ravi"),
        ("রবি", "rabi"),               # Bengali
        ("రవి", "ravi"),               # Telugu keeps its final vowel
        ("ರವಿ", "ravi"),               # Kannada
        ("ரவி", "ravi"),               # Tamil
        ("சென்னை", "chennai"),
        ("बेंगलुरु", "bengaluru"),
    ],
)
def test_indic_names_come_out_readable(spoken, expected):
    assert translit.romanise(spoken) == expected


def test_latin_input_is_returned_untouched():
    """No round trip for a value that is already Latin — including romanised
    Indian input like "naaku plumber kaavali"."""
    assert translit.romanise("Rahul Sharma") == "Rahul Sharma"
    assert translit.romanise("naaku plumber kaavali") == "naaku plumber kaavali"


# --- names are respelled, never translated -------------------------------


@pytest.mark.asyncio
async def test_a_name_that_is_also_a_word_is_not_translated():
    """"आशा" is a common name and the Hindi word for hope. A translator returns
    "Hope" — a real word, a wrong name, and undetectable downstream."""
    assert await translate.english_name("आशा") == "asha"


@pytest.mark.asyncio
async def test_every_field_helper_returns_latin():
    for value in ("ఫణి కుమార్", "हैदराबाद", "कार वॉश"):
        for helper in (translate.english_name, translate.english_place,
                       translate.english_service):
            assert not translate.needs_translation(await helper(value))


# --- through the tools ----------------------------------------------------


@pytest.mark.asyncio
async def test_a_hindi_call_stores_english_everywhere():
    rec = LeadRecorder()
    t = tools(rec)
    await t["set_language"](language="हिंदी")
    await t["set_name"](name="राहुल")
    await t["set_service"](service="मुझे प्लंबर चाहिए")
    await t["set_city"](city="मुंबई")

    assert rec.session.user_name == "Rahul"
    # The rule catalogue already knows this trade, so it lands on the canonical
    # English label without any conversion at all.
    assert rec.session.requested_service == "plumber"
    assert rec.session.city_or_area == "Mumbai"


@pytest.mark.asyncio
async def test_the_state_line_shows_the_model_english_too():
    """What the tools report back is what was stored — otherwise the read-back
    and the lead disagree about the caller's own name."""
    rec = LeadRecorder()
    t = tools(rec)
    reply = await t["set_name"](name="राहुल")
    assert "Rahul" in reply
    assert "राहुल" not in reply


@pytest.mark.asyncio
async def test_an_unlisted_trade_still_reaches_the_catalogue_in_latin():
    """SERVICE_CATALOG covers 18 trades; the directory has 50 categories. A
    phrase it does not know must still not be stored in Devanagari."""
    rec = LeadRecorder()
    t = tools(rec)
    await t["set_service"](service="कार वॉश")
    stored = rec.session.requested_service
    assert stored and not translate.needs_translation(stored)


@pytest.mark.asyncio
async def test_english_input_is_unchanged():
    rec = LeadRecorder()
    t = tools(rec)
    await t["set_name"](name="Ravi Kumar")
    await t["set_city"](city="Pune")
    assert rec.session.user_name == "Ravi Kumar"
    assert rec.session.city_or_area == "Pune"

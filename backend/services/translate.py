"""Turn a caller's own words into English, so the catalogue can be searched.

The vendor catalogue is English — 50 categories like "car wash" and "mobile
shops", and English keywords beside them. Matching is literal (see `brain.py`
for why it must not be semantic), so a phrase in Telugu script cannot match any
of it: `'car wash' LIKE '%కార్ వాష్%'` is simply false, and the caller is told
we found nothing.

`entity_extractor.SERVICE_CATALOG` already canonicalises the trades it knows,
in every language it lists. But it holds 18 home-services trades and the
directory has 50 categories, only 4 of which a catalog label can reach — there
is no entry for bakeries, jewellers, hotels or mobile shops, and no hand-written
list is going to cover 50 categories across five languages. That is what this
is for.

Sarvam rather than a general model because the input is Indic and often a
single noun phrase, where a translation model with Indic training does better:
it returns "Jewelry store" for "నగల దుకాణం" and "Mobile shop" for
"मोबाइल की दुकान". Note it also normalises to *American* spelling, which is why
the caller of this must still match fuzzily — the catalogue says "jewellery
stores".

**Names and places are transliterated, not translated.** A translator renders
meaning, which is the right answer for a trade and a disaster for a proper
noun: "आशा" comes back as "Hope" and "नई दिल्ली" as "New Delhi" only by luck.
Sarvam's `/transliterate` respells instead, and `english_name`/`english_place`
use it.

**Every one of these is guaranteed to return Latin text.** When Sarvam is
disabled, slow, or down, the offline table in `translit.py` takes over rather
than the value being handed back in Devanagari — these values are stored, sent
over WhatsApp, and matched against an English catalogue, so a script nobody
downstream reads is not an acceptable degradation.

Runs after the caller has hung up, so the timeout is generous and the offline
table is a genuine last resort rather than a routine outcome — the reverse of
how this worked inside the agent. Never raises: a failed conversion must cost
spelling quality, not the lead.
"""

from __future__ import annotations

from ..config import settings
from ..logger import get_logger
from . import meter, translit

logger = get_logger("localmama.translate")

TRANSLATE_URL = "https://api.sarvam.ai/translate"
#: A name is not translated, it is respelled. Sarvam has a separate endpoint for
#: that, and the difference is not academic: put "आशा" through the translator and
#: it comes back "Hope", which is a real word, a wrong name, and completely
#: undetectable downstream.
TRANSLITERATE_URL = "https://api.sarvam.ai/transliterate"
#: mayura handles short noun phrases; sarvam-translate:v1 is tuned for prose.
MODEL = "mayura:v1"
#: Sarvam's cap for mayura is 1000; a service phrase is a handful of words, and
#: anything longer is a transcription accident rather than a trade.
MAX_INPUT = 200
#: Nothing is on a phone line here. Inside the agent this had to be 2 seconds,
#: because every tenth of a second past that was dead air the caller sat
#: through — and the offline table was often the better answer purely on speed.
#: In the backend the trade reverses: a better romanisation is worth waiting
#: for, so the timeout is generous and the offline table is a genuine last
#: resort rather than a routine outcome.
def _timeout() -> float:
    return settings.translate_timeout


def available() -> bool:
    return bool(settings.translate_enabled and settings.sarvam_api_key)


def needs_translation(text: str) -> bool:
    """Whether `text` has anything outside Latin script.

    Romanised input ("naaku plumber kaavali") is left alone deliberately. It is
    already Latin, the catalogue's keywords are Latin, and a round trip through
    a translator risks turning a usable string into a worse one — as well as
    spending a network call on the common case.
    """
    return any(ord(ch) > 0x24F for ch in text or "")


async def _post(url: str, payload: dict, timeout: float) -> dict:
    """One Sarvam call. Returns {} for every failure — never raises.

    The single choke point for both endpoints, and therefore the one place
    worth metering: Sarvam bills per character of input, and the character
    count is right here in the payload. A failure is recorded as an attempt
    that consumed nothing, so a degrading provider shows up as a run of
    not-ok rows rather than as an absence of rows.
    """
    import httpx

    operation = url.rsplit("/", 1)[-1]
    characters = len(payload.get("input") or "")
    with meter.metered("sarvam", "characters", model=payload.get("model") or "",
                       operation=operation) as measured:
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    url,
                    headers={"api-subscription-key": settings.sarvam_api_key},
                    json=payload,
                )
                response.raise_for_status()
                measured.succeeded(characters)
                return response.json()
        except Exception as exc:  # noqa: BLE001 - a conversion must never end a call
            # Zero characters billed: a request that never landed is not
            # charged for. The row still exists, carrying the error.
            measured.failed(0, repr(exc))
            logger.warning("sarvam %s failed (%s)", operation, exc)
            return {}


async def to_english(text: str, *, timeout: float | None = None) -> str:
    """`text` translated to English, or `text` unchanged if it cannot be.

    Meaning, not spelling — for a trade or a service phrase. Use
    `english_name`/`english_place` for a proper noun.
    """
    phrase = (text or "").strip()
    if not phrase or not needs_translation(phrase):
        return phrase
    if not available():
        logger.info("translation unavailable; using %r as-is", phrase[:40])
        return phrase
    if len(phrase) > MAX_INPUT:
        phrase = phrase[:MAX_INPUT]

    timeout = timeout if timeout is not None else _timeout()
    body = await _post(
        TRANSLATE_URL,
        {
            "input": phrase,
            # Detected rather than taken from the session language: the two
            # disagree in practice. A caller who picked Telugu says a business
            # name in English, and transcripts wander across scripts mid-call.
            "source_language_code": "auto",
            "target_language_code": "en-IN",
            "model": MODEL,
        },
        timeout,
    )
    english = (body.get("translated_text") or "").strip()
    if not english:
        return text
    logger.info(
        "translated %r -> %r (detected %s)",
        phrase[:40], english[:40], body.get("source_language_code"),
    )
    return english


async def transliterate(text: str, *, timeout: float | None = None) -> str:
    """`text` respelled in Latin letters, or `text` unchanged if it cannot be.

    Sarvam's romanisation of Indian names beats the offline table — it knows
    "ऋषभ" is Rishabh — so it is tried first and `english_name` falls back.
    """
    phrase = (text or "").strip()
    if not phrase or not needs_translation(phrase):
        return phrase
    if not available():
        return phrase
    if len(phrase) > MAX_INPUT:
        phrase = phrase[:MAX_INPUT]

    timeout = timeout if timeout is not None else _timeout()
    body = await _post(
        TRANSLITERATE_URL,
        {
            "input": phrase,
            "source_language_code": "auto",
            "target_language_code": "en-IN",
            # A phone number or a house number said mid-phrase stays digits.
            "numerals_format": "international",
        },
        timeout,
    )
    latin = (body.get("transliterated_text") or "").strip()
    if not latin:
        return phrase
    logger.info("transliterated %r -> %r", phrase[:40], latin[:40])
    return latin


def _latin(text: str, fallback_source: str) -> str:
    """`text` if it is already Latin, else the offline romanisation.

    The last line of the guarantee: whatever Sarvam did or did not do, what
    comes out of this module is readable by every system downstream.
    """
    if not needs_translation(text):
        return text
    romanised = translit.romanise(fallback_source)
    logger.info("falling back to offline romanisation: %r -> %r",
                fallback_source[:40], romanised[:40])
    return romanised


async def english_name(text: str) -> str:
    """A caller's name in Latin letters. Respelled, never translated."""
    spoken = (text or "").strip()
    if not spoken or not needs_translation(spoken):
        return spoken
    return _latin(await transliterate(spoken), spoken)


async def english_place(text: str) -> str:
    """A city or locality in Latin letters.

    Transliterated for the same reason a name is: a place name is a proper noun,
    and "मोती नगर" is Moti Nagar, not "Pearl City".
    """
    spoken = (text or "").strip()
    if not spoken or not needs_translation(spoken):
        return spoken
    return _latin(await transliterate(spoken), spoken)


async def english_service(text: str) -> str:
    """A service phrase in English — translated, because it is a description.

    Falls back to romanising rather than to the caller's script: a lead that
    reads "kar vosh" is still a lead a human can action, and it is still
    something the fuzzy category pass can work with.
    """
    spoken = (text or "").strip()
    if not spoken or not needs_translation(spoken):
        return spoken
    return _latin(await to_english(spoken), spoken)

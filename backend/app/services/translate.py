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

Runs off the caller's clock, in the post-call background work, so a slow
translation delays nobody. Never raises: a failed translation must cost a
vendor match, not the handoff.
"""

from __future__ import annotations

from ..config import settings
from ..logger import get_logger

logger = get_logger("localmama.translate")

TRANSLATE_URL = "https://api.sarvam.ai/translate"
#: mayura handles short noun phrases; sarvam-translate:v1 is tuned for prose.
MODEL = "mayura:v1"
#: Sarvam's cap for mayura is 1000; a service phrase is a handful of words, and
#: anything longer is a transcription accident rather than a trade.
MAX_INPUT = 200


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


async def to_english(text: str) -> str:
    """`text` in English, or `text` unchanged if it cannot be translated.

    Returning the input on failure is what keeps this safe to put in front of
    the vendor match: the worst case is the behaviour we already had.
    """
    phrase = (text or "").strip()
    if not phrase or not needs_translation(phrase):
        return phrase
    if not available():
        logger.info("translation unavailable; matching %r as-is", phrase[:40])
        return phrase
    if len(phrase) > MAX_INPUT:
        phrase = phrase[:MAX_INPUT]

    import httpx

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                TRANSLATE_URL,
                headers={"api-subscription-key": settings.sarvam_api_key},
                json={
                    "input": phrase,
                    # Detected rather than taken from the session language: the
                    # two disagree in practice. A caller who picked Telugu says
                    # a business name in English, and transcripts wander across
                    # scripts mid-call.
                    "source_language_code": "auto",
                    "target_language_code": "en-IN",
                    "model": MODEL,
                },
            )
            response.raise_for_status()
            body = response.json()
    except Exception as exc:  # noqa: BLE001 - a lookup must never end a handoff
        logger.warning("translation failed (%s); matching %r as-is", exc, phrase[:40])
        return text

    english = (body.get("translated_text") or "").strip()
    if not english:
        logger.warning("translation returned nothing for %r", phrase[:40])
        return text
    logger.info(
        "translated %r -> %r (detected %s)",
        phrase[:40], english[:40], body.get("source_language_code"),
    )
    return english

"""Turning a finished call into an actionable lead.

Four steps, in order, each of which can fail without losing the lead:

  normalise  →  audit  →  match  →  notify

**normalise** — the caller's own words become English. A name or a place is
transliterated ("आशा" is Asha, not "Hope"); a service is translated, because it
is a description rather than a proper noun. Both the raw and the English values
are kept: the raw ones are the only thing that can be re-processed when the
normaliser improves.

**audit** — every captured value is scored against an independent transcript of
the caller's own audio. This used to run inside the agent as a *gate*, which
meant a caller could be sent back to repeat their name at the moment they
expected to hang up. Here it produces a confidence score and a review flag. The
read-back is what corrects a bad value while the caller is on the line; this is
what catches the one that got past it.

**match** — the English service and city are matched literally against the
vendor catalogue. Finding nothing is a valid outcome and the message says so.

**notify** — the WhatsApp handoff. Its outcome is recorded on the lead, so one
that did not get through stays in the outbox and is retried by the worker
rather than being lost.

Runs after the API has already answered the agent. Nothing here is on anyone's
clock, which is the whole reason the split exists: the agent said goodbye and
hung up several seconds ago.
"""

from __future__ import annotations

from contract import CapturedField

from .config import settings
from .logger import get_logger
from .services import brain, translate, whatsapp
from .services.entity_extractor import extract, format_name
from .services.extraction import FOR_FIELD
from .services.grounding import score as ground_score
from . import store

logger = get_logger("localmama.pipeline")

#: Whether every word of a value has to trace back to the caller's audio.
#: A name and a city are proper nouns that exist only because the caller
#: uttered them. A service is a description and may fairly carry a word they
#: never said — "AC repair" for "मुझे एसी ठीक करवाना है".
_EVERY_WORD = {
    CapturedField.NAME: True,
    CapturedField.CITY: True,
    CapturedField.SERVICE: False,
}


async def normalise(raw: dict[str, str]) -> dict[str, str | None]:
    """Captured values in English, sanitised and ready to store and match."""
    name = raw.get(CapturedField.NAME.value)
    service = raw.get(CapturedField.SERVICE.value)
    city = raw.get(CapturedField.CITY.value)

    out: dict[str, str | None] = {"name": None, "service": None, "city": None}

    if name:
        # The rule extractor first: it strips "मेरा नाम X है" down to X, so what
        # goes to the transliterator is a name rather than a sentence.
        found = extract(name, expecting=FOR_FIELD[CapturedField.NAME])
        out["name"] = format_name(await translate.english_name(found.name or name))

    if service:
        found = extract(service, expecting=FOR_FIELD[CapturedField.SERVICE])
        # A trade the catalogue already knows is canonical in every language it
        # lists, in which case this costs nothing — there is no non-Latin text
        # left to send.
        out["service"] = await translate.english_service(found.requested_service or service)

    if city:
        found = extract(city, expecting=FOR_FIELD[CapturedField.CITY])
        out["city"] = format_name(await translate.english_place(found.city_or_area or city))

    return out


def _extractor_agrees(canonical: str | None, heard: list[str]) -> bool:
    """Whether the rules reach the same trade from the caller's own words.

    The second chance a service gets and a name does not, and it is not an
    edge case — it is the common Indic call. "AC repair" shares no word with
    "मुझे एसी ठीक करवाना है", so grounding alone scores it zero, but the
    catalogue maps "एसी" to `ac repair` straight off the transcript. Two
    independent routes to the same trade is exactly the corroboration being
    asked for, and without this every such call is flagged for review.
    """
    if not canonical:
        return False
    for turn in heard:
        found = extract(turn, expecting=FOR_FIELD[CapturedField.SERVICE])
        if found.requested_service == canonical:
            return True
    return False


def audit(
    raw: dict[str, str],
    heard: list[str],
    english: dict[str, str | None] | None = None,
) -> tuple[dict[str, dict], bool, str]:
    """Score each captured value against what the caller was transcribed saying.

    Returns (confidence per field, needs_review, reason).

    **This runs after `call.ended`, which is what makes it simple.** The race it
    used to guard against — the model calling a tool the instant it hears an
    answer, while transcription is a slower pass that lands afterwards — only
    existed while this ran inside the agent, mid-call. By the time a call is
    over, every turn has been transcribed, so the evidence is either all here
    or it never arrived at all.

    That earlier version carried a per-field count of how much had been
    transcribed at capture, and skipped any field whose count had not moved.
    The two cases it tried to separate — "the transcript for this answer has
    not landed yet" and "this answer was the last thing transcribed" — produce
    identical counts, so it took the conservative branch and skipped fields it
    could in fact have checked.

    `english` is the normalised form, used only to give a service its second
    chance through the rule extractor.
    """
    confidence: dict[str, dict] = {}
    reasons: list[str] = []
    english = english or {}

    if not heard:
        # Transcription produced nothing for the entire call. That is not
        # evidence a value was invented — it is the absence of any evidence,
        # and it means the independent second decode this check depends on
        # never happened. Flagged anyway, because a call with no transcript is
        # an operational fault worth seeing rather than a quiet pass.
        for field in _EVERY_WORD:
            if raw.get(field.value):
                confidence[field.value] = {
                    "score": 0.0, "evidence": False,
                    "note": "no transcript was recorded for this call",
                }
        return confidence, bool(confidence), (
            "no transcript to check against" if confidence else ""
        )

    for field, every_word in _EVERY_WORD.items():
        value = raw.get(field.value)
        if not value:
            continue

        value_score = ground_score(value, heard, every_word=every_word)
        if (
            field is CapturedField.SERVICE
            and value_score < settings.review_threshold
            and _extractor_agrees(english.get("service"), heard)
        ):
            confidence[field.value] = {
                "score": 1.0, "evidence": True,
                "note": "corroborated by the catalogue rules from the transcript",
            }
            continue

        note = "" if value_score >= settings.review_threshold else (
            "not found in what the caller was transcribed saying"
        )
        confidence[field.value] = {
            "score": round(value_score, 3), "evidence": True, "note": note,
        }
        if note:
            reasons.append(f"{field.value} ({value_score:.2f})")

    return confidence, bool(reasons), "; ".join(reasons)


async def process(call_id: str) -> dict | None:
    """Run the whole pipeline for one finished call. Never raises.

    Idempotent: safe to re-run on any call, which is what makes a failed
    pipeline recoverable — fix the bug, replay the call_id.
    """
    lead = store.get_lead(call_id)
    if lead is None:
        logger.warning("no lead row for %s; nothing to process", call_id[:8])
        return None

    raw = lead.get("raw") or {}
    if not raw:
        logger.info("call %s captured nothing; not processing an empty lead",
                    call_id[:8])
        store.upsert_call(call_id, processed_at=_now(), needs_review=False)
        return lead

    english = await normalise(raw)
    heard = store.transcript_for(call_id)
    confidence, needs_review, reason = audit(raw, heard, english=english)

    # A call that was never read back to the caller carries no human
    # verification, whatever the transcript says. That is worth a human's
    # attention even when every score is high.
    if lead.get("confirmed") is not True:
        needs_review = True
        unread = "never read back" if lead.get("confirmed") is None else "caller disagreed"
        reason = f"{reason}; {unread}" if reason else unread

    vendors = []
    if english["service"]:
        hits = brain.matches_for_service(english["service"], city=english["city"])
        vendors = [
            {"title": h.title, "phone": h.phone, "category": h.category, "city": h.city}
            for h in hits if h.phone
        ]

    store.upsert_call(
        call_id,
        name=english["name"], service=english["service"], city=english["city"],
        confidence=_json(confidence), needs_review=needs_review,
        review_reason=reason or None, vendors=_json(vendors),
        processed_at=_now(),
    )
    # Nothing is sent for a lead with no service.
    #
    # The template reads "here are some {service} options in {city}", and with
    # those fields empty it goes out as "here are some the service options in
    # your area" — to a caller who only picked a language before hanging up.
    # A real one of those reached a real phone. The lead is still stored and
    # still worth a human following up; it is the automatic message that has
    # nothing to say.
    if not english["service"]:
        logger.info("call %s has no service; storing the lead but sending nothing",
                    call_id[:8])
        store.mark_whatsapp(call_id, False, "incomplete lead")
        return store.get_lead(call_id)

    # Nothing is sent when the audit cannot vouch for the service itself.
    #
    # The message is *about* the service — "here are some electrician options".
    # A caller asked for a plumber on a bad line; STT produced Telugu on a
    # Tamil call, the model heard "electrician", and both this score and the
    # city's came back 0.00. The lead was correctly flagged, and the message
    # went out anyway telling them about electricians.
    #
    # Naming the wrong trade to a customer is worse than saying nothing. The
    # lead is stored, flagged, and waiting for a human — which is what
    # `needs_review` is for. Only the service gates this: a shaky name still
    # produces a useful message, and a shaky city only widens the search.
    service_score = confidence.get(CapturedField.SERVICE.value, {})
    if service_score.get("evidence") and service_score.get("score", 1.0) < settings.review_threshold:
        logger.warning(
            "call %s: service %r scored %.2f — not messaging the caller about a "
            "trade we cannot verify they asked for",
            call_id[:8], english["service"], service_score.get("score", 0.0),
        )
        store.mark_whatsapp(call_id, False, "service unverified")
        return store.get_lead(call_id)

    logger.info(
        "processed %s: %s / %s / %s — %d vendor(s)%s",
        call_id[:8], english["name"], english["service"], english["city"],
        len(vendors), f"  REVIEW: {reason}" if needs_review else "",
    )

    await notify(call_id, lead.get("caller_phone"), english, vendors)
    return store.get_lead(call_id)


async def notify(
    call_id: str, phone: str | None, english: dict, vendors: list[dict]
) -> None:
    """Hand the lead off over WhatsApp and record what happened."""
    if not phone:
        store.mark_whatsapp(call_id, False, "no phone")
        return

    options = " · ".join(
        f"{v['title']} {brain.spoken_phone(v['phone'])}".strip() for v in vendors
    )
    result = await whatsapp.send(
        phone,
        name=english.get("name"),
        service=english.get("service"),
        city=english.get("city"),
        options=options,
    )
    store.mark_whatsapp(
        call_id,
        bool(result.get("ok")),
        str(result.get("reason") or result.get("error") or ""),
    )


def _now():
    from datetime import datetime

    return datetime.now().astimezone()


def _json(value):
    from psycopg.types.json import Jsonb

    return Jsonb(value)

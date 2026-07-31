"""The lead pipeline, with no database and no phone call.

This is the point of the split: normalisation, the accuracy audit and vendor
matching are now plain functions over plain data, so the behaviour that used to
need a live call to exercise is testable from a fixture.
"""

from __future__ import annotations

import pytest

from backend.pipeline import audit, normalise
from backend.services.grounding import score


# --- normalisation: raw in, English out -----------------------------------


@pytest.mark.asyncio
async def test_a_name_is_respelled_not_translated():
    """A translator renders meaning, which is right for a trade and a disaster
    for a proper noun: "आशा" comes back as "Hope"."""
    out = await normalise({"name": "आशा"})
    assert out["name"] == "Asha"


@pytest.mark.asyncio
async def test_a_name_is_stripped_out_of_the_sentence_around_it():
    out = await normalise({"name": "मेरा नाम राहुल है"})
    assert out["name"] == "Rahul"


@pytest.mark.asyncio
async def test_a_place_is_respelled_not_translated():
    """"मोती नगर" is Moti Nagar. A translator offers "Pearl City"."""
    out = await normalise({"city": "मोती नगर"})
    assert "moti" in (out["city"] or "").lower()


@pytest.mark.asyncio
async def test_a_service_reaches_a_canonical_trade():
    """The catalogue is English and matching is literal, so a Telugu phrase has
    to arrive as something the catalogue actually contains."""
    out = await normalise({"service": "నాకు ఎలక్ట్రీషియన్ కావాలి"})
    assert out["service"] == "electrician"


@pytest.mark.asyncio
async def test_nothing_captured_normalises_to_nothing():
    out = await normalise({})
    assert out == {
        "name": None, "service": None, "service_said": None, "city": None,
    }


@pytest.mark.asyncio
async def test_the_callers_own_words_are_kept_alongside_the_canonical_trade():
    """Matching needs one label per trade, so "beauty parlor" becomes `salon`
    and reaches the salons. But the message is read by the person who made the
    call, and "here are some salon options" to someone who asked for a beauty
    parlour reads as though nobody was listening."""
    out = await normalise({"service": "beauty parlor"})
    assert out["service"] == "salon"                    # what we match on
    assert "parlo" in out["service_said"].lower()       # what they read


# --- the audit: a score, not a gate ---------------------------------------


def test_a_name_the_caller_said_scores_full():
    assert score("राहुल", ["मेरा नाम राहुल है"]) == 1.0


def test_a_name_the_caller_never_said_scores_zero():
    """The failure this exists for: a realtime model on a bad line does not
    fail loudly, it produces a fluent, plausible Indian name."""
    assert score("Suresh", ["मेरा नाम राहुल है"]) == 0.0


def test_a_partly_heard_name_scores_partly():
    """The whole reason this is a score and not a verdict. "Ravi Kumar" where
    only "Ravi" was transcribed is a far better lead than one where neither
    word was, and the old boolean threw that distinction away."""
    partial = score("Ravi Kumar", ["my name is Ravi"])
    assert 0.0 < partial < 1.0


def test_a_service_only_has_to_be_anchored_somewhere():
    """A description is anchored rather than matched word for word: "plumber
    chahiye" grounds "plumber" even though "chahiye" is not part of the value."""
    assert score("plumber", ["mujhe plumber chahiye"], every_word=False) == 1.0


@pytest.mark.asyncio
async def test_a_rephrased_service_is_rescued_by_the_catalogue():
    """The common Indic call, and the case that made this worth a second route.

    "AC repair" shares no word with "मुझे एसी ठीक करवाना है", so grounding alone
    scores it zero. The catalogue maps "एसी" to `ac repair` straight off the
    transcript, and two independent routes to the same trade is corroboration.
    Without this every such call would be flagged for review.
    """
    heard = ["मुझे एसी ठीक करवाना है"]
    assert score("AC repair", heard, every_word=False) == 0.0   # grounding alone

    english = await normalise({"service": "मुझे एसी ठीक करवाना है"})
    confidence, needs_review, _ = audit(
        raw={"service": "AC repair"}, heard=heard, english=english,
    )
    assert confidence["service"]["score"] == 1.0
    assert not needs_review


def test_two_romanisations_of_one_name_agree():
    """Two decoders of the same phone audio spell Indian names differently.
    Rejecting that would reject correct leads."""
    assert score("Rahool", ["rahul"]) == 1.0
    assert score("Laxmi", ["lakshmi"]) == 1.0


# --- the audit's verdict on a whole call ----------------------------------


def test_a_clean_call_needs_no_review():
    confidence, needs_review, reason = audit(
        raw={"name": "राहुल", "service": "plumber", "city": "मुंबई"},
        heard=["मेरा नाम राहुल है", "plumber chahiye", "मुंबई में"],
    )
    assert not needs_review
    assert reason == ""
    assert confidence["name"]["score"] == 1.0


def test_an_invented_name_is_flagged_with_a_reason():
    confidence, needs_review, reason = audit(
        raw={"name": "Suresh"},
        heard=["मेरा नाम राहुल है", "हाँ"],
    )
    assert needs_review
    assert "name" in reason
    assert confidence["name"]["score"] == 0.0
    assert confidence["name"]["evidence"] is True


def test_a_value_captured_on_the_last_transcribed_turn_is_still_checked():
    """The case the old guard skipped, found by running the smoke test against
    a real deployment.

    It carried a per-field count of how much had been transcribed at capture
    and refused to judge any field whose count had not moved since. That is
    indistinguishable from "this answer was the last thing transcribed" — so a
    real, checkable value went unaudited and a wrong one would have passed.
    """
    confidence, needs_review, _ = audit(
        raw={"city": "మాదాపూర్"},
        heard=["మాదాపూర్"],
    )
    assert confidence["city"]["evidence"] is True
    assert confidence["city"]["score"] == 1.0
    assert not needs_review


def test_a_call_with_no_transcript_at_all_is_flagged():
    """The audit runs after the call has ended, so the whole transcript is
    either here or it never arrived. Nothing to check against means the
    independent second decode never happened — an operational fault worth
    seeing rather than a quiet pass. Still not evidence a value was invented."""
    confidence, needs_review, reason = audit(raw={"name": "Ravi"}, heard=[])
    assert confidence["name"]["evidence"] is False
    assert confidence["name"]["score"] == 0.0
    assert needs_review
    assert "no transcript" in reason


def test_nothing_captured_and_no_transcript_is_not_flagged():
    """A call that captured nothing is noise, not a lead needing review."""
    confidence, needs_review, _ = audit(raw={}, heard=[])
    assert confidence == {}
    assert not needs_review


def test_a_field_the_caller_never_gave_is_not_audited():
    confidence, needs_review, _ = audit(
        raw={"name": "राहुल"},
        heard=["मेरा नाम राहुल है", "ok"],
    )
    assert "city" not in confidence
    assert not needs_review

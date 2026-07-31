"""Read-only access to the shared vendor catalogue ("the brain").

Backed by the NeonDB instance the Vaani bridge also uses, scoped by
`owner_id`/`agent_id` so both products share one store without colliding. Local
Mama's rows live under `localmama/localmama`.

**Matching is literal, never semantic.** The hybrid pgvector retrieval this
started as is gone: it was already dead code by the time it was removed, and it
was removed rather than revived because the failure it produces is the worst
kind available here. Semantic search always returns its nearest neighbour, so
"electrician" matched an EV charging company at 0.59 — a real business, with a
real phone number, sent to a caller who wanted their wiring fixed. Finding
nothing is a valid and honest outcome; the WhatsApp template falls back to "our
team is shortlisting", which is true.

Deliberately read-only. Ingest, editing and curation stay in the Vaani
dashboard, which owns the write path.

Everything here degrades to `[]` on a database problem rather than raising: a
lookup failing must cost a lead its vendor list, not the lead itself.
"""

from __future__ import annotations

import difflib
import time
from dataclasses import dataclass, field

from .. import db
from ..config import settings
from ..logger import get_logger

logger = get_logger("localmama.brain")


@dataclass
class Hit:
    """One catalogue entry."""

    id: int
    title: str
    content: str
    category: str | None
    city: str | None
    phone: str | None
    attrs: dict = field(default_factory=dict)
    rank: int = 0
    #: True when this was reached by spelling rather than by a literal match.
    #: The caller is asked to confirm the name before the number is read out —
    #: a close spelling is a guess, and the cost of a wrong one is a stranger's
    #: phone ringing.
    approximate: bool = False


_COLUMNS = (
    "id, title, content, category, city, phone, coalesce(attrs,'{}'::jsonb)"
)


def available() -> bool:
    return db.available()


def _scope() -> tuple[str, str]:
    return settings.brain_owner_id, settings.brain_agent_id


# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------

#: The catalogue's category list changes when someone edits the dashboard,
#: which is on the order of once a week. Re-querying it on every lookup — which
#: the agent did, opening a fresh connection each time — bought nothing.
_CATEGORY_TTL = 300.0
_categories: set[str] = set()
_categories_at: float = 0.0


def categories(force: bool = False) -> set[str]:
    """Every category in the catalogue, lowercased. Cached for `_CATEGORY_TTL`."""
    global _categories, _categories_at
    if not available():
        return set()
    if not force and _categories and (time.monotonic() - _categories_at) < _CATEGORY_TTL:
        return _categories
    try:
        owner, agent = _scope()
        with db.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT lower(category) FROM utter.knowledge"
                " WHERE owner_id = %s AND agent_id = %s AND category IS NOT NULL",
                (owner, agent),
            )
            _categories = {r[0] for r in cur.fetchall() if r[0]}
        _categories_at = time.monotonic()
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not read categories: %s", exc)
    return _categories


def looks_like_a_category(query: str) -> bool:
    """Whether the caller named a kind of business rather than a business.

    Matched as a whole word so "wash" is caught by "Car Wash" while "WOW Wash"
    is still read as the business it is.
    """
    q = (query or "").strip().lower()
    if not q:
        return False
    for category in categories():
        if q == category or q in category.split():
            return True
    return False


#: How close a phrase has to be to a category before it is treated as that
#: category. 0.75 keeps "jewelry store"/"jewellery stores" (0.86) and
#: "mobile shop"/"mobile shops" (0.96) while rejecting "photographer", whose
#: best neighbour is "professional services" at 0.46 — there is no
#: photographer in the catalogue, and saying so is the correct answer.
CATEGORY_CUTOFF = 0.75


def closest_category(query: str) -> str | None:
    """The catalogue category nearest to `query` by spelling, or None.

    Only ever consulted after a literal match has failed. Whole-phrase
    similarity rather than per-word, so "car wash" is not dragged towards
    "car rentals" by its first word.
    """
    phrase = (query or "").strip().lower()
    if not phrase:
        return None
    found = difflib.get_close_matches(
        phrase, sorted(categories()), n=1, cutoff=CATEGORY_CUTOFF
    )
    return found[0] if found else None


# ---------------------------------------------------------------------------
# Business lookup, by name
# ---------------------------------------------------------------------------

#: Names are matched tighter than categories. A category is a word the caller
#: chose loosely; a name is a specific record, and the cost of the nearest one
#: being wrong is a stranger's phone ringing. 0.82 keeps "Pax Jewellers" ->
#: "Pax Jwellers" (0.96) and "Cleanmates" -> "Clean Mates" (0.95) while
#: rejecting "Speed Autos" -> "Speed Kawasaki Kolkata".
NAME_CUTOFF = 0.82

#: Typography the catalogue contains but a caller cannot say or a transcriber
#: produce: a curly apostrophe in "Brinda's kitchen", a slash in "M/S B R
#: Enterprises". Folded on BOTH sides before comparing, so neither has to be
#: cleaned up in the data.
#: Every form maps straight to a space in ONE pass. Mapping "’" to "'" and
#: "'" to " " looks equivalent and is not: `str.translate` does not re-examine
#: what it substitutes, so "Brinda’s" became "brinda's" while "Brinda's" became
#: "brinda s" — the exact pair this exists to reconcile.
_PUNCT = str.maketrans({ch: " " for ch in "’‘'\"“”/.,-–—()&"})


def normalise_name(text: str) -> str:
    """A business name reduced to what a caller could actually say."""
    return " ".join((text or "").lower().translate(_PUNCT).split())


def find_business(name: str, limit: int = 5) -> list[Hit]:
    """Businesses matching a NAME, most literal first.

    A caller asking "what is Brimmies Cafe's number?" wants an exact record.
    """
    query = (name or "").strip()
    if not available() or not query:
        return []
    try:
        found = _literal_business(query, limit)
        if found:
            logger.info("name lookup %r -> %s", query[:40], [h.title for h in found])
            return found

        # Try again without the words a caller wraps a name in.
        #
        # A caller asked for "Pax business" and got nothing, because the
        # catalogue calls it "Pax Jwellers" and `LIKE '%pax business%'` matches
        # no title on earth. "Pax" on its own finds it immediately. People say
        # "the Pax people", "Clean Mates shop", "that car wash company" — the
        # name is in there, wearing a coat.
        #
        # Deliberately the *second* attempt: a business genuinely called "The
        # Coffee Shop" is still found by its real name first, and only a query
        # that has already failed gets taken apart.
        core = _core_name(query)
        if core and core != normalise_name(query):
            found = _literal_business(core, limit)
            if found:
                logger.info("name lookup %r -> %r -> %s",
                            query[:40], core, [h.title for h in found])
                return found

        # Then every word, all of which must appear. Word order stops
        # mattering: "Jwellers Pax" and "Mates Clean" find nothing as a
        # contiguous string and are obviously the business the caller means.
        #
        # ALL of them, never any of them. Matching on any word was measured
        # against this catalogue and is actively harmful — "car wash company"
        # returned seventeen businesses including a healthcare firm and a
        # tutoring academy, because one common word hit unrelated keywords.
        # Several matches with no exact title makes the agent ask the caller to
        # repeat themselves, and a single spurious one makes it read out a
        # stranger's number.
        found = _all_words_business(core or normalise_name(query), limit)
        if found:
            logger.info("name lookup %r -> every word -> %s",
                        query[:40], [h.title for h in found])
            return found
    except Exception as exc:  # noqa: BLE001
        logger.warning("name lookup failed (%s); continuing without it", exc)
        return []
    return _approximate_business(query, limit)


#: Words too short to narrow anything down. "Of", "at" and the like appear in
#: half the catalogue's keywords and would make the AND below meaningless.
_MIN_WORD = 3

#: A caller names a business in a few words. More than this is a transcription
#: accident, and an unbounded AND is an unbounded query.
_MAX_WORDS = 6


def _all_words_business(query: str, limit: int) -> list[Hit]:
    """Entries whose title or keywords contain EVERY word of `query`."""
    words = [w for w in query.split() if len(w) >= _MIN_WORD][:_MAX_WORDS]
    if len(words) < 2:
        return []          # one word is what the literal search already tried
    owner, agent = _scope()
    clauses = " AND ".join(
        f"(lower(title) LIKE %(w{i})s OR lower(coalesce(keywords,'')) LIKE %(w{i})s)"
        for i in range(len(words))
    )
    params: dict = {f"w{i}": f"%{w}%" for i, w in enumerate(words)}
    params.update(owner=owner, agent=agent, k=limit)
    with db.cursor() as cur:
        cur.execute(
            f"SELECT {_COLUMNS}, 2"
            " FROM utter.knowledge"
            f" WHERE owner_id = %(owner)s AND agent_id = %(agent)s AND ({clauses})"
            " ORDER BY title LIMIT %(k)s",
            params,
        )
        return [Hit(*row) for row in cur.fetchall()]


#: Words a caller wraps a business name in, and which stop a literal match
#: dead. None of them is ever the distinguishing part of a name.
_FILLER = frozenset({
    "business", "businesses", "shop", "shops", "store", "stores",
    "company", "co", "people", "guys", "place",
    "phone", "number", "contact", "the", "a", "an", "of", "for",
})


def _core_name(query: str) -> str:
    """A business name with the words around it removed.

    Empty if nothing survives — "the shop" names no business, and searching on
    what is left of it would match half the catalogue.
    """
    words = [w for w in normalise_name(query).split() if w not in _FILLER]
    return " ".join(words)


def _literal_business(query: str, limit: int) -> list[Hit]:
    """Titles and keywords containing `query`, most literal first."""
    owner, agent = _scope()
    with db.cursor() as cur:
        cur.execute(
            f"SELECT {_COLUMNS},"
            "  CASE WHEN lower(title) = lower(%(q)s) THEN 0"
            "       WHEN lower(title) LIKE lower(%(q)s) || '%%' THEN 1"
            "       ELSE 2 END AS rank"
            " FROM utter.knowledge"
            " WHERE owner_id = %(owner)s AND agent_id = %(agent)s"
            "   AND (lower(title) LIKE '%%' || lower(%(q)s) || '%%'"
            "        OR lower(coalesce(keywords,'')) LIKE '%%' || lower(%(q)s) || '%%')"
            " ORDER BY rank, title LIMIT %(k)s",
            {"q": query, "k": limit, "owner": owner, "agent": agent},
        )
        return [Hit(*row) for row in cur.fetchall()]


def _approximate_business(query: str, limit: int) -> list[Hit]:
    """Nearest titles by spelling, once a literal match has failed.

    Deliberately edit distance and not embeddings. The neighbourhood of a
    spelling is a handful of near-identical strings; the neighbourhood of a
    meaning is the whole catalogue.
    """
    wanted = normalise_name(query)
    if not wanted:
        return []
    owner, agent = _scope()
    try:
        with db.cursor() as cur:
            cur.execute(
                "SELECT title FROM utter.knowledge"
                " WHERE owner_id = %s AND agent_id = %s AND title IS NOT NULL",
                (owner, agent),
            )
            titles = [r[0] for r in cur.fetchall()]

        scored = sorted(
            ((difflib.SequenceMatcher(None, wanted, normalise_name(t)).ratio(), t)
             for t in titles),
            key=lambda pair: (-pair[0], pair[1]),
        )
        near = [t for score, t in scored[:limit] if score >= NAME_CUTOFF]
        if not near:
            logger.info("no business close to %r (best %.2f)",
                        query[:40], scored[0][0] if scored else 0.0)
            return []

        with db.cursor() as cur:
            cur.execute(
                f"SELECT {_COLUMNS}, 0"
                " FROM utter.knowledge"
                " WHERE owner_id = %s AND agent_id = %s AND title = ANY(%s)"
                " ORDER BY title",
                (owner, agent, near),
            )
            hits = [Hit(*row) for row in cur.fetchall()]
    except Exception as exc:  # noqa: BLE001
        logger.warning("approximate lookup failed (%s)", exc)
        return []

    for hit in hits:
        hit.approximate = True
    logger.info("name lookup %r -> %s (approximate)", query[:40], [h.title for h in hits])
    return hits


# ---------------------------------------------------------------------------
# Business lookup, by service
# ---------------------------------------------------------------------------


#: Every word the catalogue actually uses, from categories and keywords.
#: Cached alongside the category list and refreshed on the same clock.
_VOCAB: set[str] = set()
_vocab_at: float = 0.0


def vocabulary(force: bool = False) -> set[str]:
    """The words this catalogue is written in.

    Spelling correction is generated rather than listed (see `_variants`), and
    a generated variant is only trusted if it appears here. That is what makes
    guessing safe: a wrong guess matches nothing and costs nothing.
    """
    global _VOCAB, _vocab_at
    if not available():
        return set()
    if not force and _VOCAB and (time.monotonic() - _vocab_at) < _CATEGORY_TTL:
        return _VOCAB
    try:
        owner, agent = _scope()
        with db.cursor() as cur:
            cur.execute(
                "SELECT lower(coalesce(category,'') || ' ' || coalesce(keywords,''))"
                " FROM utter.knowledge WHERE owner_id = %s AND agent_id = %s",
                (owner, agent),
            )
            words: set[str] = set()
            for (blob,) in cur.fetchall():
                words.update(w for w in normalise_name(blob).split() if w)
        _VOCAB, _vocab_at = words, time.monotonic()
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not read the catalogue vocabulary: %s", exc)
    return _VOCAB


#: How American and Indian/British English differ, as transformations rather
#: than a word list. Applied in both directions, because a caller may say
#: either and the catalogue may be written in either.
#:
#: These are deliberately over-generous — "doctor" -> "doctour" is nonsense and
#: "motor" -> "motour" worse. It does not matter: every candidate is checked
#: against `vocabulary()` before it is used, so a nonsense variant matches
#: nothing and is discarded. Generating and filtering beats maintaining a list
#: that is always one word short of the call you just lost.
_SPELLING_RULES: tuple[tuple[str, str], ...] = (
    ("or", "our"),      # parlor/parlour, color/colour, labor/labour
    ("er", "re"),       # center/centre, theater/theatre, meter/metre
    ("ize", "ise"),     # organize/organise, specialize/specialise
    ("yze", "yse"),     # analyze/analyse
    ("og", "ogue"),     # catalog/catalogue, dialog/dialogue
    ("ler", "ller"),    # traveler/traveller, jeweler/jeweller
    ("lry", "llery"),   # jewelry/jewellery
    ("se", "ce"),       # defense/defence, license/licence
    ("ense", "ence"),
    ("ll", "l"),        # enrollment/enrolment
)

#: Whole-word pairs no suffix rule reaches.
_SPELLING_WORDS: dict[str, str] = {
    "tire": "tyre", "tires": "tyres", "aluminum": "aluminium",
    "gray": "grey", "mold": "mould", "plow": "plough",
    "pajamas": "pyjamas", "mustache": "moustache", "check": "cheque",
}


def _variants(word: str) -> list[str]:
    """Plausible other-spellings of one word. Unfiltered; the caller filters."""
    out = []
    for src, dst in list(_SPELLING_WORDS.items()) + [
        (v, k) for k, v in _SPELLING_WORDS.items()
    ]:
        if word == src:
            out.append(dst)
    for a, b in _SPELLING_RULES:
        if word.endswith(a):
            out.append(word[: -len(a)] + b)
        if word.endswith(b):
            out.append(word[: -len(b)] + a)
    return out


def anglicise(text: str) -> str:
    """`text` respelled into whatever spelling the catalogue actually uses.

    A word already in the catalogue is left alone. One that is not gets its
    variants generated and the first that the catalogue does contain wins.
    """
    known = vocabulary()
    out = []
    for word in normalise_name(text).split():
        if word in known:
            out.append(word)
            continue
        out.append(next((v for v in _variants(word) if v in known), word))
    return " ".join(out)


#: Words that make a phrase a request rather than a trade. "service" leads the
#: list because it is the one that did damage: a motorcycle dealer carries the
#: keywords "car service, bike service", so every query containing the word
#: matched it.
_SERVICE_FILLER = frozenset({
    "service", "services", "servicing", "wala", "walla",
    "near", "nearby", "me", "my", "our", "in", "at", "for", "of",
    "the", "a", "an", "some", "any", "please", "need", "needed",
    "want", "wanted", "looking", "require", "required", "guy", "guys",
    "person", "people", "work", "job",
    # Whatever a caller wraps the request in. "I need a plumber" is a plumber.
    "i", "we", "us", "you", "is", "am", "are", "do", "does", "can",
    "could", "would", "get", "find", "help", "there", "hi", "hello",
})



def _known_words(text: str) -> str:
    """Only the words this catalogue has ever heard of.

    A word that appears in no category and no keyword cannot contribute to a
    match — requiring it in an AND guarantees zero rows. Dropping it therefore
    costs nothing and cannot introduce a false hit, while it rescues every
    phrasing nobody thought to put in a filler list.

    A caller said "necessity of parlour". "of" was filler and "parlour" was
    real, but "necessity" survived, went into the AND, and matched nothing —
    so a query that named the trade exactly returned nothing. Indian English
    is full of these: "requirement of plumber", "urgent need of electrician",
    "kindly arrange one carpenter".

    This is the general form of the filler list, and the list stays because it
    strips words the catalogue *does* contain — "service" is in Speed
    Kawasaki's keywords, so only an explicit rule can remove it.
    """
    known = vocabulary()
    if not known:
        return normalise_name(text)
    return " ".join(w for w in normalise_name(text).split() if w in known)


def _service_core(text: str) -> str:
    """A service phrase reduced to the trade in it.

    Empty when nothing is left — "service near me" names no trade, and
    searching on the remains would match anything.
    """
    return " ".join(
        w for w in normalise_name(text).split() if w not in _SERVICE_FILLER
    )


def _service_rows(query: str, city: str | None) -> list[Hit]:
    """Listings with a phone whose category or keywords contain `query`."""
    owner, agent = _scope()
    where = [
        "owner_id = %(owner)s", "agent_id = %(agent)s", "phone IS NOT NULL",
        "(lower(coalesce(category,'')) LIKE '%%' || lower(%(q)s) || '%%'"
        " OR lower(coalesce(keywords,'')) LIKE '%%' || lower(%(q)s) || '%%')",
    ]
    params: dict = {"q": query, "owner": owner, "agent": agent}
    _add_city(where, params, city)
    with db.cursor() as cur:
        cur.execute(
            f"SELECT {_COLUMNS},"
            "  CASE WHEN lower(coalesce(category,'')) LIKE '%%' || lower(%(q)s) || '%%'"
            "       THEN 0 ELSE 1 END AS rank"
            " FROM utter.knowledge"
            f" WHERE {' AND '.join(where)}"
            " ORDER BY rank, title",
            params,
        )
        return [Hit(*row) for row in cur.fetchall()]


def _service_rows_all_words(query: str, city: str | None) -> list[Hit]:
    """Listings containing EVERY word of `query`, in any order.

    All of them, never any of them — for the same reason as the name lookup.
    One common word matches unrelated keywords across half the catalogue.
    """
    words = [w for w in query.split() if len(w) >= _MIN_WORD][:_MAX_WORDS]
    if len(words) < 2:
        return []
    owner, agent = _scope()
    clauses = " AND ".join(
        f"(lower(coalesce(category,'')) LIKE %(w{i})s"
        f" OR lower(coalesce(keywords,'')) LIKE %(w{i})s)"
        for i in range(len(words))
    )
    where = ["owner_id = %(owner)s", "agent_id = %(agent)s",
             "phone IS NOT NULL", f"({clauses})"]
    params: dict = {f"w{i}": f"%{w}%" for i, w in enumerate(words)}
    params.update(owner=owner, agent=agent)
    _add_city(where, params, city)
    with db.cursor() as cur:
        cur.execute(
            f"SELECT {_COLUMNS}, 1 FROM utter.knowledge"
            f" WHERE {' AND '.join(where)} ORDER BY title",
            params,
        )
        return [Hit(*row) for row in cur.fetchall()]


def _add_city(where: list, params: dict, city: str | None) -> None:
    """A hard city filter, matched either way round.

    The caller gives a locality as often as a city ("Madhapur"), and the
    listings carry cities. Listings with no city stay eligible, because most of
    them have none and excluding them would empty the catalogue.
    """
    place = (city or "").strip()
    if not place:
        return
    where.append("(city IS NULL OR city ILIKE %(city)s"
                 " OR %(city_plain)s ILIKE '%%' || city || '%%')")
    params["city"] = f"%{place}%"
    params["city_plain"] = place


def matches_for_service(
    service: str, limit: int = 3, *, city: str | None = None
) -> list[Hit]:
    """Businesses that actually do this work, with phone numbers.

    Category matches win outright over keyword matches. Without that, "cleaning"
    returned a dental clinic — "teeth cleaning" is a fair keyword for a dentist
    and a terrible answer for someone who wants their house cleaned.

    `city` is a hard filter, not a similarity term: a plumber in Chennai is not
    a weaker match for a Hyderabad caller, it is the wrong answer. Listings with
    no city stay eligible, because most of them have none and excluding them
    would empty the catalogue.

    Both the query and the city are English by the time they reach here — the
    pipeline normalises before matching — which is the whole reason a literal
    match can work at all.
    """
    query = (service or "").strip()
    if not available() or not query:
        return []

    try:
        # The phrase as spoken.
        rows = _service_rows(query, city)

        # Then in the catalogue's spelling. Sarvam translates to American forms
        # and this catalogue is British: a caller asking for a "beauty parlor"
        # found nothing, while "beauty parlour" found two. `closest_category`
        # was supposed to bridge that, and cannot — "beauty parlor" against
        # "beauty & personal care" is nowhere near the 0.75 cutoff.
        spelled = anglicise(query)
        if not rows and spelled != query.lower():
            rows = _service_rows(spelled, city)
            if rows:
                logger.info("service match %r -> %r (spelling)", query[:40], spelled)

        # Then without the words that make it a request rather than a trade.
        #
        # "service" is the dangerous one. Speed Kawasaki Kolkata, a motorcycle
        # dealer, carries the keywords "car service, bike service" — so any
        # query containing the word matched it, and a caller who asked for a
        # beauty parlour was sent a Kawasaki showroom.
        core = _service_core(spelled)
        if not rows and core and core != spelled:
            rows = _service_rows(core, city)
            if rows:
                logger.info("service match %r -> %r (core)", query[:40], core)

        # Then with the words the catalogue has never heard of removed. They
        # can only ever drive an AND to zero, so keeping them loses matches and
        # dropping them cannot invent one.
        known = _known_words(core)
        if not rows and known and known != core:
            rows = _service_rows(known, city) or _service_rows_all_words(known, city)
            if rows:
                logger.info("service match %r -> %r (known words)", query[:40], known)

        # Then every word of it, all of which must appear.
        if not rows and core:
            rows = _service_rows_all_words(core, city)
            if rows:
                logger.info("service match %r -> every word of %r", query[:40], core)
    except Exception as exc:  # noqa: BLE001
        logger.warning("service match failed (%s); continuing without it", exc)
        return []

    # Nothing literal at all. The nearest category by spelling is the last
    # resort — a caller's phrase is rarely a category verbatim ("mobile shop"
    # vs "mobile shops") — and it is deliberately last, because a literal hit
    # is better evidence than a close one.
    if not rows:
        nearest = closest_category(core or query)
        if nearest and nearest != query:
            logger.info("no literal match for %r; trying category %r", query[:40], nearest)
            return matches_for_service(nearest, limit, city=city)
        return []

    # A category hit is strong evidence; keyword-only hits are noise beside it.
    by_category = [h for h in rows if h.rank == 0]
    chosen = (by_category or rows)[:limit]
    logger.info("service match %r -> %s", query[:40], [h.title for h in chosen])
    return chosen


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------


def spoken_phone(phone: str | None) -> str:
    """Digits grouped so a TTS engine reads them as a number, not a year.

    "9735927627" said as one token comes out as garbage on every engine we have
    tried; in groups it is dictatable.
    """
    digits = "".join(ch for ch in (phone or "") if ch.isdigit())
    if len(digits) == 10:
        return f"{digits[:5]} {digits[5:]}"
    if len(digits) == 12 and digits.startswith("91"):
        return f"+91 {digits[2:7]} {digits[7:]}"
    return phone or ""


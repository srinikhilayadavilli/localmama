"""Read-only access to the shared knowledge base ("the brain").

Backed by the same NeonDB/pgvector instance the Vaani bridge uses, scoped by
`owner_id`/`agent_id` so both products share one store without colliding. Local
Mama's rows live under `localmama/localmama` — 120 businesses with categories
and curated keywords at the time of writing.

Retrieval is HYBRID and 0-LLM: vector cosine similarity over a local
multilingual embedding, plus Postgres full-text rank, plus a boost for the
admin-curated keywords. That combination is what catches both "someone to fix
my geyser" (semantic) and "plumber" (keyword) — and the multilingual embedding
is why a Telugu phrase retrieves an English-titled listing.

Deliberately read-only. Ingest, editing and curation stay in the Vaani
dashboard, which owns the write path; a voice agent that could write to a
shared knowledge base is a much larger blast radius for no gain here.

Ported from `bridge/engine/brain.py`. The scoring weights are copied rather
than re-tuned — they were fitted against this data, and changing them here
would silently diverge the two products' results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from ..config import settings
from ..logger import get_logger

logger = get_logger("localmama.brain")

#: 384-dim multilingual model covering en/hi/te. Must match what wrote the
#: vectors — a different model produces embeddings in a different space, and
#: every similarity score becomes noise rather than an error.
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VECTOR_WEIGHT = 1.0   # semantic
FTS_WEIGHT = 0.3      # title/content keyword rank
KW_WEIGHT = 0.6       # curated-keywords boost


@dataclass
class Hit:
    """One retrieved entry. `score` is the blended hybrid rank, not a probability."""

    id: int
    title: str
    content: str
    category: str | None
    city: str | None
    phone: str | None
    attrs: dict = field(default_factory=dict)
    score: float = 0.0
    #: True when this was reached by spelling rather than by a literal match.
    #: The caller is asked to confirm the name before the number is read out —
    #: a close spelling is a guess, and the cost of a wrong one is a stranger's
    #: phone ringing.
    approximate: bool = False


@lru_cache(maxsize=1)
def _embedder():
    """Loaded once per process, lazily, and cached.

    Lazily on purpose: loading it up front in every job process put the ~500MB
    model into all eight of LiveKit's idle processes at once, taking memory
    from 0.8GB to 2GB of 4GB and failing calls. `retrieve_async` runs this off
    the event loop, so a cold first lookup costs a few seconds in a worker
    thread while the agent keeps talking — and the Dockerfile bakes the model
    files into the image, so even that cold load reads from disk rather than
    fetching from HuggingFace mid-call.
    """
    from fastembed import TextEmbedding

    logger.info("loading embedding model %s", EMBED_MODEL)
    return TextEmbedding(model_name=EMBED_MODEL)


def _embed(text: str):
    return list(_embedder().embed([text]))[0]


def available() -> bool:
    """Whether the brain can be used at all. Everything degrades to None/[] if not."""
    return bool(settings.database_url)


async def retrieve_async(text: str, *, city: str | None = None, top_k: int = 3):
    """`retrieve` off the event loop.

    Both halves of a lookup are blocking C/socket work — the embedding pass and
    the psycopg round trip — so calling it directly from an async tool freezes
    the whole agent: audio stops flowing and the Realtime API starts rejecting
    overlapping responses with `conversation_already_has_active_response`.
    """
    import asyncio

    return await asyncio.to_thread(retrieve, text, city=city, top_k=top_k)


def _connect():
    import psycopg
    from pgvector.psycopg import register_vector

    conn = psycopg.connect(settings.database_url, connect_timeout=5)
    register_vector(conn)
    return conn


def retrieve(text: str, *, city: str | None = None, top_k: int = 3) -> list[Hit]:
    """Best matches for a caller's request, or [] if the brain is unreachable.

    Never raises: this sits in a live call, and a database hiccup must cost a
    lookup, not the conversation. `city` is a hard WHERE filter rather than a
    similarity term — a plumber in Chennai is not a weaker match for a Hyderabad
    caller, it is the wrong answer. Rows without a city stay eligible, because
    most listings currently have none.
    """
    if not available() or not (text or "").strip():
        return []
    floor = settings.brain_min_score
    try:
        vec = _embed(text)
        where = ["owner_id = %(owner)s", "agent_id = %(agent)s"]
        params: dict = {
            "v": vec, "q": text, "k": top_k,
            "owner": settings.brain_owner_id, "agent": settings.brain_agent_id,
        }
        if city:
            where.append("(city IS NULL OR city ILIKE %(city)s)")
            params["city"] = f"%{city}%"
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id, title, content, category, city, phone,"
                " coalesce(attrs,'{}'::jsonb),"
                f" {VECTOR_WEIGHT} * (1 - (embedding <=> %(v)s))"
                f" + {FTS_WEIGHT} * ts_rank(fts, plainto_tsquery('simple', %(q)s))"
                f" + {KW_WEIGHT} * ts_rank(to_tsvector('simple', coalesce(keywords,'')),"
                "   plainto_tsquery('simple', %(q)s)) AS score"
                " FROM utter.knowledge"
                f" WHERE {' AND '.join(where)}"
                " ORDER BY score DESC LIMIT %(k)s",
                params,
            )
            hits = [Hit(*row) for row in cur.fetchall()]
        # Hybrid retrieval always returns *something* — the top row for "fix my
        # geyser" was a tutoring service at 0.22, because the catalogue has no
        # plumbing listing at all. Offering that to a caller is worse than
        # admitting we have no match, so anything under the floor is dropped.
        kept = [h for h in hits if h.score >= floor]
        logger.info(
            "brain lookup %r -> %s (dropped %d below %.2f)",
            text[:40], [f"{h.title} {h.score:.2f}" for h in kept],
            len(hits) - len(kept), floor,
        )
        return kept
    except Exception as exc:  # noqa: BLE001 - a lookup must never end a call
        logger.warning("brain lookup failed (%s); continuing without it", exc)
        return []


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


def categories() -> set[str]:
    """Every category in the knowledge base, lowercased.

    Used to tell a category apart from a business name: "wash" and "tutors" are
    things we do, "WOW Wash" is someone we can put a caller through to.
    """
    if not available():
        return set()
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT lower(category) FROM utter.knowledge"
                " WHERE owner_id = %s AND agent_id = %s AND category IS NOT NULL",
                (settings.brain_owner_id, settings.brain_agent_id),
            )
            return {row[0] for row in cur.fetchall() if row[0]}
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not read categories: %s", exc)
        return set()


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
    import difflib

    phrase = (query or "").strip().lower()
    if not phrase:
        return None
    found = difflib.get_close_matches(
        phrase, sorted(categories()), n=1, cutoff=CATEGORY_CUTOFF
    )
    return found[0] if found else None


def find_business(name: str, limit: int = 5) -> list[Hit]:
    """Businesses matching a NAME, most literal first. No embeddings.

    A caller asking "what is Brimmies Cafe's number?" wants an exact record.
    Semantic search would hand back the nearest neighbour with a real phone
    number attached and send them to a stranger — it matched "electrician" to an
    EV charging company at 0.59, because the words look alike.
    """
    query = (name or "").strip()
    if not available() or not query:
        return []
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id, title, content, category, city, phone,"
                " coalesce(attrs,'{}'::jsonb),"
                "  CASE WHEN lower(title) = lower(%(q)s) THEN 0"
                "       WHEN lower(title) LIKE lower(%(q)s) || '%%' THEN 1"
                "       WHEN lower(title) LIKE '%%' || lower(%(q)s) || '%%' THEN 2"
                "       ELSE 3 END AS rank"
                " FROM utter.knowledge"
                " WHERE owner_id = %(owner)s AND agent_id = %(agent)s"
                "   AND (lower(title) LIKE '%%' || lower(%(q)s) || '%%'"
                "        OR lower(coalesce(keywords,'')) LIKE '%%' || lower(%(q)s) || '%%')"
                " ORDER BY rank, title LIMIT %(k)s",
                {"q": query, "k": limit,
                 "owner": settings.brain_owner_id, "agent": settings.brain_agent_id},
            )
            found = [Hit(*row) for row in cur.fetchall()]
        if found:
            logger.info("brain name lookup %r -> %s", query[:40], [h.title for h in found])
            return found
    except Exception as exc:  # noqa: BLE001
        logger.warning("name lookup failed (%s); continuing without it", exc)
        return []
    return _approximate_business(query, limit)


#: Names are matched tighter than categories. A category is a word the caller
#: chose loosely; a name is a specific record, and the cost of the nearest one
#: being wrong is a stranger's phone ringing. 0.82 keeps "Pax Jewellers" ->
#: "Pax Jwellers" (0.96) and "Cleanmates" -> "Clean Mates" (0.95) while
#: rejecting "Speed Autos" -> "Speed Kawasaki Kolkata".
NAME_CUTOFF = 0.82

#: Typography the catalogue contains but a caller cannot say or a transcriber
#: produce: a curly apostrophe in "Brinda’s kitchen", a slash in "M/S B R
#: Enterprises". Folded on BOTH sides before comparing, so neither has to be
#: cleaned up in the data.
#: Every form maps straight to a space in ONE pass. Mapping "’" to "'" and
#: "'" to " " looks equivalent and is not: `str.translate` does not re-examine
#: what it substitutes, so "Brinda’s" became "brinda's" while "Brinda's" became
#: "brinda s" — the exact pair this exists to reconcile.
_PUNCT = str.maketrans({
    ch: " " for ch in "’‘'\"“”/.,-–—()&"
})


def normalise_name(text: str) -> str:
    """A business name reduced to what a caller could actually say."""
    return " ".join((text or "").lower().translate(_PUNCT).split())


def _approximate_business(query: str, limit: int) -> list[Hit]:
    """Nearest titles by spelling, once a literal match has failed.

    Deliberately edit distance and not embeddings. The neighbourhood of a
    spelling is a handful of near-identical strings; the neighbourhood of a
    meaning is the whole catalogue, which is how "electrician" reached an EV
    charging company. Hits are flagged `approximate` so the caller is asked to
    confirm the name rather than simply read the number.
    """
    import difflib

    wanted = normalise_name(query)
    if not wanted:
        return []
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT title FROM utter.knowledge"
                " WHERE owner_id = %s AND agent_id = %s AND title IS NOT NULL",
                (settings.brain_owner_id, settings.brain_agent_id),
            )
            titles = [row[0] for row in cur.fetchall()]
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not read titles (%s); no approximate match", exc)
        return []

    scored = sorted(
        (
            (difflib.SequenceMatcher(None, wanted, normalise_name(t)).ratio(), t)
            for t in titles
        ),
        key=lambda pair: (-pair[0], pair[1]),
    )
    near = [title for score, title in scored[:limit] if score >= NAME_CUTOFF]
    if not near:
        logger.info("no business close to %r (best %.2f)",
                    query[:40], scored[0][0] if scored else 0.0)
        return []
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id, title, content, category, city, phone,"
                " coalesce(attrs,'{}'::jsonb), 0"
                " FROM utter.knowledge"
                " WHERE owner_id = %s AND agent_id = %s AND title = ANY(%s)"
                " ORDER BY title",
                (settings.brain_owner_id, settings.brain_agent_id, near),
            )
            hits = [Hit(*row) for row in cur.fetchall()]
    except Exception as exc:  # noqa: BLE001
        logger.warning("approximate lookup failed (%s)", exc)
        return []
    for hit in hits:
        hit.approximate = True
    logger.info("brain name lookup %r -> %s (approximate)",
                query[:40], [h.title for h in hits])
    return hits


def matches_for_service(
    service: str, limit: int = 3, *, city: str | None = None
) -> list[Hit]:
    """Businesses that actually do this work, with phone numbers.

    Category matches win outright over keyword matches. Without that, "cleaning"
    returned a dental clinic — "teeth cleaning" is a fair keyword for a dentist
    and a terrible answer for someone who wants their house cleaned.

    `city` narrows the result the same way `retrieve` does, and with the same
    caveat: a listing with no city on it stays eligible, because most of them
    have none and filtering them out would empty the catalogue. Both the query
    and the caller's city are English by the time they reach here — the tools
    convert them at capture — which is the whole reason a literal match can
    work at all.

    Finding nothing is a valid outcome: there is no electrician in the
    catalogue, so the WhatsApp template falls back to "our team is
    shortlisting", which is true, rather than naming a business that is wrong.
    """
    query = (service or "").strip()
    if not available() or not query:
        return []
    where = [
        "owner_id = %(owner)s", "agent_id = %(agent)s", "phone IS NOT NULL",
        "(lower(coalesce(category,'')) LIKE '%%' || lower(%(q)s) || '%%'"
        " OR lower(coalesce(keywords,'')) LIKE '%%' || lower(%(q)s) || '%%')",
    ]
    params: dict = {
        "q": query, "owner": settings.brain_owner_id, "agent": settings.brain_agent_id,
    }
    # The caller gives a locality as often as a city ("Madhapur"), and the
    # listings carry cities — so this matches either way round rather than
    # requiring the two strings to be the same shape.
    place = (city or "").strip()
    if place:
        where.append("(city IS NULL OR city ILIKE %(city)s OR %(city_plain)s ILIKE"
                     " '%%' || city || '%%')")
        params["city"] = f"%{place}%"
        params["city_plain"] = place
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id, title, content, category, city, phone,"
                " coalesce(attrs,'{}'::jsonb),"
                "  CASE WHEN lower(coalesce(category,'')) LIKE '%%' || lower(%(q)s) || '%%'"
                "       THEN 0 ELSE 1 END AS rank"
                " FROM utter.knowledge"
                f" WHERE {' AND '.join(where)}"
                " ORDER BY rank, title",
                params,
            )
            rows = [Hit(*row) for row in cur.fetchall()]
        # Nothing matched literally. Before giving up, try the nearest category
        # by spelling: translation returns American forms ("Jewelry store")
        # where the catalogue is British ("jewellery stores"), and a caller's
        # phrase is rarely a category verbatim ("mobile shop" vs "mobile
        # shops"). This is deliberately the *second* attempt — a literal hit is
        # better evidence than a close one, so it is never overridden.
        if not rows:
            nearest = closest_category(query)
            if nearest:
                logger.info("no literal match for %r; trying category %r",
                            query[:40], nearest)
                return (
                    matches_for_service(nearest, limit, city=city)
                    if nearest != query else []
                )
            return []
        # A category hit is strong evidence; keyword-only hits are noise beside it.
        by_category = [h for h in rows if h.score == 0]
        chosen = (by_category or rows)[:limit]
        logger.info("brain service match %r -> %s", query[:40], [h.title for h in chosen])
        return chosen
    except Exception as exc:  # noqa: BLE001
        logger.warning("service match failed (%s); continuing without it", exc)
        return []


async def matches_for_service_async(
    service: str, limit: int = 3, *, city: str | None = None
) -> list[Hit]:
    import asyncio

    return await asyncio.to_thread(matches_for_service, service, limit, city=city)


async def find_business_async(name: str, limit: int = 5) -> list[Hit]:
    import asyncio

    return await asyncio.to_thread(find_business, name, limit)


def options_line(hits: list[Hit]) -> str:
    """The WhatsApp template's {{4}}: names WITH numbers, or "" to fall back.

    A name alone is a teaser — the caller cannot act on it, and the number is
    the entire point of the message.
    """
    parts = [f"{h.title} {spoken_phone(h.phone)}".strip()
             for h in hits if h.title and h.phone]
    return " · ".join(parts)

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


@lru_cache(maxsize=1)
def _embedder():
    """Loaded once and cached: construction downloads and initialises the model,
    which is far too slow to do inside a call."""
    from fastembed import TextEmbedding

    logger.info("loading embedding model %s", EMBED_MODEL)
    return TextEmbedding(model_name=EMBED_MODEL)


def _embed(text: str):
    return list(_embedder().embed([text]))[0]


def available() -> bool:
    """Whether the brain can be used at all. Everything degrades to None/[] if not."""
    return bool(settings.database_url)


def warm() -> None:
    """Load the embedding model before any caller is on the line.

    Called at worker startup. Without it the model was constructed on the first
    `lookup_services` of the first call — which meant a HuggingFace download and
    ten seconds of dead air mid-conversation, long enough for the caller to ask
    whether anyone was still there. The Dockerfile bakes the files into the
    image; this pays the load cost too.
    """
    if not available():
        return
    try:
        _embed("warm")
        logger.info("embedding model ready")
    except Exception as exc:  # noqa: BLE001 - a cold lookup is better than no worker
        logger.warning("could not warm the embedding model: %s", exc)


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


def options_line(hits: list[Hit]) -> str:
    """The WhatsApp template's {{4}} line, or "" to fall back to the generic text."""
    names = [h.title for h in hits if h.title]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]

"""Text normalisation shared across the backend.

Lifted verbatim out of the agent's `languages.py`, which also carries the
spoken-alias tables for recognising which language a caller asked for. That is a
listening concern and stays with the agent; this is the comparison concern and
it is all the backend needs.

Verbatim matters here: `grounding`'s similarity cutoffs were fitted against the
output of this exact function, on real romanisation pairs. A subtly different
normaliser moves every score and silently invalidates the thresholds.
"""

from __future__ import annotations

import re
import unicodedata


def normalize_text(text: str) -> str:
    """Casefold, strip accents, and collapse punctuation/whitespace.

    Indic scripts are left intact — NFKD on Devanagari would split matras from
    their base consonants, so we only drop combining marks for Latin text.
    """
    text = text.strip().casefold()
    if text.isascii():
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^\w\sऀ-෿]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

from __future__ import annotations

import re


EXPANSIONS = {
    "sci-fi": "science fiction",
    "scifi": "science fiction",
    "feel-good": "uplifting positive comforting",
    "feel good": "uplifting positive comforting",
    "popular": "audience attention ratings reviews",
    "popularity": "audience attention ratings reviews",
    "reviews": "audience reactions opinions",
}


def rewrite_query(query: str) -> str:
    """Deterministically normalize and expand common cultural-search language."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    normalized = " ".join(query.strip().split())
    normalized = re.sub(
        r"^(?:please\s+)?(?:can you|could you|would you)\s+",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    expansions = [
        expanded
        for phrase, expanded in EXPANSIONS.items()
        if phrase in normalized.casefold()
    ]
    if expansions:
        normalized = f"{normalized} {' '.join(expansions)}"
    return normalized

"""Hybrid search — weighted combination of vector and keyword search."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, TypedDict

from kraang.embeddings import EmbeddingProvider
from kraang.models import Note, NoteSearchResult
from kraang.search import build_fts_query

if TYPE_CHECKING:
    from kraang.store import SQLiteStore

logger = logging.getLogger("kraang.hybrid")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_VECTOR_WEIGHT: float = 0.7
_TEXT_WEIGHT: float = 0.3
_MIN_SCORE: float = 0.15  # Lowered from 0.35 — FTS-only hit can score at most ~0.27
_CANDIDATE_MULTIPLIER: int = 4  # Over-fetch factor: fetch limit*N per source before merge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def bm25_score_to_normalized(score: float) -> float:
    """Normalize a positive BM25 score (higher = better) to the range (0, 1)."""
    return max(0.0, score) / (1.0 + max(0.0, score))


# ---------------------------------------------------------------------------
# Merge entry type
# ---------------------------------------------------------------------------


class _MergedEntry(TypedDict):
    note: Note
    vec_score: float
    fts_score: float
    snippet: str


# ---------------------------------------------------------------------------
# Hybrid search
# ---------------------------------------------------------------------------


async def hybrid_search(
    store: SQLiteStore,
    provider: EmbeddingProvider | None,
    query: str,
    limit: int = 10,
) -> list[NoteSearchResult]:
    """Run hybrid (vector + keyword) search, falling back to FTS-only.

    Parameters
    ----------
    store:
        A ``SQLiteStore`` instance.
    provider:
        An embedding provider, or ``None`` for FTS-only mode.
    query:
        Natural-language search query.
    limit:
        Maximum results to return.
    """
    from kraang.store import SQLiteStore  # runtime import

    if not isinstance(store, SQLiteStore):
        raise TypeError(f"Expected SQLiteStore, got {type(store).__name__}")

    candidates = limit * _CANDIDATE_MULTIPLIER

    # -- FTS-only fallback ---------------------------------------------------
    if provider is None:
        fts_expr = build_fts_query(query)
        if not fts_expr:
            return []
        return await store.search_notes(fts_expr, limit=limit)

    # -- Parallel vector + keyword search ------------------------------------
    try:
        query_embedding = await provider.embed_query(query)
    except Exception:
        logger.warning("embed_query failed, falling back to FTS-only", exc_info=True)
        fts_expr = build_fts_query(query)
        if not fts_expr:
            return []
        raw = await store.search_notes(fts_expr, limit=limit)
        return raw

    fts_expr = build_fts_query(query)

    vec_task = store.search_notes_vector(query_embedding, limit=candidates)
    if fts_expr:
        fts_task = store.search_notes(fts_expr, limit=candidates)
        vec_results, fts_results = await asyncio.gather(vec_task, fts_task)
    else:
        vec_results = await vec_task
        fts_results = []

    # -- Merge by note_id ----------------------------------------------------
    merged: dict[str, _MergedEntry] = {}

    for r in vec_results:
        nid = r.note.note_id
        merged[nid] = _MergedEntry(
            note=r.note,
            vec_score=r.score,
            fts_score=0.0,
            snippet=r.snippet,
        )

    for r in fts_results:
        nid = r.note.note_id
        if nid in merged:
            merged[nid]["fts_score"] = bm25_score_to_normalized(r.score)
            if r.snippet:
                merged[nid]["snippet"] = r.snippet
        else:
            merged[nid] = _MergedEntry(
                note=r.note,
                vec_score=0.0,
                fts_score=bm25_score_to_normalized(r.score),
                snippet=r.snippet,
            )

    # -- Compute hybrid scores -----------------------------------------------
    results: list[NoteSearchResult] = []
    for entry in merged.values():
        vec_s = entry["vec_score"]
        fts_s = entry["fts_score"]
        hybrid = _VECTOR_WEIGHT * vec_s + _TEXT_WEIGHT * fts_s

        if hybrid < _MIN_SCORE:
            continue

        results.append(
            NoteSearchResult(
                note=entry["note"],
                score=hybrid,
                snippet=entry["snippet"],
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:limit]

"""Hybrid search — weighted combination of vector and keyword search."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from kraang.embeddings import EmbeddingProvider
from kraang.models import NoteSearchResult
from kraang.search import build_fts_query

logger = logging.getLogger("kraang.hybrid")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class HybridConfig:
    """Tuning knobs for hybrid search."""

    vector_weight: float = 0.7
    text_weight: float = 0.3
    min_score: float = 0.35
    candidate_multiplier: int = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def bm25_score_to_normalized(score: float) -> float:
    """Normalize a positive BM25 score (higher = better) to the range (0, 1)."""
    return max(0.0, score) / (1.0 + max(0.0, score))


# ---------------------------------------------------------------------------
# Hybrid search
# ---------------------------------------------------------------------------


async def hybrid_search(
    store: object,
    provider: EmbeddingProvider | None,
    query: str,
    config: HybridConfig | None = None,
    limit: int = 10,
) -> list[NoteSearchResult]:
    """Run hybrid (vector + keyword) search, falling back to FTS-only.

    Parameters
    ----------
    store:
        A ``SQLiteStore`` instance (typed as ``object`` to avoid circular imports).
    provider:
        An embedding provider, or ``None`` for FTS-only mode.
    query:
        Natural-language search query.
    config:
        Hybrid search tuning. Defaults to ``HybridConfig()``.
    limit:
        Maximum results to return.
    """
    from kraang.store import SQLiteStore

    assert isinstance(store, SQLiteStore)
    cfg = config or HybridConfig()
    candidates = limit * cfg.candidate_multiplier

    # -- FTS-only fallback ---------------------------------------------------
    if provider is None:
        fts_expr = build_fts_query(query)
        if not fts_expr:
            return []
        return await store.search_notes(fts_expr, limit=limit)

    # -- Parallel vector + keyword search ------------------------------------
    query_embedding = await provider.embed_query(query)

    fts_expr = build_fts_query(query)

    vec_task = store.search_notes_vector(query_embedding, limit=candidates)
    if fts_expr:
        fts_task = store.search_notes(fts_expr, limit=candidates)
        vec_results, fts_results = await asyncio.gather(vec_task, fts_task)
    else:
        vec_results = await vec_task
        fts_results = []

    # -- Merge by note_id ----------------------------------------------------
    merged: dict[str, dict[str, object]] = {}

    for r in vec_results:
        nid = r.note.note_id
        merged[nid] = {
            "note": r.note,
            "vec_score": r.score,
            "fts_score": 0.0,
            "snippet": r.snippet,
        }

    for r in fts_results:
        nid = r.note.note_id
        if nid in merged:
            merged[nid]["fts_score"] = bm25_score_to_normalized(r.score)
            if r.snippet:
                merged[nid]["snippet"] = r.snippet
        else:
            merged[nid] = {
                "note": r.note,
                "vec_score": 0.0,
                "fts_score": bm25_score_to_normalized(r.score),
                "snippet": r.snippet,
            }

    # -- Compute hybrid scores -----------------------------------------------
    results: list[NoteSearchResult] = []
    for entry in merged.values():
        vec_s = float(entry["vec_score"])  # type: ignore[arg-type]
        fts_s = float(entry["fts_score"])  # type: ignore[arg-type]
        hybrid = cfg.vector_weight * vec_s + cfg.text_weight * fts_s

        note = entry["note"]  # type: ignore[assignment]
        # Note: relevance weighting is already applied by search_notes() and
        # search_notes_vector(), so we do NOT multiply again here.

        if hybrid < cfg.min_score:
            continue

        results.append(
            NoteSearchResult(
                note=note,  # type: ignore[arg-type]
                score=hybrid,
                snippet=str(entry["snippet"]),
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:limit]

"""Maximal Marginal Relevance — diversity re-ranking for search results."""

from __future__ import annotations

import re

from kraang.models import NoteSearchResult


def tokenize(text: str) -> set[str]:
    """Lowercase-tokenize *text* into a set of alphanumeric/underscore tokens."""
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Return the Jaccard index of two token sets (0.0 when both are empty)."""
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def mmr_rerank(
    results: list[NoteSearchResult],
    lambda_: float = 0.7,
) -> list[NoteSearchResult]:
    """Re-rank *results* using Maximal Marginal Relevance.

    Higher *lambda_* favours relevance; lower values favour diversity.
    Returns a **new** list — the input is not mutated.
    """
    if len(results) <= 1:
        return list(results)

    # Pre-tokenize note contents.
    tokens = [tokenize(r.note.content) for r in results]

    # Normalize scores to [0, 1].
    scores = [r.score for r in results]
    max_score = max(scores)
    min_score = min(scores)
    span = max_score - min_score
    normed = (
        [1.0] * len(scores) if span == 0
        else [(s - min_score) / span for s in scores]
    )

    selected: list[int] = []
    remaining = set(range(len(results)))

    for _ in range(len(results)):
        best_idx = -1
        best_mmr = -float("inf")

        for idx in remaining:
            relevance = normed[idx]
            if selected:
                max_sim = max(
                    jaccard_similarity(tokens[idx], tokens[s]) for s in selected
                )
            else:
                max_sim = 0.0

            mmr_score = lambda_ * relevance - (1 - lambda_) * max_sim
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = idx

        selected.append(best_idx)
        remaining.discard(best_idx)

    return [results[i] for i in selected]

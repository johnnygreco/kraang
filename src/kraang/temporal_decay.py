"""Temporal decay scoring — penalize stale content in search results."""

from __future__ import annotations

import math

from kraang.models import NoteSearchResult, utcnow


def decay_multiplier(age_days: float, half_life_days: float = 30.0) -> float:
    """Return an exponential decay multiplier in (0, 1].

    After *half_life_days* the multiplier equals 0.5; it never goes negative.
    """
    return math.exp(-math.log(2) / half_life_days * max(0.0, age_days))


def apply_temporal_decay(
    results: list[NoteSearchResult],
    half_life_days: float = 30.0,
    exempt_tags: set[str] | None = None,
) -> list[NoteSearchResult]:
    """Multiply each result's score by an age-based decay factor (in-place).

    Notes whose tags intersect *exempt_tags* are left untouched.
    Returns the same list for convenience.
    """
    if exempt_tags is None:
        exempt_tags = {"evergreen", "pinned"}

    now = utcnow()
    for result in results:
        if exempt_tags and exempt_tags.intersection(result.note.tags):
            continue
        age_days = (now - result.note.updated_at).total_seconds() / 86400
        result.score *= decay_multiplier(age_days, half_life_days)

    return results

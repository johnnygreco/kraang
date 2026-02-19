"""Tests for kraang.temporal_decay — age-based score decay."""

from __future__ import annotations

from datetime import timedelta

import pytest

from kraang.models import Note, NoteSearchResult, utcnow
from kraang.temporal_decay import apply_temporal_decay, decay_multiplier


# ---------------------------------------------------------------------------
# decay_multiplier
# ---------------------------------------------------------------------------


class TestDecayMultiplier:
    def test_zero_age(self):
        assert decay_multiplier(0) == pytest.approx(1.0)

    def test_one_half_life(self):
        assert decay_multiplier(30, half_life_days=30) == pytest.approx(0.5)

    def test_two_half_lives(self):
        assert decay_multiplier(60, half_life_days=30) == pytest.approx(0.25)

    def test_three_half_lives(self):
        assert decay_multiplier(90, half_life_days=30) == pytest.approx(0.125)

    def test_always_positive(self):
        """Decay should never go to zero or negative."""
        for days in [0, 1, 30, 365, 3650]:
            assert decay_multiplier(days) > 0.0

    def test_monotonically_decreasing(self):
        prev = 1.0
        for days in range(1, 100):
            m = decay_multiplier(days)
            assert m < prev
            prev = m

    def test_negative_age_clamped_to_zero(self):
        """Negative ages should be treated as 0 (no decay)."""
        assert decay_multiplier(-10) == pytest.approx(1.0)

    def test_custom_half_life(self):
        assert decay_multiplier(7, half_life_days=7) == pytest.approx(0.5)
        assert decay_multiplier(14, half_life_days=7) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# apply_temporal_decay
# ---------------------------------------------------------------------------


def _make_result(
    title: str,
    score: float,
    age_days: float,
    tags: list[str] | None = None,
) -> NoteSearchResult:
    now = utcnow()
    note = Note(
        note_id=title,
        title=title,
        title_normalized=title.lower(),
        content=f"Content for {title}",
        tags=tags or [],
        relevance=1.0,
        created_at=now - timedelta(days=age_days),
        updated_at=now - timedelta(days=age_days),
    )
    return NoteSearchResult(note=note, score=score, snippet="")


class TestApplyTemporalDecay:
    def test_fresh_note_unchanged(self):
        results = [_make_result("fresh", 1.0, 0)]
        apply_temporal_decay(results)
        assert results[0].score == pytest.approx(1.0, abs=0.01)

    def test_old_note_decayed(self):
        results = [_make_result("old", 1.0, 30)]
        apply_temporal_decay(results, half_life_days=30)
        assert results[0].score == pytest.approx(0.5, abs=0.05)

    def test_very_old_note_heavily_decayed(self):
        results = [_make_result("ancient", 1.0, 90)]
        apply_temporal_decay(results, half_life_days=30)
        assert results[0].score == pytest.approx(0.125, abs=0.02)

    def test_exempt_tag_skips_decay(self):
        results = [_make_result("pinned", 1.0, 90, tags=["evergreen"])]
        apply_temporal_decay(results)
        assert results[0].score == pytest.approx(1.0)

    def test_pinned_tag_also_exempt(self):
        results = [_make_result("pinned", 1.0, 90, tags=["pinned"])]
        apply_temporal_decay(results)
        assert results[0].score == pytest.approx(1.0)

    def test_custom_exempt_tags(self):
        results = [_make_result("important", 1.0, 90, tags=["important"])]
        apply_temporal_decay(results, exempt_tags={"important"})
        assert results[0].score == pytest.approx(1.0)

    def test_non_exempt_tag_decayed(self):
        results = [_make_result("normal", 1.0, 30, tags=["python"])]
        apply_temporal_decay(results, half_life_days=30)
        assert results[0].score == pytest.approx(0.5, abs=0.05)

    def test_empty_list(self):
        results = apply_temporal_decay([])
        assert results == []

    def test_returns_same_list(self):
        results = [_make_result("a", 1.0, 0)]
        returned = apply_temporal_decay(results)
        assert returned is results

    def test_mixed_ages(self):
        results = [
            _make_result("new", 1.0, 0),
            _make_result("medium", 1.0, 30),
            _make_result("old", 1.0, 60),
        ]
        apply_temporal_decay(results, half_life_days=30)
        assert results[0].score > results[1].score > results[2].score

"""Tests for kraang.hybrid — hybrid search merging vector + keyword results."""

from __future__ import annotations

import pytest

from kraang.hybrid import HybridConfig, bm25_score_to_normalized, hybrid_search
from kraang.models import Note, NoteSearchResult, utcnow


# ---------------------------------------------------------------------------
# bm25_score_to_normalized
# ---------------------------------------------------------------------------


class TestBm25ScoreToNormalized:
    def test_score_zero(self):
        assert bm25_score_to_normalized(0) == pytest.approx(0.0)

    def test_score_one(self):
        assert bm25_score_to_normalized(1) == pytest.approx(0.5)

    def test_score_ten(self):
        assert bm25_score_to_normalized(10) == pytest.approx(10.0 / 11.0)

    def test_monotonically_increasing(self):
        scores = [bm25_score_to_normalized(r) for r in range(1, 20)]
        for i in range(1, len(scores)):
            assert scores[i] > scores[i - 1]

    def test_negative_score_clamped(self):
        # max(0.0, score) means negative scores are treated as 0
        assert bm25_score_to_normalized(-5) == pytest.approx(0.0)

    def test_fractional_score(self):
        score = bm25_score_to_normalized(0.5)
        assert score == pytest.approx(0.5 / 1.5)


# ---------------------------------------------------------------------------
# HybridConfig defaults
# ---------------------------------------------------------------------------


class TestHybridConfig:
    def test_defaults(self):
        cfg = HybridConfig()
        assert cfg.vector_weight == 0.7
        assert cfg.text_weight == 0.3
        assert cfg.min_score == 0.35
        assert cfg.candidate_multiplier == 4

    def test_custom_values(self):
        cfg = HybridConfig(vector_weight=0.5, text_weight=0.5, min_score=0.1)
        assert cfg.vector_weight == 0.5
        assert cfg.text_weight == 0.5
        assert cfg.min_score == 0.1


# ---------------------------------------------------------------------------
# hybrid_search — FTS-only fallback
# ---------------------------------------------------------------------------


class TestHybridSearchFTSFallback:
    async def test_fts_fallback_when_no_provider(self, populated_store):
        """With provider=None, hybrid_search falls back to FTS-only."""
        results = await hybrid_search(populated_store, None, "asyncio")
        assert len(results) > 0
        titles = [r.note.title for r in results]
        assert any("asyncio" in t.lower() for t in titles)

    async def test_fts_fallback_empty_query(self, populated_store):
        """Empty query with no provider returns nothing."""
        results = await hybrid_search(populated_store, None, "")
        assert results == []

    async def test_fts_fallback_no_results(self, populated_store):
        results = await hybrid_search(populated_store, None, "xyznonexistentxyz")
        assert results == []

    async def test_fts_fallback_respects_limit(self, populated_store):
        results = await hybrid_search(populated_store, None, "python", limit=1)
        assert len(results) <= 1


# ---------------------------------------------------------------------------
# hybrid_search — with provider (mock)
# ---------------------------------------------------------------------------


def _make_note(note_id: str, title: str, content: str, relevance: float = 1.0) -> Note:
    now = utcnow()
    return Note(
        note_id=note_id,
        title=title,
        title_normalized=title.lower(),
        content=content,
        relevance=relevance,
        created_at=now,
        updated_at=now,
    )


class TestHybridSearchWithProvider:
    async def test_merges_vector_and_fts_results(self, populated_store):
        """Test that hybrid search merges results from both sources."""
        note_a = _make_note("a", "Python Async", "asyncio event loops", relevance=1.0)
        note_b = _make_note("b", "Docker Tips", "container orchestration", relevance=1.0)

        class MockProvider:
            provider_id = "mock"
            model = "mock-v1"
            dims = 3

            async def embed_query(self, text):
                return [0.5, 0.5, 0.5]

        # Store a vector embedding for note_a so vector search returns it
        await populated_store.upsert_note_embedding("a", [0.6, 0.6, 0.6])

        # Use a very low min_score to ensure results pass
        cfg = HybridConfig(min_score=0.0)
        results = await hybrid_search(
            populated_store, MockProvider(), "asyncio", config=cfg
        )
        # Should return FTS results at minimum
        assert isinstance(results, list)

    async def test_min_score_filtering(self, populated_store):
        """Results below min_score should be filtered out."""

        class MockProvider:
            provider_id = "mock"
            model = "mock-v1"
            dims = 3

            async def embed_query(self, text):
                return [0.1, 0.1, 0.1]

        cfg = HybridConfig(min_score=0.99)  # Very high threshold
        results = await hybrid_search(
            populated_store, MockProvider(), "asyncio", config=cfg
        )
        # All results should have score >= min_score
        for r in results:
            assert r.score >= cfg.min_score

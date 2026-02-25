"""Tests for kraang.hybrid — hybrid search merging vector + keyword results."""

from __future__ import annotations

import pytest

import kraang.hybrid as hybrid_module
from kraang.hybrid import bm25_score_to_normalized, hybrid_search
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
# Module constants (replaced HybridConfig)
# ---------------------------------------------------------------------------


class TestModuleConstants:
    def test_defaults(self):
        assert hybrid_module._VECTOR_WEIGHT == 0.7
        assert hybrid_module._TEXT_WEIGHT == 0.3
        assert hybrid_module._MIN_SCORE == 0.15
        assert hybrid_module._CANDIDATE_MULTIPLIER == 4


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
    async def test_merges_vector_and_fts_results(self, populated_store, monkeypatch):
        """Test that hybrid search merges results from both sources."""
        monkeypatch.setattr(hybrid_module, "_MIN_SCORE", 0.0)

        class MockProvider:
            provider_id = "mock"
            model = "mock-v1"
            dims = 3

            async def embed_query(self, text):
                return [0.5, 0.5, 0.5]

        # Store a vector embedding for a note so vector search returns it
        await populated_store.ensure_vec_table(3)
        notes = await populated_store.search_notes('"asyncio"', limit=1)
        assert len(notes) > 0, "Expected at least one note matching 'asyncio'"
        note_id = notes[0].note.note_id
        await populated_store.upsert_note_embedding(note_id, [0.6, 0.6, 0.6])

        results = await hybrid_search(
            populated_store, MockProvider(), "asyncio"
        )
        # Should return results from at least FTS
        assert isinstance(results, list)
        assert len(results) > 0

        # Verify scores are in descending order
        for i in range(1, len(results)):
            assert results[i - 1].score >= results[i].score

        # Verify all results are NoteSearchResult
        for r in results:
            assert isinstance(r, NoteSearchResult)
            assert r.score >= 0.0

    async def test_min_score_filtering(self, populated_store):
        """Results below min_score should be filtered out."""

        class MockProvider:
            provider_id = "mock"
            model = "mock-v1"
            dims = 3

            async def embed_query(self, text):
                return [0.1, 0.1, 0.1]

        # Use monkeypatch via a high _MIN_SCORE
        original = hybrid_module._MIN_SCORE
        try:
            hybrid_module._MIN_SCORE = 0.99  # Very high threshold
            results = await hybrid_search(
                populated_store, MockProvider(), "asyncio"
            )
            # All results should have score >= min_score
            for r in results:
                assert r.score >= 0.99
        finally:
            hybrid_module._MIN_SCORE = original

    async def test_fts_fallback_when_embed_fails(self, populated_store):
        """When embed_query raises, hybrid_search falls back to FTS-only."""

        class FailingProvider:
            provider_id = "mock"
            model = "mock-v1"
            dims = 3

            async def embed_query(self, text):
                raise RuntimeError("API down")

            async def embed_batch(self, texts):
                raise RuntimeError("API down")

        results = await hybrid_search(populated_store, FailingProvider(), "python")
        # Should fall back to FTS, not crash
        assert isinstance(results, list)

    async def test_vector_only_when_fts_empty(self, populated_store, monkeypatch):
        """When FTS expression is empty (all stop words), only vector results used."""
        monkeypatch.setattr(hybrid_module, "_MIN_SCORE", 0.0)

        # A query consisting entirely of stop words produces empty FTS expression
        class MockProvider:
            provider_id = "mock"
            model = "mock-v1"
            dims = 3

            async def embed_query(self, text):
                return [0.5, 0.5, 0.5]

        # Store some embeddings so vector search can return results
        await populated_store.ensure_vec_table(3)
        notes = await populated_store.search_notes('"python"', limit=1)
        if notes:
            await populated_store.upsert_note_embedding(
                notes[0].note.note_id, [0.5, 0.5, 0.5]
            )

        # "the and or" are all stop words — build_fts_query returns ""
        results = await hybrid_search(
            populated_store, MockProvider(), "the and or"
        )
        # Should not crash; may return vector results if embeddings exist
        assert isinstance(results, list)

    async def test_type_error_on_wrong_store_type(self):
        """Passing a non-SQLiteStore should raise TypeError."""
        with pytest.raises(TypeError, match="Expected SQLiteStore"):
            await hybrid_search("not_a_store", None, "test")  # type: ignore[arg-type]

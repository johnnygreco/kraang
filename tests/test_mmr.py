"""Tests for kraang.mmr — Maximal Marginal Relevance re-ranking."""

from __future__ import annotations

import pytest

from kraang.mmr import jaccard_similarity, mmr_rerank, tokenize
from kraang.models import Note, NoteSearchResult, utcnow


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_basic_text(self):
        tokens = tokenize("Hello World")
        assert tokens == {"hello", "world"}

    def test_lowercase(self):
        tokens = tokenize("Python ASYNC programming")
        assert "python" in tokens
        assert "async" in tokens
        assert "programming" in tokens

    def test_empty_string(self):
        assert tokenize("") == set()

    def test_special_characters(self):
        tokens = tokenize("hello! @world #test $$$")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_underscores_preserved(self):
        tokens = tokenize("my_variable another_one")
        assert "my_variable" in tokens
        assert "another_one" in tokens

    def test_numbers(self):
        tokens = tokenize("python3 version 2")
        assert "python3" in tokens
        assert "version" in tokens
        assert "2" in tokens

    def test_punctuation_only(self):
        assert tokenize("!@#$%^&*()") == set()

    def test_mixed_content(self):
        tokens = tokenize("Use asyncio.run() for Python 3.7+")
        assert "asyncio" in tokens
        assert "run" in tokens
        assert "python" in tokens


# ---------------------------------------------------------------------------
# jaccard_similarity
# ---------------------------------------------------------------------------


class TestJaccardSimilarity:
    def test_identical_sets(self):
        a = {"hello", "world"}
        assert jaccard_similarity(a, a) == pytest.approx(1.0)

    def test_disjoint_sets(self):
        a = {"hello", "world"}
        b = {"foo", "bar"}
        assert jaccard_similarity(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = {"a", "b", "c"}
        b = {"b", "c", "d"}
        # intersection = {b, c} = 2, union = {a, b, c, d} = 4
        assert jaccard_similarity(a, b) == pytest.approx(0.5)

    def test_both_empty(self):
        assert jaccard_similarity(set(), set()) == pytest.approx(0.0)

    def test_one_empty(self):
        assert jaccard_similarity({"a"}, set()) == pytest.approx(0.0)
        assert jaccard_similarity(set(), {"a"}) == pytest.approx(0.0)

    def test_subset(self):
        a = {"a", "b"}
        b = {"a", "b", "c"}
        # intersection = 2, union = 3
        assert jaccard_similarity(a, b) == pytest.approx(2.0 / 3.0)

    def test_single_element_match(self):
        assert jaccard_similarity({"x"}, {"x"}) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# mmr_rerank
# ---------------------------------------------------------------------------


def _make_result(title: str, content: str, score: float) -> NoteSearchResult:
    now = utcnow()
    note = Note(
        note_id=title,
        title=title,
        title_normalized=title.lower(),
        content=content,
        relevance=1.0,
        created_at=now,
        updated_at=now,
    )
    return NoteSearchResult(note=note, score=score, snippet="")


class TestMmrRerank:
    def test_empty_input(self):
        assert mmr_rerank([]) == []

    def test_single_result(self):
        results = [_make_result("A", "hello world", 1.0)]
        reranked = mmr_rerank(results)
        assert len(reranked) == 1
        assert reranked[0].note.title == "A"

    def test_preserves_all_results(self):
        results = [
            _make_result("A", "python programming async", 0.9),
            _make_result("B", "docker kubernetes deployment", 0.8),
            _make_result("C", "python testing pytest", 0.7),
        ]
        reranked = mmr_rerank(results)
        assert len(reranked) == len(results)
        titles = {r.note.title for r in reranked}
        assert titles == {"A", "B", "C"}

    def test_returns_new_list(self):
        results = [
            _make_result("A", "hello", 1.0),
            _make_result("B", "world", 0.5),
        ]
        reranked = mmr_rerank(results)
        assert reranked is not results

    def test_does_not_mutate_input(self):
        results = [
            _make_result("A", "hello world", 1.0),
            _make_result("B", "goodbye world", 0.5),
        ]
        original_scores = [r.score for r in results]
        mmr_rerank(results)
        assert [r.score for r in results] == original_scores

    def test_diversity_pushes_similar_apart(self):
        """With low lambda (diversity mode), similar items should be pushed apart."""
        results = [
            _make_result("A", "python async programming patterns", 1.0),
            _make_result("B", "python async programming guide", 0.9),
            _make_result("C", "docker kubernetes deployment cloud", 0.8),
        ]
        # With high lambda (relevance-focused), A should be first
        high_lambda = mmr_rerank(results, lambda_=0.99)
        assert high_lambda[0].note.title == "A"

        # With low lambda (diversity-focused), C should move up
        low_lambda = mmr_rerank(results, lambda_=0.1)
        # C should appear before B since B is very similar to A
        c_pos_low = next(i for i, r in enumerate(low_lambda) if r.note.title == "C")
        b_pos_low = next(i for i, r in enumerate(low_lambda) if r.note.title == "B")
        assert c_pos_low < b_pos_low

    def test_identical_scores(self):
        results = [
            _make_result("A", "same content here", 1.0),
            _make_result("B", "same content here", 1.0),
        ]
        reranked = mmr_rerank(results)
        assert len(reranked) == 2

    def test_highest_relevance_first_high_lambda(self):
        """With lambda close to 1.0, highest relevance should be first."""
        results = [
            _make_result("Low", "low relevance item content", 0.1),
            _make_result("High", "high relevance item different", 1.0),
            _make_result("Mid", "medium relevance something else", 0.5),
        ]
        reranked = mmr_rerank(results, lambda_=0.99)
        assert reranked[0].note.title == "High"

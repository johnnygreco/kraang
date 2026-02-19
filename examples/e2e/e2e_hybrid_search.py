"""End-to-end test for hybrid search — vector + keyword fusion with fallbacks.

Exercises hybrid_search with a FakeEmbeddingProvider, FTS-only fallback
(provider=None), provider-failure fallback, bm25_score_to_normalized helper,
and _MIN_SCORE filtering.

Run:
    python examples/e2e/e2e_hybrid_search.py
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import sys

from kraang.hybrid import hybrid_search, bm25_score_to_normalized
from kraang.search import build_fts_query
from kraang.store import SQLiteStore

# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

_passed = 0
_failed = 0


def _check(name: str, condition: bool, detail: str = "") -> None:
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  [PASS] {name}")
    else:
        _failed += 1
        reason = f": {detail}" if detail else ""
        print(f"  [FAIL] {name}{reason}")


# ---------------------------------------------------------------------------
# Fake embedding provider
# ---------------------------------------------------------------------------

_DIMS = 8


def _hash_to_vec(text: str) -> list[float]:
    """Deterministic 8-dim vector from text words.

    Each word hashes (via MD5) to an index in the vector, accumulating weight.
    The result is L2-normalized so cosine similarity == dot product.
    """
    vec = [0.0] * _DIMS
    for word in text.lower().split():
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        idx = h % _DIMS
        vec[idx] += 1.0
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


class FakeEmbeddingProvider:
    """Deterministic embedding provider for testing."""

    @property
    def provider_id(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return "test-model"

    @property
    def dims(self) -> int:
        return _DIMS

    async def embed_query(self, text: str) -> list[float]:
        return _hash_to_vec(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [_hash_to_vec(t) for t in texts]


class FailingEmbeddingProvider(FakeEmbeddingProvider):
    """Provider whose embed_query always raises."""

    async def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("Simulated embedding API failure")

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("Simulated embedding API failure")


# ---------------------------------------------------------------------------
# Sample corpus (12 notes across categories)
# ---------------------------------------------------------------------------

NOTES: list[tuple[str, str, list[str], str]] = [
    ("Python asyncio patterns", "Event loops coroutines and tasks are core asyncio primitives for async programming.", ["python", "async"], "engineering"),
    ("SQLite FTS5 guide", "FTS5 is a full-text search extension for SQLite using BM25 ranking algorithm.", ["sqlite", "search"], "engineering"),
    ("Weekly grocery list", "Milk eggs bread avocados chicken rice and fresh vegetables.", ["shopping", "food"], "personal"),
    ("MCP protocol overview", "Model Context Protocol enables LLMs to interact with external tools and data.", ["mcp", "ai"], "engineering"),
    ("Meditation practice", "Focus on breath for ten minutes. Body scan technique helps relaxation.", ["health", "mindfulness"], "wellness"),
    ("Docker compose tips", "Use volumes for persistent data. Networks for service communication.", ["docker", "devops"], "engineering"),
    ("Investment portfolio", "Index funds bonds international equities alternatives rebalance quarterly.", ["finance"], "finance"),
    ("Machine learning basics", "Supervised unsupervised learning neural networks gradient descent backpropagation.", ["ml", "ai"], "learning"),
    ("REST API design", "Use nouns for resources HTTP verbs for actions pagination filtering versioning.", ["api", "rest"], "engineering"),
    ("Pydantic v2 migration", "model_validator replaces root_validator ConfigDict replaces class Config.", ["python", "pydantic"], "engineering"),
    ("Morning routine habits", "Wake early meditate exercise cold shower healthy breakfast productivity.", ["health", "routine"], "wellness"),
    ("Garden planting schedule", "Tomatoes in March herbs in April squash in May water daily in summer.", ["garden", "plants"], "personal"),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def populate(store: SQLiteStore, provider: FakeEmbeddingProvider) -> dict[str, str]:
    """Insert all notes and embed them. Returns {title: note_id}."""
    ids: dict[str, str] = {}
    for title, content, tags, category in NOTES:
        note, _ = await store.upsert_note(title, content, tags, category)
        ids[title] = note.note_id

    # Embed all notes
    for title, content, _tags, _cat in NOTES:
        nid = ids[title]
        vec = await provider.embed_query(f"{title} {content}")
        await store.upsert_note_embedding(nid, vec)

    return ids


async def test_hybrid_search_basic(store: SQLiteStore, provider: FakeEmbeddingProvider) -> None:
    """Hybrid search should return results combining vector + FTS scores."""
    print("\n--- hybrid_search (basic) ---")
    results = await hybrid_search(store, provider, "Python asyncio programming", limit=5)
    _check("returns results", len(results) > 0, f"got {len(results)}")
    if results:
        _check("scores are positive", all(r.score > 0 for r in results))
        _check("results sorted by score desc", all(
            results[i].score >= results[i + 1].score
            for i in range(len(results) - 1)
        ))
        # The asyncio note should be highly relevant
        top_titles = [r.note.title for r in results[:3]]
        _check(
            "asyncio note in top 3",
            any("asyncio" in t.lower() for t in top_titles),
            f"top3={top_titles}",
        )


async def test_hybrid_semantic_ranking(store: SQLiteStore, provider: FakeEmbeddingProvider) -> None:
    """Semantic similarity should influence ranking beyond keyword overlap."""
    print("\n--- hybrid_search (semantic ranking) ---")
    # Search for "machine learning neural networks" — should rank ML note high
    results = await hybrid_search(store, provider, "machine learning neural networks", limit=5)
    _check("ML query returns results", len(results) > 0)
    if results:
        top_titles = [r.note.title for r in results[:2]]
        _check(
            "ML note ranks in top 2",
            any("machine learning" in t.lower() for t in top_titles),
            f"top2={top_titles}",
        )

    # Search for "healthy morning exercise" — wellness notes should rank
    results2 = await hybrid_search(store, provider, "healthy morning exercise meditation", limit=5)
    _check("wellness query returns results", len(results2) > 0)
    if results2:
        categories = [r.note.category for r in results2[:3]]
        _check(
            "wellness notes in top 3",
            "wellness" in categories,
            f"top3_cats={categories}",
        )


async def test_fts_only_fallback(store: SQLiteStore) -> None:
    """With provider=None, hybrid_search should fall back to FTS-only."""
    print("\n--- hybrid_search (FTS-only fallback) ---")
    results = await hybrid_search(store, None, "SQLite FTS5 search", limit=5)
    _check("FTS-only returns results", len(results) > 0, f"got {len(results)}")
    if results:
        _check("top result is SQLite note", "sqlite" in results[0].note.title.lower(), f"got '{results[0].note.title}'")
        _check("scores are positive", all(r.score > 0 for r in results))


async def test_provider_failure_fallback(store: SQLiteStore) -> None:
    """When provider.embed_query raises, should fall back to FTS-only."""
    print("\n--- hybrid_search (provider failure fallback) ---")
    failing = FailingEmbeddingProvider()
    results = await hybrid_search(store, failing, "Docker compose volumes", limit=5)
    _check("failure fallback returns results", len(results) > 0, f"got {len(results)}")
    if results:
        _check(
            "docker note found via FTS fallback",
            any("docker" in r.note.title.lower() for r in results),
            f"titles={[r.note.title for r in results]}",
        )


async def test_bm25_score_normalized() -> None:
    """Test bm25_score_to_normalized helper function."""
    print("\n--- bm25_score_to_normalized ---")
    # score=0 → 0.0
    _check("score=0 → 0.0", bm25_score_to_normalized(0.0) == 0.0)
    # Negative score → 0.0 (clamped)
    _check("score=-5 → 0.0", bm25_score_to_normalized(-5.0) == 0.0)
    # score=1 → 0.5
    _check("score=1 → 0.5", abs(bm25_score_to_normalized(1.0) - 0.5) < 1e-9)
    # Large score → approaches 1.0
    big = bm25_score_to_normalized(100.0)
    _check("score=100 → ~1.0", big > 0.99, f"got {big}")
    # Monotonically increasing
    scores = [bm25_score_to_normalized(s) for s in [0.1, 0.5, 1.0, 5.0, 10.0]]
    _check("monotonically increasing", all(scores[i] < scores[i + 1] for i in range(len(scores) - 1)), f"scores={scores}")
    # Always in [0, 1)
    _check("output in [0, 1)", all(0.0 <= s < 1.0 for s in scores))


async def test_min_score_filtering(store: SQLiteStore, provider: FakeEmbeddingProvider) -> None:
    """Results below _MIN_SCORE should be filtered out."""
    print("\n--- _MIN_SCORE filtering ---")
    # Search for a very specific term that most notes don't match
    results = await hybrid_search(store, provider, "garden planting tomatoes squash", limit=20)
    # Should return some results but not all 12 notes (low-relevance ones filtered)
    _check("filtered results not empty", len(results) > 0, f"got {len(results)}")
    _check("not all notes returned (filtering active)", len(results) < len(NOTES), f"got {len(results)} of {len(NOTES)}")
    if results:
        # All returned results should have meaningful scores
        min_returned = min(r.score for r in results)
        _check("all returned scores >= 0.15", min_returned >= 0.15, f"min={min_returned}")


async def test_empty_query(store: SQLiteStore, provider: FakeEmbeddingProvider) -> None:
    """Empty/whitespace queries should return empty results gracefully."""
    print("\n--- empty query handling ---")
    results = await hybrid_search(store, provider, "   ", limit=5)
    _check("whitespace query returns empty", len(results) == 0, f"got {len(results)}")

    results2 = await hybrid_search(store, None, "", limit=5)
    _check("empty string returns empty", len(results2) == 0, f"got {len(results2)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    print("=" * 60)
    print("E2E: Hybrid Search — Vector + FTS Fusion with Fallbacks")
    print("=" * 60)

    store = SQLiteStore(":memory:")
    await store.initialize()
    provider = FakeEmbeddingProvider()

    try:
        await populate(store, provider)

        await test_hybrid_search_basic(store, provider)
        await test_hybrid_semantic_ranking(store, provider)
        await test_fts_only_fallback(store)
        await test_provider_failure_fallback(store)
        await test_bm25_score_normalized()
        await test_min_score_filtering(store, provider)
        await test_empty_query(store, provider)
    finally:
        await store.close()

    print("\n" + "=" * 60)
    print(f"Results: {_passed} passed, {_failed} failed")
    print("=" * 60)
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    asyncio.run(main())

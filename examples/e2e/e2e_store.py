"""End-to-end test for SQLiteStore — CRUD, FTS search, and embedding cache.

Exercises the core SQLiteStore API: note upsert/update, FTS5 full-text search,
relevance (forget), counting helpers, tag/category aggregation, recent/stale
queries, embedding cache round-trip, and brute-force vector search.

Run:
    python examples/e2e/e2e_store.py
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import struct
import sys

from kraang.search import build_fts_query
from kraang.store import SQLiteStore, content_hash

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
# Sample data
# ---------------------------------------------------------------------------

NOTES = [
    ("Python asyncio patterns", "Event loops, coroutines, and tasks.", ["python", "async"], "engineering"),
    ("SQLite FTS5 guide", "FTS5 full-text search extension for SQLite with BM25.", ["sqlite", "search"], "engineering"),
    ("Weekly grocery list", "Milk, eggs, bread, avocados, chicken.", ["shopping", "food"], "personal"),
    ("MCP protocol overview", "Model Context Protocol for LLM tool interaction.", ["mcp", "ai"], "engineering"),
    ("Meditation practice", "Focus on breath. Body scan for relaxation.", ["health", "mindfulness"], "wellness"),
    ("Docker compose tips", "Volumes for data. Networks for services. Health checks.", ["docker", "devops"], "engineering"),
    ("Investment portfolio", "60% index funds, 20% bonds, 10% international.", ["finance"], "finance"),
    ("Machine learning basics", "Supervised learning, neural networks, gradient descent.", ["ml", "ai"], "learning"),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_upsert_create(store: SQLiteStore) -> dict[str, str]:
    """Insert all sample notes, return {title: note_id}."""
    print("\n--- upsert_note (create) ---")
    ids: dict[str, str] = {}
    for title, content, tags, category in NOTES:
        note, created = await store.upsert_note(title, content, tags, category)
        ids[title] = note.note_id
        _check(
            f"create '{title[:30]}'",
            created and note.title == title and note.content == content,
            f"created={created}, title_match={note.title == title}",
        )
    return ids


async def test_upsert_update(store: SQLiteStore) -> None:
    """Update an existing note and verify updated_at changes."""
    print("\n--- upsert_note (update) ---")
    title = "Python asyncio patterns"
    original, _ = await store.upsert_note(title, "original content", ["python"], "engineering")
    # Small delay to ensure updated_at differs
    await asyncio.sleep(0.05)
    updated, created = await store.upsert_note(title, "updated content with new info", ["python", "async"], "engineering")
    _check("update returns created=False", not created)
    _check("note_id preserved", original.note_id == updated.note_id)
    _check("content updated", updated.content == "updated content with new info")
    _check("updated_at changed", updated.updated_at > original.updated_at)


async def test_fts_search(store: SQLiteStore) -> None:
    """Search via FTS5 and verify results."""
    print("\n--- FTS search ---")
    # Search for "docker volumes" — not affected by the upsert-update test
    expr = build_fts_query("docker volumes")
    _check("build_fts_query non-empty", bool(expr), f"expr={expr!r}")

    results = await store.search_notes(expr, limit=5)
    _check("search returns results", len(results) > 0, f"got {len(results)}")
    if results:
        top = results[0]
        _check("top result is docker note", "docker" in top.note.title.lower(), f"got '{top.note.title}'")
        _check("score is positive", top.score > 0, f"score={top.score}")
        _check("snippet present", isinstance(top.snippet, str))

    # Search for something in a specific category
    expr2 = build_fts_query("SQLite FTS5 BM25")
    results2 = await store.search_notes(expr2, limit=3)
    _check("sqlite search returns results", len(results2) > 0)
    if results2:
        _check("top result is FTS5 note", "fts5" in results2[0].note.title.lower(), f"got '{results2[0].note.title}'")


async def test_find_similar_titles(store: SQLiteStore) -> None:
    """Test fuzzy title matching."""
    print("\n--- find_similar_titles ---")
    # Use words that directly appear in titles (FTS5 requires exact token match)
    similar = await store.find_similar_titles("Docker compose")
    _check("finds similar titles", len(similar) > 0, f"got {len(similar)}")
    if similar:
        titles = [n.title for n in similar]
        _check("docker note in similar", any("docker" in t.lower() for t in titles), f"titles={titles}")


async def test_set_relevance(store: SQLiteStore) -> None:
    """Forget a note and verify it's excluded from search."""
    print("\n--- set_relevance (forget) ---")
    title = "Weekly grocery list"
    note = await store.set_relevance(title, 0.0)
    _check("set_relevance returns note", note is not None)
    if note:
        _check("relevance is 0.0", note.relevance == 0.0, f"got {note.relevance}")

    # Verify forgotten note excluded from FTS search
    expr = build_fts_query("grocery milk eggs")
    results = await store.search_notes(expr, limit=10)
    found_titles = [r.note.title for r in results]
    _check("forgotten note excluded from search", title not in found_titles, f"found={found_titles}")

    # Verify forgotten note excluded from list_notes (default)
    listed = await store.list_notes(include_forgotten=False)
    listed_titles = [n.title for n in listed]
    _check("forgotten note excluded from list", title not in listed_titles)

    # But included when include_forgotten=True
    all_listed = await store.list_notes(include_forgotten=True)
    all_titles = [n.title for n in all_listed]
    _check("forgotten note in full list", title in all_titles)


async def test_count_notes(store: SQLiteStore) -> None:
    """Verify active/forgotten counts."""
    print("\n--- count_notes ---")
    active, forgotten = await store.count_notes()
    _check("active count > 0", active > 0, f"active={active}")
    _check("forgotten count >= 1", forgotten >= 1, f"forgotten={forgotten}")
    total_notes = len(NOTES)
    _check("active + forgotten = total", active + forgotten == total_notes, f"{active}+{forgotten} vs {total_notes}")


async def test_category_counts(store: SQLiteStore) -> None:
    """Verify category aggregation."""
    print("\n--- category_counts ---")
    cats = await store.category_counts()
    _check("has categories", len(cats) > 0, f"cats={cats}")
    _check("engineering in categories", "engineering" in cats, f"keys={list(cats.keys())}")
    if "engineering" in cats:
        _check("engineering count >= 3", cats["engineering"] >= 3, f"count={cats['engineering']}")


async def test_tag_counts(store: SQLiteStore) -> None:
    """Verify tag aggregation."""
    print("\n--- tag_counts ---")
    tags = await store.tag_counts()
    _check("has tags", len(tags) > 0, f"len={len(tags)}")
    _check("'python' in tags", "python" in tags, f"keys={list(tags.keys())[:10]}")
    _check("'ai' appears in multiple notes", tags.get("ai", 0) >= 2, f"ai_count={tags.get('ai', 0)}")


async def test_recent_notes(store: SQLiteStore) -> None:
    """Verify recent notes returns freshly created notes."""
    print("\n--- recent_notes ---")
    recent = await store.recent_notes(days=1, limit=20)
    _check("recent notes non-empty", len(recent) > 0, f"got {len(recent)}")
    # All our notes were just created, so they should all be recent
    # (minus the forgotten one which has relevance=0)
    active, _ = await store.count_notes()
    _check("recent includes all active", len(recent) == active, f"recent={len(recent)}, active={active}")


async def test_stale_notes(store: SQLiteStore) -> None:
    """Verify stale notes (none should be stale since just created)."""
    print("\n--- stale_notes ---")
    stale = await store.stale_notes(days=30)
    _check("no stale notes (all fresh)", len(stale) == 0, f"got {len(stale)} stale")


async def test_embedding_cache(store: SQLiteStore) -> None:
    """Test embedding cache round-trip with struct-packed floats."""
    print("\n--- embedding_cache ---")
    provider_id = "test-provider"
    model = "test-model"
    text = "Python asyncio patterns"
    c_hash = content_hash(text)
    dims = 8
    embedding = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    # Cache miss
    cached = await store.get_cached_embedding(provider_id, model, c_hash)
    _check("cache miss returns None", cached is None)

    # Store embedding
    await store.cache_embedding(provider_id, model, c_hash, embedding, dims)

    # Cache hit
    cached = await store.get_cached_embedding(provider_id, model, c_hash)
    _check("cache hit returns list", cached is not None)
    if cached:
        _check("cached dims match", len(cached) == dims, f"len={len(cached)}")
        # Float comparison with tolerance (struct packing may lose precision)
        close = all(abs(a - b) < 1e-6 for a, b in zip(cached, embedding))
        _check("cached values match", close, f"cached={cached[:3]}...")

    # Different hash → miss
    other = await store.get_cached_embedding(provider_id, model, "other_hash")
    _check("different hash is miss", other is None)


async def test_vector_search(store: SQLiteStore, note_ids: dict[str, str]) -> None:
    """Test brute-force vector search via upsert_note_embedding + search_notes_vector."""
    print("\n--- vector search (brute-force) ---")
    dims = 8

    def _make_vec(words: list[str]) -> list[float]:
        """Deterministic vector: hash each word to a position, L2-normalize."""
        vec = [0.0] * dims
        for w in words:
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)
            idx = h % dims
            vec[idx] += 1.0
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    # Embed notes that still exist (skip forgotten grocery list)
    embeddings: dict[str, list[float]] = {}
    for title, content, tags, _cat in NOTES:
        if title not in note_ids:
            continue
        words = title.lower().split() + content.lower().split()
        embeddings[title] = _make_vec(words)
        await store.upsert_note_embedding(note_ids[title], embeddings[title])

    # Query for something similar to "docker volumes networks"
    query_vec = _make_vec(["docker", "volumes", "networks", "services", "health"])
    results = await store.search_notes_vector(query_vec, limit=5)
    _check("vector search returns results", len(results) > 0, f"got {len(results)}")
    if results:
        _check("top result score > 0", results[0].score > 0, f"score={results[0].score}")
        # The docker note shares the most query words, so it should rank high
        top_titles = [r.note.title for r in results[:3]]
        _check(
            "docker note in top 3",
            any("docker" in t.lower() for t in top_titles),
            f"top3={top_titles}",
        )

    # Query with a very different vector should return different ranking
    query_vec2 = _make_vec(["finance", "investment", "bonds", "funds"])
    results2 = await store.search_notes_vector(query_vec2, limit=5)
    _check("different query returns results", len(results2) > 0)
    if results2:
        top2 = [r.note.title for r in results2[:2]]
        _check("finance note ranks high", any("invest" in t.lower() for t in top2), f"top2={top2}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    print("=" * 60)
    print("E2E: SQLiteStore — CRUD, FTS, Embedding Cache, Vector Search")
    print("=" * 60)

    store = SQLiteStore(":memory:")
    await store.initialize()

    try:
        note_ids = await test_upsert_create(store)
        await test_upsert_update(store)
        await test_fts_search(store)
        await test_find_similar_titles(store)
        await test_set_relevance(store)
        await test_count_notes(store)
        await test_category_counts(store)
        await test_tag_counts(store)
        await test_recent_notes(store)
        await test_stale_notes(store)
        await test_embedding_cache(store)
        await test_vector_search(store, note_ids)
    finally:
        await store.close()

    print("\n" + "=" * 60)
    print(f"Results: {_passed} passed, {_failed} failed")
    print("=" * 60)
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    asyncio.run(main())

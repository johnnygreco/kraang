"""Unit tests for SQLiteStore — notes + sessions, upsert, search, relevance."""

from __future__ import annotations

import asyncio

import pytest

from kraang.config import normalize_title
from kraang.models import Session, utcnow
from kraang.search import build_fts_query
from kraang.store import SQLiteStore, _cosine_similarity, content_hash

# ---------------------------------------------------------------------------
# Cosine similarity edge cases
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_mismatched_dimensions(self):
        assert _cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0

    def test_zero_vectors(self):
        assert _cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0

    def test_opposite_vectors(self):
        assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# content_hash
# ---------------------------------------------------------------------------


class TestContentHash:
    def test_deterministic(self):
        assert content_hash("hello") == content_hash("hello")

    def test_different_inputs_different_hashes(self):
        assert content_hash("hello") != content_hash("world")

    def test_returns_hex_string(self):
        h = content_hash("test")
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex digest


# ---------------------------------------------------------------------------
# Note upsert
# ---------------------------------------------------------------------------


class TestUpsertNote:
    async def test_create_new(self, store):
        note, created = await store.upsert_note("Hello", "World")
        assert created is True
        assert note.title == "Hello"
        assert note.content == "World"
        assert note.note_id
        assert note.relevance == 1.0

    async def test_update_existing(self, store):
        note1, created1 = await store.upsert_note("Test", "Content 1")
        assert created1 is True

        note2, created2 = await store.upsert_note("Test", "Content 2")
        assert created2 is False
        assert note2.note_id == note1.note_id
        assert note2.content == "Content 2"

    async def test_case_insensitive_upsert(self, store):
        note1, _ = await store.upsert_note("My Title", "v1")
        note2, created = await store.upsert_note("my title", "v2")
        assert created is False
        assert note2.note_id == note1.note_id
        assert note2.content == "v2"

    async def test_whitespace_normalized(self, store):
        note1, _ = await store.upsert_note("  Extra   Spaces  ", "v1")
        note2, created = await store.upsert_note("Extra Spaces", "v2")
        assert created is False
        assert note2.note_id == note1.note_id

    async def test_with_tags_and_category(self, store):
        note, _ = await store.upsert_note("Tagged", "Content", tags=["a", "b"], category="cat1")
        assert note.tags == ["a", "b"]
        assert note.category == "cat1"

    async def test_upsert_restores_relevance(self, store):
        """remember() after forget() should restore relevance to 1.0."""
        await store.upsert_note("Forgotten", "Content")
        await store.set_relevance("Forgotten", 0.0)

        note = await store.get_note_by_title("Forgotten")
        assert note is not None
        assert note.relevance == 0.0

        note2, created = await store.upsert_note("Forgotten", "Updated content")
        assert created is False
        assert note2.relevance == 1.0

    async def test_title_normalized_stored(self, store):
        note, _ = await store.upsert_note("Pytest Config", "Content")
        assert note.title_normalized == normalize_title("Pytest Config")

    async def test_timestamps_set(self, store):
        note, _ = await store.upsert_note("Timestamp test", "Content")
        assert note.created_at is not None
        assert note.updated_at is not None
        assert note.created_at.tzinfo is not None


# ---------------------------------------------------------------------------
# Get by title / id
# ---------------------------------------------------------------------------


class TestGetNote:
    async def test_by_title(self, store):
        await store.upsert_note("Find me", "Body")
        found = await store.get_note_by_title("Find me")
        assert found is not None
        assert found.title == "Find me"

    async def test_by_title_case_insensitive(self, store):
        await store.upsert_note("CamelCase", "Body")
        found = await store.get_note_by_title("camelcase")
        assert found is not None

    async def test_by_id(self, store):
        note, _ = await store.upsert_note("ID test", "Body")
        found = await store.get_note(note.note_id)
        assert found is not None
        assert found.note_id == note.note_id

    async def test_not_found(self, store):
        assert await store.get_note_by_title("nonexistent") is None
        assert await store.get_note("nonexistent_id") is None


# ---------------------------------------------------------------------------
# Relevance / forget
# ---------------------------------------------------------------------------


class TestRelevance:
    async def test_set_relevance(self, store):
        await store.upsert_note("Forgettable", "Content")
        note = await store.set_relevance("Forgettable", 0.3)
        assert note is not None
        assert note.relevance == 0.3

    async def test_fully_forgotten(self, store):
        await store.upsert_note("Gone", "Content")
        note = await store.set_relevance("Gone", 0.0)
        assert note is not None
        assert note.relevance == 0.0

    async def test_set_relevance_not_found(self, store):
        result = await store.set_relevance("nonexistent", 0.5)
        assert result is None

    async def test_clamps_above_one(self, store):
        await store.upsert_note("Clamped high", "Content")
        note = await store.set_relevance("Clamped high", 5.0)
        assert note is not None
        assert note.relevance == 1.0

    async def test_clamps_below_zero(self, store):
        await store.upsert_note("Clamped low", "Content")
        note = await store.set_relevance("Clamped low", -1.0)
        assert note is not None
        assert note.relevance == 0.0


# ---------------------------------------------------------------------------
# List notes
# ---------------------------------------------------------------------------


class TestListNotes:
    async def test_empty(self, store):
        assert await store.list_notes() == []

    async def test_returns_all(self, populated_store):
        notes = await populated_store.list_notes()
        assert len(notes) == 15

    async def test_excludes_forgotten(self, store):
        await store.upsert_note("Active", "Content")
        await store.upsert_note("Forgotten", "Content")
        await store.set_relevance("Forgotten", 0.0)

        notes = await store.list_notes(include_forgotten=False)
        assert len(notes) == 1
        assert notes[0].title == "Active"

    async def test_includes_forgotten(self, store):
        await store.upsert_note("Active", "Content")
        await store.upsert_note("Forgotten", "Content")
        await store.set_relevance("Forgotten", 0.0)

        notes = await store.list_notes(include_forgotten=True)
        assert len(notes) == 2

    async def test_pagination(self, populated_store):
        page1 = await populated_store.list_notes(limit=5, offset=0)
        page2 = await populated_store.list_notes(limit=5, offset=5)
        assert len(page1) == 5
        assert len(page2) == 5
        ids1 = {n.note_id for n in page1}
        ids2 = {n.note_id for n in page2}
        assert ids1.isdisjoint(ids2)


# ---------------------------------------------------------------------------
# Search notes
# ---------------------------------------------------------------------------


class TestSearchNotes:
    async def test_basic_search(self, populated_store):
        fts = build_fts_query("asyncio")
        results = await populated_store.search_notes(fts)
        assert len(results) > 0
        titles = [r.note.title for r in results]
        assert any("asyncio" in t.lower() for t in titles)

    async def test_title_ranks_higher(self, populated_store):
        fts = build_fts_query("SQLite FTS5")
        results = await populated_store.search_notes(fts)
        assert len(results) > 0
        assert "SQLite FTS5" in results[0].note.title

    async def test_no_results(self, populated_store):
        fts = build_fts_query("xyznonexistentxyz")
        results = await populated_store.search_notes(fts)
        assert len(results) == 0

    async def test_excludes_forgotten(self, store):
        await store.upsert_note("Findable", "searchterm content")
        await store.upsert_note("Hidden", "searchterm content")
        await store.set_relevance("Hidden", 0.0)

        fts = build_fts_query("searchterm")
        results = await store.search_notes(fts)
        assert len(results) == 1
        assert results[0].note.title == "Findable"

    async def test_relevance_weighting(self, store):
        await store.upsert_note("Full weight", "searchterm content")
        await store.upsert_note("Half weight", "searchterm content")
        await store.set_relevance("Half weight", 0.5)

        fts = build_fts_query("searchterm")
        results = await store.search_notes(fts)
        assert len(results) == 2
        # Full weight should score higher
        assert results[0].note.title == "Full weight"

    async def test_score_positive(self, populated_store):
        fts = build_fts_query("python")
        results = await populated_store.search_notes(fts)
        for r in results:
            assert r.score > 0

    async def test_malformed_query_returns_empty(self, store):
        results = await store.search_notes('invalid "query')
        assert results == []


# ---------------------------------------------------------------------------
# Similar titles
# ---------------------------------------------------------------------------


class TestFindSimilarTitles:
    async def test_finds_similar(self, populated_store):
        similar = await populated_store.find_similar_titles("Python programming")
        assert len(similar) > 0

    async def test_empty_title(self, populated_store):
        similar = await populated_store.find_similar_titles("")
        assert similar == []


# ---------------------------------------------------------------------------
# Counts & analytics
# ---------------------------------------------------------------------------


class TestCounts:
    async def test_count_notes_empty(self, store):
        active, forgotten = await store.count_notes()
        assert active == 0
        assert forgotten == 0

    async def test_count_notes(self, store):
        await store.upsert_note("Active", "Content")
        await store.upsert_note("Forgotten", "Content")
        await store.set_relevance("Forgotten", 0.0)

        active, forgotten = await store.count_notes()
        assert active == 1
        assert forgotten == 1

    async def test_tag_counts(self, populated_store):
        tags = await populated_store.tag_counts()
        assert "python" in tags
        assert tags["python"] >= 2

    async def test_category_counts(self, populated_store):
        cats = await populated_store.category_counts()
        assert "engineering" in cats
        assert cats["engineering"] >= 5

    async def test_stale_notes_fresh(self, populated_store):
        stale = await populated_store.stale_notes(days=30)
        assert len(stale) == 0

    async def test_recent_notes(self, populated_store):
        recent = await populated_store.recent_notes(days=7)
        assert len(recent) > 0


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


class TestSessions:
    async def _make_session(self, **overrides) -> Session:
        defaults = {
            "session_id": "test-session-123",
            "slug": "test-slug",
            "project_path": "/test/project",
            "git_branch": "main",
            "model": "claude-opus-4-6",
            "started_at": utcnow(),
            "ended_at": utcnow(),
            "duration_s": 300,
            "user_turn_count": 5,
            "assistant_turn_count": 5,
            "summary": "Test session summary",
            "user_text": "User asked about testing",
            "assistant_text": "Agent explained testing",
            "tools_used": ["Read", "Edit"],
            "files_edited": ["/test/file.py"],
            "source_mtime": 1234567890.0,
            "source_size": 1024,
        }
        defaults.update(overrides)
        return Session(**defaults)

    async def test_upsert_and_get(self, store):
        session = await self._make_session()
        await store.upsert_session(session)

        found = await store.get_session("test-session-123")
        assert found is not None
        assert found.session_id == "test-session-123"
        assert found.slug == "test-slug"

    async def test_get_by_prefix(self, store):
        session = await self._make_session()
        await store.upsert_session(session)

        found = await store.get_session("test-ses")
        assert found is not None
        assert found.session_id == "test-session-123"

    async def test_get_not_found(self, store):
        assert await store.get_session("nonexistent") is None

    async def test_list_sessions(self, store):
        s1 = await self._make_session(session_id="s1")
        s2 = await self._make_session(session_id="s2")
        await store.upsert_session(s1)
        await store.upsert_session(s2)

        sessions = await store.list_sessions()
        assert len(sessions) == 2

    async def test_list_sessions_pagination(self, store):
        for i in range(5):
            s = await self._make_session(session_id=f"page-session-{i}")
            await store.upsert_session(s)

        page1 = await store.list_sessions(limit=2, offset=0)
        page2 = await store.list_sessions(limit=2, offset=2)
        assert len(page1) == 2
        assert len(page2) == 2
        ids1 = {s.session_id for s in page1}
        ids2 = {s.session_id for s in page2}
        assert ids1.isdisjoint(ids2)

    async def test_search_sessions(self, store):
        session = await self._make_session(
            user_text="How do I configure FTS5 search in SQLite?",
            summary="FTS5 configuration question",
        )
        await store.upsert_session(session)

        fts = build_fts_query("FTS5 search")
        results = await store.search_sessions(fts)
        assert len(results) > 0

    async def test_count_sessions(self, store):
        assert await store.count_sessions() == 0

        session = await self._make_session()
        await store.upsert_session(session)
        assert await store.count_sessions() == 1

    async def test_needs_reindex(self, store):
        assert await store.needs_reindex("new-id", 123.0, 456) is True

        session = await self._make_session(source_mtime=123.0, source_size=456)
        await store.upsert_session(session)
        assert await store.needs_reindex("test-session-123", 123.0, 456) is False
        assert await store.needs_reindex("test-session-123", 999.0, 456) is True

    async def test_upsert_replaces(self, store):
        s1 = await self._make_session(summary="Original")
        await store.upsert_session(s1)

        s2 = await self._make_session(summary="Updated")
        await store.upsert_session(s2)

        found = await store.get_session("test-session-123")
        assert found is not None
        assert found.summary == "Updated"
        assert await store.count_sessions() == 1

    async def test_ambiguous_prefix_raises(self, store):
        s1 = await self._make_session(session_id="aaaa-1111-0000-0000-000000000001")
        s2 = await self._make_session(session_id="aaaa-2222-0000-0000-000000000002")
        await store.upsert_session(s1)
        await store.upsert_session(s2)

        with pytest.raises(ValueError, match="Ambiguous"):
            await store.get_session("aaaa")

    async def test_last_indexed_at_empty(self, store):
        assert await store.last_indexed_at() is None

    async def test_last_indexed_at(self, store):
        session = await self._make_session()
        await store.upsert_session(session)
        result = await store.last_indexed_at()
        assert result is not None
        assert result.tzinfo is not None

    async def test_corrupted_session_json_fallback(self, store):
        """Corrupted tools_used_json/files_edited_json should fall back gracefully."""
        session = await self._make_session()
        await store.upsert_session(session)

        await store._conn.execute(
            "UPDATE sessions SET tools_used_json = 'not-json', "
            "files_edited_json = 'also-bad' WHERE session_id = ?",
            (session.session_id,),
        )
        await store._conn.commit()

        loaded = await store.get_session(session.session_id)
        assert loaded is not None
        assert loaded.tools_used == []
        assert loaded.files_edited == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    async def test_unicode_content(self, store):
        note, _ = await store.upsert_note(
            "Unicode test: caf\u00e9 r\u00e9sum\u00e9",
            "Content with \u65e5\u672c\u8a9e and \u00dc",
            tags=["\u00fcber", "caf\u00e9"],
        )
        found = await store.get_note(note.note_id)
        assert found is not None
        assert "caf\u00e9" in found.title

    async def test_long_content(self, store):
        long_text = "x" * 100_000
        note, _ = await store.upsert_note("Long note", long_text)
        found = await store.get_note(note.note_id)
        assert found is not None
        assert len(found.content) == 100_000

    async def test_special_chars_in_search(self, populated_store):
        fts = build_fts_query("test (with) [brackets]")
        if fts:
            results = await populated_store.search_notes(fts)
            assert isinstance(results, list)

    async def test_concurrent_creates(self, store):
        async def create_one(i: int):
            return await store.upsert_note(f"Concurrent {i}", f"Content {i}")

        results = await asyncio.gather(*[create_one(i) for i in range(10)])
        assert len(results) == 10
        ids = {note.note_id for note, _ in results}
        assert len(ids) == 10

    async def test_corrupted_tags_json_fallback(self, store):
        """Corrupted tags_json should fall back to empty list, not crash."""
        note, _ = await store.upsert_note("CorruptTags", "Content", tags=["a"])
        await store._conn.execute(
            "UPDATE notes SET tags_json = 'not-valid-json' WHERE note_id = ?",
            (note.note_id,),
        )
        await store._conn.commit()
        loaded = await store.get_note(note.note_id)
        assert loaded is not None
        assert loaded.tags == []  # Graceful fallback

    async def test_context_manager(self, tmp_db_path):
        async with SQLiteStore(str(tmp_db_path)) as s:
            note, _ = await s.upsert_note("Context manager test", "Content")
            assert note.note_id


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------


class TestEmbeddingCache:
    async def test_store_and_retrieve(self, store):
        embedding = [0.1, 0.2, 0.3, 0.4]
        await store.cache_embedding("openai", "text-embedding-3-small", "hash123", embedding, 4)

        cached = await store.get_cached_embedding("openai", "text-embedding-3-small", "hash123")
        assert cached is not None
        assert cached == pytest.approx(embedding, abs=1e-6)

    async def test_cache_miss(self, store):
        cached = await store.get_cached_embedding("openai", "model", "nonexistent")
        assert cached is None

    async def test_cache_different_provider(self, store):
        embedding = [0.5, 0.5]
        await store.cache_embedding("openai", "model-a", "hash1", embedding, 2)

        # Different provider should miss
        assert await store.get_cached_embedding("cohere", "model-a", "hash1") is None
        # Different model should miss
        assert await store.get_cached_embedding("openai", "model-b", "hash1") is None
        # Same combo should hit
        assert await store.get_cached_embedding("openai", "model-a", "hash1") is not None

    async def test_cache_replace(self, store):
        await store.cache_embedding("p", "m", "h", [1.0, 2.0], 2)
        await store.cache_embedding("p", "m", "h", [3.0, 4.0], 2)

        cached = await store.get_cached_embedding("p", "m", "h")
        assert cached is not None
        assert cached == pytest.approx([3.0, 4.0], abs=1e-6)

    async def test_corrupted_embedding_returns_none(self, store):
        """A corrupted BLOB in the cache should return None, not crash."""
        await store._conn.execute(
            "INSERT INTO embedding_cache"
            " (provider, model, content_hash, embedding, dims, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                "test",
                "test-model",
                "corrupt_hash",
                b"not-valid-float-data",
                3,
                "2024-01-01T00:00:00+00:00",
            ),
        )
        await store._conn.commit()
        result = await store.get_cached_embedding("test", "test-model", "corrupt_hash")
        assert result is None  # Should not crash


# ---------------------------------------------------------------------------
# Vector search (brute-force fallback)
# ---------------------------------------------------------------------------


class TestVectorSearch:
    async def test_search_empty(self, store):
        store._vec_loaded = False
        results = await store.search_notes_vector([0.1, 0.2, 0.3], limit=5)
        assert results == []

    async def test_upsert_and_search_bruteforce(self, store):
        """Test upsert_note_embedding + search_notes_vector with brute-force fallback."""
        store._vec_loaded = False
        # Create a note first
        note, _ = await store.upsert_note("Vector test", "Content about vectors")

        # Store an embedding for it (uses fallback path since sqlite-vec not loaded)
        embedding = [0.6, 0.8, 0.0]
        await store.upsert_note_embedding(note.note_id, embedding)

        # Search with a similar vector
        query = [0.6, 0.8, 0.0]
        results = await store.search_notes_vector(query, limit=5)
        assert len(results) > 0
        assert results[0].note.note_id == note.note_id
        assert results[0].score > 0.0

    async def test_bruteforce_skips_forgotten_notes(self, store):
        """Brute-force search should skip forgotten notes and fill the limit."""
        store._vec_loaded = False
        # Create 5 notes with embeddings
        notes = []
        for i in range(5):
            n, _ = await store.upsert_note(f"BF Note {i}", f"Content {i}")
            # Assign embeddings that spread across a dimension so we control ranking
            emb = [0.0] * 5
            emb[i] = 1.0
            await store.upsert_note_embedding(n.note_id, emb)
            notes.append(n)

        # Forget the top-2 ranked notes (closest to query)
        # Query will be [1,1,1,1,1] so all are equally similar; forget first two
        await store.set_relevance(notes[0].title, 0.0)
        await store.set_relevance(notes[1].title, 0.0)

        # Request limit=3 — should get 3 results despite 2 forgotten notes
        query = [1.0, 1.0, 1.0, 1.0, 1.0]
        results = await store.search_notes_vector(query, limit=3)
        assert len(results) == 3
        result_ids = {r.note.note_id for r in results}
        # The forgotten notes should NOT appear
        assert notes[0].note_id not in result_ids
        assert notes[1].note_id not in result_ids

    async def test_bruteforce_multiple_results(self, store):
        store._vec_loaded = False
        n1, _ = await store.upsert_note("Vec A", "Content A")
        n2, _ = await store.upsert_note("Vec B", "Content B")

        await store.upsert_note_embedding(n1.note_id, [1.0, 0.0, 0.0])
        await store.upsert_note_embedding(n2.note_id, [0.0, 1.0, 0.0])

        # Query closer to n1
        results = await store.search_notes_vector([0.9, 0.1, 0.0], limit=5)
        assert len(results) == 2
        assert results[0].note.note_id == n1.note_id


# ---------------------------------------------------------------------------
# Note embedding upsert
# ---------------------------------------------------------------------------


class TestNoteEmbedding:
    async def test_upsert_creates_entry(self, store):
        await store.ensure_vec_table(3)
        note, _ = await store.upsert_note("Embed test", "Content")
        await store.upsert_note_embedding(note.note_id, [0.1, 0.2, 0.3])

        # Verify it's searchable
        results = await store.search_notes_vector([0.1, 0.2, 0.3], limit=1)
        assert len(results) == 1

    async def test_upsert_replaces_embedding(self, store):
        await store.ensure_vec_table(3)
        note, _ = await store.upsert_note("Replace embed", "Content")
        await store.upsert_note_embedding(note.note_id, [1.0, 0.0, 0.0])
        await store.upsert_note_embedding(note.note_id, [0.0, 1.0, 0.0])

        # Search with new embedding direction
        results = await store.search_notes_vector([0.0, 1.0, 0.0], limit=1)
        assert len(results) == 1
        assert results[0].note.note_id == note.note_id

    async def test_ensure_vec_table_no_op_without_vec(self, store):
        """ensure_vec_table should be a no-op when sqlite-vec is unavailable."""
        store._vec_loaded = False
        await store.ensure_vec_table(1536)  # Should not raise

    async def test_ensure_vec_table_rebuilds_on_dim_mismatch(self, store):
        """ensure_vec_table should drop and recreate when dimensions change."""
        assert store._vec_loaded is True, "sqlite-vec must be available for this test"

        # Create with 1536 dims
        await store.ensure_vec_table(1536)
        note, _ = await store.upsert_note("Dim test", "Content")
        await store.upsert_note_embedding(note.note_id, [0.1] * 1536)

        # Verify initial table works
        results = await store.search_notes_vector([0.1] * 1536, limit=1)
        assert len(results) == 1

        # Rebuild with 768 dims — should drop and recreate
        await store.ensure_vec_table(768)

        # Old embedding should be gone; new table has different dims
        results = await store.search_notes_vector([0.1] * 768, limit=1)
        assert len(results) == 0

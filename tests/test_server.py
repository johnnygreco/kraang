"""Integration tests for the kraang MCP server tools."""

from __future__ import annotations

import pytest

import kraang.server as server

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_store(store, monkeypatch):
    """Point the server module's store singleton at the test store and reset provider state."""

    async def _get_test_store():
        return store

    monkeypatch.setattr(server, "_get_store", _get_test_store)

    # Reset provider globals so each test starts clean (FTS-only by default).
    monkeypatch.setattr(server, "_provider_init_attempted", False)
    monkeypatch.setattr(server, "_provider", None)

    async def _no_provider():
        return None

    monkeypatch.setattr(server, "_get_provider", _no_provider)


# ---------------------------------------------------------------------------
# remember
# ---------------------------------------------------------------------------


class TestRemember:
    async def test_creates_note(self, store):
        result = await server.remember("My Title", "My content")
        assert 'Created "My Title"' in result

    async def test_updates_existing(self, store):
        await server.remember("Title", "v1")
        result = await server.remember("Title", "v2")
        assert 'Updated "Title"' in result

    async def test_with_tags_and_category(self, store):
        result = await server.remember("Tagged", "Content", tags=["a", "b"], category="cat1")
        assert 'Created "Tagged"' in result
        assert "a | b" in result

    async def test_note_persists(self, store):
        await server.remember("Persist", "Persisted content", tags=["t1"])
        notes = await store.list_notes()
        assert len(notes) == 1
        assert notes[0].title == "Persist"

    async def test_error_handling(self, store, monkeypatch):
        async def _broken(*a, **kw):
            raise RuntimeError("db error")

        monkeypatch.setattr(store, "upsert_note", _broken)
        result = await server.remember("Title", "Content")
        assert "Error:" in result


class TestRememberValidation:
    async def test_empty_title(self, store):
        result = await server.remember("", "content")
        assert "Error:" in result
        assert "title" in result.lower()

    async def test_whitespace_title(self, store):
        result = await server.remember("   ", "content")
        assert "Error:" in result
        assert "title" in result.lower()

    async def test_empty_content(self, store):
        result = await server.remember("Title", "")
        assert "Error:" in result
        assert "content" in result.lower()

    async def test_whitespace_content(self, store):
        result = await server.remember("Title", "   ")
        assert "Error:" in result
        assert "content" in result.lower()


# ---------------------------------------------------------------------------
# recall
# ---------------------------------------------------------------------------


class TestRecall:
    async def test_returns_results(self, populated_store):
        result = await server.recall("asyncio")
        assert "Results for" in result
        assert "asyncio" in result.lower()

    async def test_no_results(self, populated_store):
        result = await server.recall("zzzznonexistentzzzz")
        assert "No results found" in result

    async def test_scope_notes(self, populated_store):
        result = await server.recall("python", scope="notes")
        assert "Results for" in result or "No results" in result

    async def test_scope_sessions(self, populated_store):
        result = await server.recall("python", scope="sessions")
        # No sessions indexed in test, so should be empty
        assert "No results" in result or "Results for" in result

    async def test_error_handling(self, store, monkeypatch):
        async def _broken(*a, **kw):
            raise RuntimeError("db error")

        monkeypatch.setattr(store, "search_notes", _broken)
        result = await server.recall("query")
        assert "Error:" in result


# ---------------------------------------------------------------------------
# forget
# ---------------------------------------------------------------------------


class TestForget:
    async def test_forget_note(self, store):
        await server.remember("Forgettable", "Content")
        result = await server.forget("Forgettable")
        assert 'Forgot "Forgettable"' in result
        assert "0.0" in result

    async def test_forget_partial(self, store):
        await server.remember("Partial", "Content")
        result = await server.forget("Partial", relevance=0.3)
        assert "0.3" in result

    async def test_forget_not_found(self, store):
        result = await server.forget("nonexistent")
        assert "not found" in result

    async def test_remember_restores(self, store):
        await server.remember("Restore me", "v1")
        await server.forget("Restore me")

        # Note is hidden
        notes = await store.list_notes()
        assert len(notes) == 0

        # remember restores it
        await server.remember("Restore me", "v2")
        notes = await store.list_notes()
        assert len(notes) == 1
        assert notes[0].relevance == 1.0

    async def test_forget_out_of_range(self, store):
        await server.remember("Range test", "Content")
        result = await server.forget("Range test", relevance=5.0)
        assert "between 0.0 and 1.0" in result

    async def test_error_handling(self, store, monkeypatch):
        async def _broken(*a, **kw):
            raise RuntimeError("db error")

        monkeypatch.setattr(store, "set_relevance", _broken)
        result = await server.forget("title")
        assert "Error:" in result


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


class TestStatus:
    async def test_empty_store(self, store):
        result = await server.status()
        assert "Kraang Status" in result
        assert "Notes:" in result

    async def test_populated(self, populated_store):
        result = await server.status()
        assert "Kraang Status" in result
        assert "Categories" in result
        assert "Tags" in result

    async def test_error_handling(self, store, monkeypatch):
        async def _broken(*a, **kw):
            raise RuntimeError("db error")

        monkeypatch.setattr(store, "count_notes", _broken)
        result = await server.status()
        assert "Error:" in result


# ---------------------------------------------------------------------------
# read_session
# ---------------------------------------------------------------------------


class TestReadSession:
    async def test_not_found(self, store):
        result = await server.read_session("nonexistent-session-id")
        assert "not found" in result.lower()

    async def test_missing_transcript_file(self, store):
        """Session exists in DB but JSONL file is missing."""
        from kraang.models import Session, utcnow

        session = Session(
            session_id="test-read-session-123",
            slug="test-slug",
            project_path="/nonexistent/project/path",
            git_branch="main",
            model="claude-opus-4-6",
            started_at=utcnow(),
            ended_at=utcnow(),
            duration_s=60,
            user_turn_count=2,
            assistant_turn_count=2,
            summary="Test session",
            user_text="user text",
            assistant_text="assistant text",
            tools_used=["Read"],
            files_edited=[],
            source_mtime=123.0,
            source_size=456,
        )
        await store.upsert_session(session)

        result = await server.read_session("test-read-session-123")
        assert "not found" in result.lower()


# ---------------------------------------------------------------------------
# context
# ---------------------------------------------------------------------------


class TestContext:
    async def test_context_returns_xml(self, populated_store, monkeypatch):
        """context tool should return safety-framed XML."""
        monkeypatch.setattr(server, "_provider_init_attempted", False)
        monkeypatch.setattr(server, "_provider", None)

        async def _no_provider():
            return None

        monkeypatch.setattr(server, "_get_provider", _no_provider)

        result = await server.context("asyncio")
        assert "<relevant-memories>" in result
        assert "</relevant-memories>" in result

    async def test_context_includes_safety_warning(self, populated_store, monkeypatch):
        monkeypatch.setattr(server, "_provider_init_attempted", False)
        monkeypatch.setattr(server, "_provider", None)

        async def _no_provider():
            return None

        monkeypatch.setattr(server, "_get_provider", _no_provider)

        result = await server.context("python")
        assert "untrusted" in result.lower()

    async def test_context_no_results(self, store, monkeypatch):
        monkeypatch.setattr(server, "_provider_init_attempted", False)
        monkeypatch.setattr(server, "_provider", None)

        async def _no_provider():
            return None

        monkeypatch.setattr(server, "_get_provider", _no_provider)

        result = await server.context("xyznonexistentxyz")
        assert "No relevant memories" in result

    async def test_context_error_handling(self, store, monkeypatch):
        async def _broken():
            raise RuntimeError("db error")

        monkeypatch.setattr(server, "_get_store", _broken)
        result = await server.context("query")
        assert "Error" in result


# ---------------------------------------------------------------------------
# recall — graceful degradation without embeddings
# ---------------------------------------------------------------------------


class TestRecallWithoutEmbeddings:
    async def test_recall_fts_only(self, populated_store, monkeypatch):
        """recall should work without embeddings using FTS-only fallback."""
        monkeypatch.setattr(server, "_provider_init_attempted", False)
        monkeypatch.setattr(server, "_provider", None)

        async def _no_provider():
            return None

        monkeypatch.setattr(server, "_get_provider", _no_provider)

        result = await server.recall("python")
        assert "Results for" in result

    async def test_recall_notes_scope_fts_only(self, populated_store, monkeypatch):
        monkeypatch.setattr(server, "_provider_init_attempted", False)
        monkeypatch.setattr(server, "_provider", None)

        async def _no_provider():
            return None

        monkeypatch.setattr(server, "_get_provider", _no_provider)

        result = await server.recall("asyncio", scope="notes")
        assert "Results for" in result or "No results" in result


# ---------------------------------------------------------------------------
# remember — graceful handling of embedding failures
# ---------------------------------------------------------------------------


class TestRememberEmbeddingFailure:
    async def test_remember_succeeds_if_embedding_fails(self, store, monkeypatch):
        """remember should still save the note even if embedding fails."""
        monkeypatch.setattr(server, "_provider_init_attempted", False)
        monkeypatch.setattr(server, "_provider", None)

        class BrokenProvider:
            provider_id = "broken"
            model = "broken-v1"
            dims = 3

            async def embed_query(self, text):
                raise RuntimeError("API down")

        async def _broken_provider():
            return BrokenProvider()

        monkeypatch.setattr(server, "_get_provider", _broken_provider)

        result = await server.remember("Embed fail test", "Still saves")
        assert 'Created "Embed fail test"' in result

        # Verify note was saved
        note = await store.get_note_by_title("Embed fail test")
        assert note is not None

    async def test_remember_succeeds_without_provider(self, store, monkeypatch):
        """remember should work fine when no embedding provider is available."""
        monkeypatch.setattr(server, "_provider_init_attempted", False)
        monkeypatch.setattr(server, "_provider", None)

        async def _no_provider():
            return None

        monkeypatch.setattr(server, "_get_provider", _no_provider)

        result = await server.remember("No embed", "Content here")
        assert 'Created "No embed"' in result


# ---------------------------------------------------------------------------
# status — with embedding status
# ---------------------------------------------------------------------------


class TestStatusEmbeddings:
    async def test_status_shows_embedding_disabled(self, store, monkeypatch):
        monkeypatch.setattr(server, "_provider_init_attempted", False)
        monkeypatch.setattr(server, "_provider", None)

        async def _no_provider():
            return None

        monkeypatch.setattr(server, "_get_provider", _no_provider)

        result = await server.status()
        assert "Kraang Status" in result
        assert "disabled" in result.lower() or "Embeddings" in result

    async def test_status_disabled_no_api_key(self, store, monkeypatch):
        """Status should show 'set OPENAI_API_KEY' message when no key configured."""
        async def _no_provider():
            return None

        monkeypatch.setattr(server, "_get_provider", _no_provider)

        result = await server.status()
        assert "OPENAI_API_KEY" in result


# ---------------------------------------------------------------------------
# context — max_results parameter
# ---------------------------------------------------------------------------


class TestContextMaxResults:
    async def test_context_respects_max_results(self, populated_store, monkeypatch):
        """context tool should respect max_results parameter."""
        monkeypatch.setattr(server, "_provider_init_attempted", False)
        monkeypatch.setattr(server, "_provider", None)

        async def _no_provider():
            return None

        monkeypatch.setattr(server, "_get_provider", _no_provider)

        result = await server.context("python", max_results=2)
        assert "<relevant-memories>" in result
        assert "</relevant-memories>" in result
        # Count the numbered entries in the result
        lines = result.strip().split("\n")
        numbered_lines = [line for line in lines if line and line[0].isdigit() and ". " in line]
        assert len(numbered_lines) <= 2

    async def test_context_error_includes_exception_type(self, store, monkeypatch):
        """context error message should include exception type."""
        async def _broken():
            raise RuntimeError("db error")

        monkeypatch.setattr(server, "_get_store", _broken)
        result = await server.context("query")
        assert "RuntimeError" in result


# ---------------------------------------------------------------------------
# remember — embedding feedback
# ---------------------------------------------------------------------------


class TestRememberEmbeddingFeedback:
    async def test_remember_shows_fts_only_when_no_provider(self, store, monkeypatch):
        """remember output should show [FTS only] when no embedding provider."""
        monkeypatch.setattr(server, "_provider_init_attempted", False)
        monkeypatch.setattr(server, "_provider", None)

        async def _no_provider():
            return None

        monkeypatch.setattr(server, "_get_provider", _no_provider)

        result = await server.remember("FTS Test", "Some content")
        assert "[FTS only]" in result

    async def test_remember_shows_embedded_when_provider_succeeds(self, store, monkeypatch):
        """remember output should show [embedded] when embedding succeeds."""
        await store.ensure_vec_table(3)

        class FakeProvider:
            provider_id = "fake"
            model = "fake-v1"
            dims = 3

            async def embed_query(self, text):
                return [0.1, 0.2, 0.3]

        async def _fake_provider():
            return FakeProvider()

        monkeypatch.setattr(server, "_get_provider", _fake_provider)

        result = await server.remember("Embed Test", "Some content")
        assert "[embedded]" in result

    async def test_update_shows_fts_only(self, store, monkeypatch):
        """Updated note output should also show [FTS only] indicator."""
        async def _no_provider():
            return None

        monkeypatch.setattr(server, "_get_provider", _no_provider)

        await server.remember("Update Test", "v1")
        result = await server.remember("Update Test", "v2")
        assert 'Updated "Update Test"' in result
        assert "[FTS only]" in result

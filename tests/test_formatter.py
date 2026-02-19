"""Tests for kraang.formatter — markdown output formatting."""

from __future__ import annotations

from datetime import datetime, timezone

from kraang.formatter import (
    format_forget,
    format_recall_results,
    format_remember_created,
    format_remember_updated,
    format_status,
    format_transcript,
)
from kraang.models import (
    Note,
    NoteSearchResult,
    Session,
    SessionSearchResult,
    TranscriptTurn,
)


def _make_note(**overrides) -> Note:
    defaults = {
        "note_id": "abc123",
        "title": "Test Note",
        "title_normalized": "test note",
        "content": "Test content here.",
        "tags": ["python", "testing"],
        "category": "engineering",
        "relevance": 1.0,
        "created_at": datetime(2026, 2, 10, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 2, 10, tzinfo=timezone.utc),
    }
    defaults.update(overrides)
    return Note(**defaults)


def _make_session(**overrides) -> Session:
    defaults = {
        "session_id": "test-session-abc",
        "slug": "lazy-gathering-dove",
        "project_path": "/test/project",
        "git_branch": "feature/search",
        "model": "claude-opus-4-6",
        "started_at": datetime(2026, 2, 9, 14, 30, tzinfo=timezone.utc),
        "ended_at": datetime(2026, 2, 9, 14, 42, tzinfo=timezone.utc),
        "duration_s": 720,
        "user_turn_count": 4,
        "assistant_turn_count": 4,
        "summary": "User asked about FTS5 search configuration",
        "user_text": "How do I configure FTS5?",
        "assistant_text": "Here is how to configure FTS5.",
        "tools_used": ["Read", "Edit"],
        "files_edited": ["/test/search.py"],
        "source_mtime": 123.0,
        "source_size": 1024,
    }
    defaults.update(overrides)
    return Session(**defaults)


# ---------------------------------------------------------------------------
# remember formatting
# ---------------------------------------------------------------------------


class TestFormatRemember:
    def test_created(self):
        note = _make_note()
        result = format_remember_created(note)
        assert 'Created "Test Note"' in result
        assert "python | testing" in result
        assert "engineering" in result

    def test_created_with_similar(self):
        note = _make_note(title="API design v1")
        similar = [_make_note(title="v1 API design decisions", note_id="xyz")]
        result = format_remember_created(note, similar=similar)
        assert 'Created "API design v1"' in result
        assert "similar existing note" in result
        assert "v1 API design decisions" in result

    def test_updated(self):
        note = _make_note(title="Updated Note")
        result = format_remember_updated(note)
        assert 'Updated "Updated Note"' in result

    def test_no_tags(self):
        note = _make_note(tags=[])
        result = format_remember_created(note)
        assert 'Created "Test Note"' in result


# ---------------------------------------------------------------------------
# recall formatting
# ---------------------------------------------------------------------------


class TestFormatRecall:
    def test_with_notes_and_sessions(self):
        note = _make_note()
        session = _make_session()
        notes = [NoteSearchResult(note=note, score=5.0, snippet="FTS5 search...")]
        sessions = [SessionSearchResult(session=session, score=3.0, snippet="FTS5 config...")]

        result = format_recall_results("FTS5", notes, sessions)
        assert '## Results for "FTS5"' in result
        assert "### Notes (1 match)" in result
        assert "### Sessions (1 match)" in result
        assert "Test Note" in result
        assert "lazy-gathering-dove" in result

    def test_notes_only(self):
        note = _make_note()
        notes = [NoteSearchResult(note=note, score=5.0)]
        result = format_recall_results("test", notes, [])
        assert "### Notes" in result
        assert "### Sessions" not in result

    def test_no_results(self):
        result = format_recall_results("nothing", [], [])
        assert "No results found" in result


# ---------------------------------------------------------------------------
# transcript formatting
# ---------------------------------------------------------------------------


class TestFormatTranscript:
    def test_basic_transcript(self):
        session = _make_session()
        turns = [
            TranscriptTurn(
                role="User",
                timestamp="2026-02-09T14:30:00Z",
                text="How do I configure FTS5?",
            ),
            TranscriptTurn(
                role="Agent",
                timestamp="2026-02-09T14:31:00Z",
                text="Let me investigate.",
                tool_calls=["Read /test/search.py"],
            ),
        ]
        result = format_transcript(session, turns)
        assert "## Session: lazy-gathering-dove" in result
        assert "**User**" in result
        assert "**Agent**" in result
        assert "Read /test/search.py" in result

    def test_max_turns(self):
        session = _make_session()
        turns = [TranscriptTurn(role="User", text=f"Turn {i}") for i in range(10)]
        result = format_transcript(session, turns, max_turns=3)
        assert "Turn 0" in result
        assert "Turn 2" in result
        assert "Turn 5" not in result


# ---------------------------------------------------------------------------
# forget formatting
# ---------------------------------------------------------------------------


class TestFormatForget:
    def test_fully_forgotten(self):
        result = format_forget("Old note", 0.0)
        assert 'Forgot "Old note"' in result
        assert "0.0" in result
        assert "hidden from search" in result

    def test_partial_forget(self):
        result = format_forget("Some note", 0.3)
        assert 'Forgot "Some note"' in result
        assert "0.3" in result


# ---------------------------------------------------------------------------
# status formatting
# ---------------------------------------------------------------------------


class TestFormatStatus:
    def test_basic_status(self):
        result = format_status(
            active_notes=10,
            forgotten_notes=2,
            session_count=5,
            last_indexed=datetime(2026, 2, 14, tzinfo=timezone.utc),
            recent_notes=[_make_note()],
            categories={"engineering": 5, "personal": 3},
            tags={"python": 4, "testing": 2},
            stale_notes=[],
        )
        assert "## Kraang Status" in result
        assert "10" in result
        assert "2 forgotten" in result
        assert "engineering" in result
        assert "python" in result

    def test_empty_status(self):
        result = format_status(
            active_notes=0,
            forgotten_notes=0,
            session_count=0,
            last_indexed=None,
            recent_notes=[],
            categories={},
            tags={},
            stale_notes=[],
        )
        assert "## Kraang Status" in result
        assert "0 total" in result

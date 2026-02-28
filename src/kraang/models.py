"""Pydantic models for kraang — the shared data contracts."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_id() -> str:
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SearchScope(StrEnum):
    """What to search in a recall query."""

    ALL = "all"
    NOTES = "notes"
    SESSIONS = "sessions"


# ---------------------------------------------------------------------------
# Note model
# ---------------------------------------------------------------------------


class Note(BaseModel):
    """A single knowledge note stored in the brain."""

    note_id: str = Field(default_factory=new_id)
    title: str
    title_normalized: str = ""
    content: str
    tags: list[str] = Field(default_factory=list)
    category: str = ""
    relevance: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


# ---------------------------------------------------------------------------
# Session model
# ---------------------------------------------------------------------------


class Session(BaseModel):
    """An indexed conversation session from Claude Code."""

    session_id: str
    slug: str = ""
    project_path: str
    git_branch: str = ""
    model: str = ""
    started_at: datetime
    ended_at: datetime
    duration_s: int = 0
    user_turn_count: int = 0
    assistant_turn_count: int = 0
    summary: str = ""
    user_text: str = ""
    assistant_text: str = ""
    tools_used: list[str] = Field(default_factory=list)
    files_edited: list[str] = Field(default_factory=list)
    source_mtime: float = 0.0
    source_size: int = 0
    indexed_at: datetime = Field(default_factory=utcnow)


# ---------------------------------------------------------------------------
# Search results
# ---------------------------------------------------------------------------


class NoteSearchResult(BaseModel):
    """A note search hit with relevance score."""

    note: Note
    score: float
    snippet: str = ""


class SessionSearchResult(BaseModel):
    """A session search hit with relevance score."""

    session: Session
    score: float
    snippet: str = ""


# ---------------------------------------------------------------------------
# Transcript models (for read_session)
# ---------------------------------------------------------------------------


class TranscriptTurn(BaseModel):
    """A single turn in a conversation transcript."""

    role: str  # "User" or "Agent"
    timestamp: str = ""
    text: str = ""
    tool_calls: list[str] = Field(default_factory=list)

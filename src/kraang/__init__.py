"""Kraang — a second brain for humans and their agents."""

from __future__ import annotations

try:
    from kraang._version import __version__
except ImportError:
    __version__ = "0.0.0"

from kraang.hybrid import HybridConfig
from kraang.models import (
    Note,
    NoteSearchResult,
    SearchScope,
    Session,
    SessionSearchResult,
    TranscriptTurn,
)
from kraang.store import SQLiteStore

__all__ = [
    "HybridConfig",
    "Note",
    "NoteSearchResult",
    "SQLiteStore",
    "SearchScope",
    "Session",
    "SessionSearchResult",
    "TranscriptTurn",
]

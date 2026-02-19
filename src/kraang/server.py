"""MCP server for kraang — 6 tools: remember, recall, context, read_session, forget, status."""

from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from mcp.server.fastmcp import FastMCP

from kraang.config import resolve_db_path
from kraang.formatter import (
    format_forget,
    format_recall_results,
    format_remember_created,
    format_remember_updated,
    format_status,
    format_transcript,
)
from kraang.search import build_fts_query

try:
    from kraang.embeddings import create_provider
except ImportError:
    create_provider = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from kraang.embeddings import EmbeddingProvider
    from kraang.store import SQLiteStore

logger = logging.getLogger("kraang.server")
logging.basicConfig(stream=sys.stderr, level=logging.INFO)

mcp = FastMCP(
    "kraang",
    instructions=(
        "A second brain for humans and their agents — "
        "project-scoped knowledge management with session indexing, full-text search, "
        "and optional semantic search. "
        "Tools: remember (save knowledge), recall (search), context (auto-recall for prompt injection), "
        "read_session (view transcript), forget (downweight), status (overview)."
    ),
)

# ---------------------------------------------------------------------------
# Store singleton — initialised lazily on first tool call
# ---------------------------------------------------------------------------

_store: SQLiteStore | None = None
_provider: EmbeddingProvider | None = None
_provider_checked: bool = False


async def _get_store() -> SQLiteStore:
    """Return the initialised SQLiteStore singleton."""
    global _store
    if _store is None:
        from kraang.store import SQLiteStore

        db_path = resolve_db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _store = SQLiteStore(str(db_path))
        await _store.initialize()
    return _store


async def _get_provider() -> EmbeddingProvider | None:
    """Return the embedding provider singleton, or ``None`` if unavailable."""
    global _provider, _provider_checked
    if _provider_checked:
        return _provider
    _provider_checked = True
    if create_provider is None:
        logger.debug("Embeddings extra not installed — semantic search disabled")
        return None
    try:
        _provider = await create_provider()
        if _provider is not None:
            store = await _get_store()
            await store.ensure_vec_table(_provider.dims)
            logger.info(
                "Embedding provider ready: %s/%s (%d dims)",
                _provider.provider_id,
                _provider.model,
                _provider.dims,
            )
    except Exception:
        logger.warning("Failed to initialise embedding provider", exc_info=True)
        _provider = None
    return _provider


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def remember(
    title: str,
    content: str,
    tags: list[str] | None = None,
    category: str = "",
) -> str:
    """Save knowledge to the brain. If a note with this title exists, update it. Otherwise create.

    Args:
        title: Title of the note. Use the exact same title to update an existing note.
        content: Full content of the note.
        tags: Optional list of tags for categorisation.
        category: Optional category label.
    """
    try:
        # Validate inputs
        title = title.strip()
        content = content.strip()
        if not title:
            return "Error: title must not be empty."
        if not content:
            return "Error: content must not be empty."

        store = await _get_store()
        note, created = await store.upsert_note(
            title=title,
            content=content,
            tags=tags,
            category=category,
        )

        # Try to embed and store the vector (non-blocking, never fails the tool)
        try:
            provider = await _get_provider()
            if provider is not None:
                text = f"{note.title}\n{note.content}"
                content_key = hashlib.sha256(text.encode()).hexdigest()
                cached = await store.get_cached_embedding(
                    provider.provider_id, provider.model, content_key
                )
                if cached is not None:
                    embedding = cached
                else:
                    embedding = await provider.embed_query(text)
                    await store.cache_embedding(
                        provider.provider_id,
                        provider.model,
                        content_key,
                        embedding,
                        provider.dims,
                    )
                await store.upsert_note_embedding(note.note_id, embedding)
        except Exception:
            logger.warning("Embedding failed for note '%s' — skipping", title, exc_info=True)

        if created:
            # Check for similar existing notes (fuzzy duplicate detection)
            similar = await store.find_similar_titles(title, limit=3)
            # Filter out the just-created note itself
            similar = [s for s in similar if s.note_id != note.note_id]
            return format_remember_created(note, similar if similar else None)
        else:
            return format_remember_updated(note)
    except Exception:
        logger.exception("remember failed")
        return f'Error: could not save "{title}".'


@mcp.tool()
async def recall(
    query: str,
    scope: Literal["all", "notes", "sessions"] = "all",
    limit: int = 10,
) -> str:
    """Search notes and conversation sessions. Returns markdown-formatted results.

    Args:
        query: Natural language search query.
        scope: What to search — "all" (default), "notes", or "sessions".
        limit: Maximum number of results per type (default 10).
    """
    try:
        store = await _get_store()
        provider = await _get_provider()

        from kraang.models import NoteSearchResult, SessionSearchResult

        notes: list[NoteSearchResult] = []
        sessions: list[SessionSearchResult] = []

        if scope in ("all", "notes"):
            from kraang.hybrid import hybrid_search

            notes = await hybrid_search(store, provider, query, limit=limit)

        if scope in ("all", "sessions"):
            fts_expr = build_fts_query(query)
            if fts_expr:
                sessions = await store.search_sessions(fts_expr, limit=limit)

        if not notes and not sessions:
            return f'No results found for "{query}".'

        return format_recall_results(query, notes, sessions)
    except Exception:
        logger.exception("recall failed")
        return f'Error: search for "{query}" failed.'


@mcp.tool()
async def context(
    query: str,
    max_results: int = 5,
) -> str:
    """Auto-recall relevant memories for context injection. Returns safety-framed XML.

    Use this to inject relevant prior knowledge into your context before answering.
    The output wraps recalled memories in a safety frame to prevent prompt injection.

    Args:
        query: Natural language query describing what context you need.
        max_results: Maximum memories to return (default 5).
    """
    try:
        from kraang.safety import format_recalled_context

        store = await _get_store()
        provider = await _get_provider()

        from kraang.hybrid import hybrid_search

        results = await hybrid_search(store, provider, query, limit=max_results)
        return format_recalled_context(results)
    except Exception:
        logger.exception("context failed")
        return "<relevant-memories>\nError retrieving context.\n</relevant-memories>"


@mcp.tool()
async def read_session(
    session_id: str,
    max_turns: int = 0,
) -> str:
    """Load a full conversation transcript by session ID.

    Use `recall` to find sessions first, then use the session ID to read the full transcript.

    Args:
        session_id: Full UUID or 8-char prefix of the session.
        max_turns: Maximum turns to include (0 = all).
    """
    try:
        store = await _get_store()
        try:
            session = await store.get_session(session_id)
        except ValueError as e:
            return str(e)

        if session is None:
            return f'Session "{session_id}" not found.'

        # Find the JSONL file
        from kraang.config import encode_project_path

        encoded = encode_project_path(session.project_path)
        sessions_dir = Path.home() / ".claude" / "projects" / encoded
        jsonl_path = sessions_dir / f"{session.session_id}.jsonl"

        if not jsonl_path.exists():
            return f'Session transcript file not found for "{session_id}".'

        from kraang.indexer import read_transcript

        turns = read_transcript(jsonl_path)
        return format_transcript(session, turns, max_turns=max_turns)
    except Exception:
        logger.exception("read_session failed")
        return f'Error: could not read session "{session_id}".'


@mcp.tool()
async def forget(
    title: str,
    relevance: float = 0.0,
) -> str:
    """Adjust a note's relevance. Use to downweight or hide outdated/wrong notes.

    - forget("title") -> hidden from search (relevance=0.0)
    - forget("title", 0.3) -> deprioritized (30% of natural score)
    - To restore: use remember() with the same title (resets to 1.0)

    Args:
        title: Title of the note to adjust.
        relevance: Score from 0.0 (hidden) to 1.0 (full weight). Default: 0.0.
    """
    if not (0.0 <= relevance <= 1.0):
        return f"Error: relevance must be between 0.0 and 1.0, got {relevance}."

    try:
        store = await _get_store()
        note = await store.set_relevance(title, relevance)
        if note is None:
            return f'Note "{title}" not found.'
        return format_forget(note.title, relevance)
    except Exception:
        logger.exception("forget failed")
        return f'Error: could not forget "{title}".'


@mcp.tool()
async def status() -> str:
    """Get a knowledge base overview: counts, recent activity, tags."""
    try:
        store = await _get_store()
        provider = await _get_provider()

        active, forgotten = await store.count_notes()
        session_count = await store.count_sessions()
        last_indexed = await store.last_indexed_at()
        recent = await store.recent_notes(days=7, limit=10)
        categories = await store.category_counts()
        tags = await store.tag_counts()
        stale = await store.stale_notes(days=30)

        if provider is not None:
            embedding_status = f"{provider.provider_id}/{provider.model} ({provider.dims} dims)"
        else:
            embedding_status = "disabled (no API key or extras not installed)"

        return format_status(
            active_notes=active,
            forgotten_notes=forgotten,
            session_count=session_count,
            last_indexed=last_indexed,
            recent_notes=recent,
            categories=categories,
            tags=tags,
            stale_notes=stale,
            embedding_status=embedding_status,
        )
    except Exception:
        logger.exception("status failed")
        return "Error: could not generate status."


# ---------------------------------------------------------------------------
# Entry point (used by `kraang serve`)
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the kraang MCP server over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

"""Unified SQLiteStore — notes + sessions, backed by SQLite + FTS5."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import struct
from datetime import datetime, timedelta, timezone
from typing import Any

import aiosqlite

from kraang.config import normalize_title
from kraang.models import (
    Note,
    NoteSearchResult,
    Session,
    SessionSearchResult,
    new_id,
    utcnow,
)

logger = logging.getLogger("kraang.store")

# ---------------------------------------------------------------------------
# sqlite-vec availability
# ---------------------------------------------------------------------------

# True if `import sqlite_vec` succeeded at module load time. Does NOT guarantee
# the extension loads into a specific connection — check self._vec_loaded after initialize().
_SQLITE_VEC_IMPORTABLE = False
try:
    import sqlite_vec  # noqa: F401

    _SQLITE_VEC_IMPORTABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# SQL schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    note_id          TEXT PRIMARY KEY,
    title            TEXT NOT NULL,
    title_normalized TEXT NOT NULL UNIQUE,
    content          TEXT NOT NULL,
    tags_json        TEXT NOT NULL DEFAULT '[]',
    category         TEXT NOT NULL DEFAULT '',
    relevance        REAL NOT NULL DEFAULT 1.0,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_notes_updated ON notes(updated_at);
CREATE INDEX IF NOT EXISTS idx_notes_category ON notes(category);
CREATE INDEX IF NOT EXISTS idx_notes_relevance ON notes(relevance);

CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
    title, content, tags_json,
    content=notes, content_rowid=rowid,
    tokenize='porter unicode61'
);

-- FTS sync triggers
CREATE TRIGGER IF NOT EXISTS notes_ai AFTER INSERT ON notes BEGIN
    INSERT INTO notes_fts(rowid, title, content, tags_json)
    VALUES (NEW.rowid, NEW.title, NEW.content, NEW.tags_json);
END;

CREATE TRIGGER IF NOT EXISTS notes_ad AFTER DELETE ON notes BEGIN
    INSERT INTO notes_fts(notes_fts, rowid, title, content, tags_json)
    VALUES ('delete', OLD.rowid, OLD.title, OLD.content, OLD.tags_json);
END;

CREATE TRIGGER IF NOT EXISTS notes_au AFTER UPDATE ON notes BEGIN
    INSERT INTO notes_fts(notes_fts, rowid, title, content, tags_json)
    VALUES ('delete', OLD.rowid, OLD.title, OLD.content, OLD.tags_json);
    INSERT INTO notes_fts(rowid, title, content, tags_json)
    VALUES (NEW.rowid, NEW.title, NEW.content, NEW.tags_json);
END;

-- Embedding cache
CREATE TABLE IF NOT EXISTS embedding_cache (
    provider     TEXT NOT NULL,
    model        TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    embedding    BLOB NOT NULL,
    dims         INTEGER NOT NULL,
    created_at   TEXT NOT NULL,
    PRIMARY KEY (provider, model, content_hash)
);

CREATE INDEX IF NOT EXISTS idx_embedding_cache_created ON embedding_cache(created_at);

-- Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    session_id        TEXT PRIMARY KEY,
    slug              TEXT,
    project_path      TEXT NOT NULL,
    git_branch        TEXT DEFAULT '',
    model             TEXT DEFAULT '',
    started_at        TEXT NOT NULL,
    ended_at          TEXT NOT NULL,
    duration_s        INTEGER DEFAULT 0,
    user_turn_count   INTEGER DEFAULT 0,
    assistant_turn_count INTEGER DEFAULT 0,
    summary           TEXT DEFAULT '',
    user_text         TEXT DEFAULT '',
    assistant_text    TEXT DEFAULT '',
    tools_used_json   TEXT DEFAULT '[]',
    files_edited_json TEXT DEFAULT '[]',
    source_mtime      REAL NOT NULL,
    source_size       INTEGER NOT NULL,
    indexed_at        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_path);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at);
CREATE INDEX IF NOT EXISTS idx_sessions_slug ON sessions(slug);

CREATE VIRTUAL TABLE IF NOT EXISTS sessions_fts USING fts5(
    summary, user_text, assistant_text,
    content=sessions, content_rowid=rowid,
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS sessions_ai AFTER INSERT ON sessions BEGIN
    INSERT INTO sessions_fts(rowid, summary, user_text, assistant_text)
    VALUES (NEW.rowid, NEW.summary, NEW.user_text, NEW.assistant_text);
END;

CREATE TRIGGER IF NOT EXISTS sessions_ad AFTER DELETE ON sessions BEGIN
    INSERT INTO sessions_fts(sessions_fts, rowid, summary, user_text, assistant_text)
    VALUES ('delete', OLD.rowid, OLD.summary, OLD.user_text, OLD.assistant_text);
END;

CREATE TRIGGER IF NOT EXISTS sessions_au AFTER UPDATE ON sessions BEGIN
    INSERT INTO sessions_fts(sessions_fts, rowid, summary, user_text, assistant_text)
    VALUES ('delete', OLD.rowid, OLD.summary, OLD.user_text, OLD.assistant_text);
    INSERT INTO sessions_fts(rowid, summary, user_text, assistant_text)
    VALUES (NEW.rowid, NEW.summary, NEW.user_text, NEW.assistant_text);
END;
"""


def _dt_to_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _iso_to_dt(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# ---------------------------------------------------------------------------
# SQLiteStore
# ---------------------------------------------------------------------------


class SQLiteStore:
    """Unified store for notes and sessions, backed by SQLite + FTS5."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None
        self._write_lock = asyncio.Lock()
        self._vec_loaded = _SQLITE_VEC_IMPORTABLE

    # -- lifecycle -----------------------------------------------------------

    async def initialize(self) -> None:
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA busy_timeout=5000")
        await self._db.execute("PRAGMA foreign_keys=ON")

        # Load sqlite-vec extension if available.
        if self._vec_loaded:
            try:
                import sqlite_vec

                await self._db.enable_load_extension(True)
                await self._db.execute("SELECT load_extension(?)", (sqlite_vec.loadable_path(),))
                await self._db.enable_load_extension(False)
            except Exception:
                logger.warning(
                    "sqlite-vec failed to load — brute-force vector fallback",
                    exc_info=True,
                )
                self._vec_loaded = False

        await self._db.executescript(_SCHEMA)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def __aenter__(self) -> SQLiteStore:
        await self.initialize()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    @property
    def _conn(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Store not initialised — call initialize() first")
        return self._db

    # =========================================================================
    # NOTES
    # =========================================================================

    def _row_to_note(self, row: aiosqlite.Row) -> Note:
        try:
            tags = json.loads(row["tags_json"]) if row["tags_json"] else []
        except (json.JSONDecodeError, TypeError):
            logger.warning("Invalid tags_json for note %s, falling back to []", row["note_id"])
            tags = []
        return Note(
            note_id=row["note_id"],
            title=row["title"],
            title_normalized=row["title_normalized"],
            content=row["content"],
            tags=tags,
            category=row["category"],
            relevance=row["relevance"],
            created_at=_iso_to_dt(row["created_at"]),
            updated_at=_iso_to_dt(row["updated_at"]),
        )

    # -- upsert (for remember tool) ------------------------------------------

    async def upsert_note(
        self,
        title: str,
        content: str,
        tags: list[str] | None = None,
        category: str = "",
    ) -> tuple[Note, bool]:
        """Create or update a note by title. Returns (note, created).

        If a note with the same normalized title exists, update it.
        Otherwise create a new one. Always sets relevance=1.0.
        """
        norm = normalize_title(title)
        tags_list = tags if tags is not None else []
        tags_json = json.dumps(tags_list)
        now = utcnow()
        note_id = new_id()

        async with self._write_lock:
            # Try to find existing note by normalized title
            cur = await self._conn.execute(
                "SELECT note_id, created_at FROM notes WHERE title_normalized = ?",
                (norm,),
            )
            existing = await cur.fetchone()

            if existing:
                # Update existing note
                await self._conn.execute(
                    """UPDATE notes SET
                        title = ?, content = ?, tags_json = ?, category = ?,
                        relevance = 1.0, updated_at = ?
                    WHERE title_normalized = ?""",
                    (title, content, tags_json, category, _dt_to_iso(now), norm),
                )
                await self._conn.commit()
                note_id = existing["note_id"]
                created_at = _iso_to_dt(existing["created_at"])
                return Note(
                    note_id=note_id,
                    title=title,
                    title_normalized=norm,
                    content=content,
                    tags=tags_list,
                    category=category,
                    relevance=1.0,
                    created_at=created_at,
                    updated_at=now,
                ), False
            else:
                # Create new note
                await self._conn.execute(
                    """INSERT INTO notes
                        (note_id, title, title_normalized, content, tags_json,
                         category, relevance, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, 1.0, ?, ?)""",
                    (
                        note_id,
                        title,
                        norm,
                        content,
                        tags_json,
                        category,
                        _dt_to_iso(now),
                        _dt_to_iso(now),
                    ),
                )
                await self._conn.commit()
                return Note(
                    note_id=note_id,
                    title=title,
                    title_normalized=norm,
                    content=content,
                    tags=tags_list,
                    category=category,
                    relevance=1.0,
                    created_at=now,
                    updated_at=now,
                ), True

    # -- get by title --------------------------------------------------------

    async def get_note_by_title(self, title: str) -> Note | None:
        """Fetch a note by its normalized title."""
        norm = normalize_title(title)
        cur = await self._conn.execute("SELECT * FROM notes WHERE title_normalized = ?", (norm,))
        row = await cur.fetchone()
        if row is None:
            return None
        return self._row_to_note(row)

    # -- get by id -----------------------------------------------------------

    async def get_note(self, note_id: str) -> Note | None:
        """Fetch a single note by ID."""
        cur = await self._conn.execute("SELECT * FROM notes WHERE note_id = ?", (note_id,))
        row = await cur.fetchone()
        if row is None:
            return None
        return self._row_to_note(row)

    # -- set relevance (for forget tool) -------------------------------------

    async def set_relevance(self, title: str, relevance: float) -> Note | None:
        """Set the relevance score of a note by title. Returns updated note or None."""
        relevance = max(0.0, min(1.0, relevance))
        norm = normalize_title(title)
        now = utcnow()
        async with self._write_lock:
            cur = await self._conn.execute(
                "SELECT * FROM notes WHERE title_normalized = ?", (norm,)
            )
            row = await cur.fetchone()
            if row is None:
                return None

            await self._conn.execute(
                "UPDATE notes SET relevance = ?, updated_at = ? WHERE title_normalized = ?",
                (relevance, _dt_to_iso(now), norm),
            )
            await self._conn.commit()
        note = self._row_to_note(row)
        note.relevance = relevance
        note.updated_at = now
        return note

    # -- list notes ----------------------------------------------------------

    async def list_notes(
        self,
        include_forgotten: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Note]:
        """List notes ordered by updated_at descending."""
        if include_forgotten:
            cur = await self._conn.execute(
                "SELECT * FROM notes ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
        else:
            cur = await self._conn.execute(
                "SELECT * FROM notes WHERE relevance > 0.0"
                " ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
        rows = await cur.fetchall()
        return [self._row_to_note(row) for row in rows]

    # -- search notes --------------------------------------------------------

    async def search_notes(
        self,
        fts_expr: str,
        limit: int = 10,
    ) -> list[NoteSearchResult]:
        """Search notes via FTS5. Returns results with relevance-weighted scores.

        Notes with relevance=0.0 are excluded.
        """
        try:
            # bm25() weights: title=3.0, content=1.0, tags_json=2.0
            sql = """
                SELECT notes.*, bm25(notes_fts, 3.0, 1.0, 2.0) AS score,
                       snippet(notes_fts, 1, '>>>', '<<<', '...', 32) AS snip
                FROM notes_fts
                JOIN notes ON notes.rowid = notes_fts.rowid
                WHERE notes_fts MATCH ?
                  AND notes.relevance > 0.0
                ORDER BY (bm25(notes_fts, 3.0, 1.0, 2.0) * notes.relevance)
                LIMIT ?
            """
            cur = await self._conn.execute(sql, (fts_expr, limit))
            rows = await cur.fetchall()

            results: list[NoteSearchResult] = []
            for row in rows:
                note = self._row_to_note(row)
                raw_score = abs(row["score"])
                final_score = raw_score * note.relevance
                results.append(
                    NoteSearchResult(
                        note=note,
                        score=final_score,
                        snippet=row["snip"] or "",
                    )
                )
            return results
        except Exception:
            logger.warning("search_notes failed for query: %s", fts_expr, exc_info=True)
            return []

    # -- find similar titles (for fuzzy duplicate detection) -----------------

    async def find_similar_titles(self, title: str, limit: int = 3) -> list[Note]:
        """Search for notes with similar titles via FTS5."""
        try:
            from kraang.search import build_fts_query

            # Extract meaningful words from the title for search
            fts_expr = build_fts_query(title)
            if not fts_expr:
                return []

            # Search in the title column only
            sql = """
                SELECT notes.*
                FROM notes_fts
                JOIN notes ON notes.rowid = notes_fts.rowid
                WHERE notes_fts MATCH ?
                  AND notes.relevance > 0.0
                ORDER BY bm25(notes_fts, 10.0, 0.0, 0.0)
                LIMIT ?
            """
            cur = await self._conn.execute(sql, (fts_expr, limit))
            rows = await cur.fetchall()
            return [self._row_to_note(row) for row in rows]
        except Exception:
            logger.warning("find_similar_titles failed for: %s", title, exc_info=True)
            return []

    # -- note counts ---------------------------------------------------------

    async def count_notes(self) -> tuple[int, int]:
        """Return (total_active, total_forgotten) note counts."""
        cur = await self._conn.execute("SELECT COUNT(*) AS cnt FROM notes WHERE relevance > 0.0")
        row = await cur.fetchone()
        active = row["cnt"] if row else 0

        cur = await self._conn.execute("SELECT COUNT(*) AS cnt FROM notes WHERE relevance = 0.0")
        row = await cur.fetchone()
        forgotten = row["cnt"] if row else 0

        return active, forgotten

    # -- recent notes --------------------------------------------------------

    async def recent_notes(self, days: int = 7, limit: int = 10) -> list[Note]:
        """Return notes updated within the last N days."""
        cutoff = utcnow() - timedelta(days=days)
        cur = await self._conn.execute(
            """SELECT * FROM notes
            WHERE relevance > 0.0 AND updated_at >= ?
            ORDER BY updated_at DESC LIMIT ?""",
            (_dt_to_iso(cutoff), limit),
        )
        rows = await cur.fetchall()
        return [self._row_to_note(row) for row in rows]

    # -- stale notes ---------------------------------------------------------

    async def stale_notes(self, days: int = 30) -> list[Note]:
        """Find active notes not updated in the last N days."""
        cutoff = utcnow() - timedelta(days=days)
        cur = await self._conn.execute(
            """SELECT * FROM notes
            WHERE relevance > 0.0 AND updated_at < ?
            ORDER BY updated_at ASC""",
            (_dt_to_iso(cutoff),),
        )
        rows = await cur.fetchall()
        return [self._row_to_note(row) for row in rows]

    # -- tag / category counts -----------------------------------------------

    async def tag_counts(self) -> dict[str, int]:
        """Return tag -> count for all active notes."""
        cur = await self._conn.execute(
            """SELECT je.value AS tag, COUNT(*) AS cnt
            FROM notes, json_each(notes.tags_json) AS je
            WHERE notes.relevance > 0.0
            GROUP BY je.value
            ORDER BY cnt DESC"""
        )
        rows = await cur.fetchall()
        return {row["tag"]: row["cnt"] for row in rows}

    async def category_counts(self) -> dict[str, int]:
        """Return category -> count for all active notes."""
        cur = await self._conn.execute(
            """SELECT category, COUNT(*) AS cnt
            FROM notes
            WHERE relevance > 0.0 AND category != ''
            GROUP BY category
            ORDER BY cnt DESC"""
        )
        rows = await cur.fetchall()
        return {row["category"]: row["cnt"] for row in rows}

    # =========================================================================
    # SESSIONS
    # =========================================================================

    def _row_to_session(self, row: aiosqlite.Row) -> Session:
        try:
            tools_used = json.loads(row["tools_used_json"]) if row["tools_used_json"] else []
        except (json.JSONDecodeError, TypeError):
            logger.warning("Invalid tools_used_json for session %s", row["session_id"])
            tools_used = []

        try:
            files_edited = json.loads(row["files_edited_json"]) if row["files_edited_json"] else []
        except (json.JSONDecodeError, TypeError):
            logger.warning("Invalid files_edited_json for session %s", row["session_id"])
            files_edited = []

        return Session(
            session_id=row["session_id"],
            slug=row["slug"] or "",
            project_path=row["project_path"],
            git_branch=row["git_branch"] or "",
            model=row["model"] or "",
            started_at=_iso_to_dt(row["started_at"]),
            ended_at=_iso_to_dt(row["ended_at"]),
            duration_s=row["duration_s"] or 0,
            user_turn_count=row["user_turn_count"] or 0,
            assistant_turn_count=row["assistant_turn_count"] or 0,
            summary=row["summary"] or "",
            user_text=row["user_text"] or "",
            assistant_text=row["assistant_text"] or "",
            tools_used=tools_used,
            files_edited=files_edited,
            source_mtime=row["source_mtime"],
            source_size=row["source_size"],
            indexed_at=_iso_to_dt(row["indexed_at"]),
        )

    # -- upsert session ------------------------------------------------------

    async def upsert_session(self, session: Session) -> None:
        """Insert or replace a session record."""
        async with self._write_lock:
            # Delete first (triggers FTS cleanup), then insert
            await self._conn.execute(
                "DELETE FROM sessions WHERE session_id = ?", (session.session_id,)
            )
            await self._conn.execute(
                """INSERT INTO sessions
                    (session_id, slug, project_path, git_branch, model,
                     started_at, ended_at, duration_s,
                     user_turn_count, assistant_turn_count,
                     summary, user_text, assistant_text,
                     tools_used_json, files_edited_json,
                     source_mtime, source_size, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session.session_id,
                    session.slug,
                    session.project_path,
                    session.git_branch,
                    session.model,
                    _dt_to_iso(session.started_at),
                    _dt_to_iso(session.ended_at),
                    session.duration_s,
                    session.user_turn_count,
                    session.assistant_turn_count,
                    session.summary,
                    session.user_text,
                    session.assistant_text,
                    json.dumps(session.tools_used),
                    json.dumps(session.files_edited),
                    session.source_mtime,
                    session.source_size,
                    _dt_to_iso(session.indexed_at),
                ),
            )
            await self._conn.commit()

    # -- get session ---------------------------------------------------------

    async def get_session(self, session_id_prefix: str) -> Session | None:
        """Fetch a session by full ID or prefix. Raises ValueError if prefix is ambiguous."""
        if len(session_id_prefix) < 36:
            cur = await self._conn.execute(
                "SELECT * FROM sessions WHERE session_id LIKE ? LIMIT 2",
                (session_id_prefix + "%",),
            )
            rows = list(await cur.fetchall())
            if len(rows) == 0:
                return None
            if len(rows) > 1:
                candidates = [r["session_id"][:12] for r in rows]
                raise ValueError(
                    f"Ambiguous prefix '{session_id_prefix}' matches multiple sessions: "
                    + ", ".join(candidates)
                )
            return self._row_to_session(rows[0])
        else:
            cur = await self._conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id_prefix,),
            )
            row = await cur.fetchone()
            if row is None:
                return None
            return self._row_to_session(row)

    # -- list sessions -------------------------------------------------------

    async def list_sessions(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Session]:
        """List sessions ordered by started_at descending."""
        cur = await self._conn.execute(
            "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        rows = await cur.fetchall()
        return [self._row_to_session(row) for row in rows]

    # -- search sessions -----------------------------------------------------

    async def search_sessions(
        self,
        fts_expr: str,
        limit: int = 10,
    ) -> list[SessionSearchResult]:
        """Search sessions via FTS5 with BM25 ranking.

        Weights: summary=5.0, user_text=3.0, assistant_text=1.0
        """
        try:
            sql = """
                SELECT sessions.*, bm25(sessions_fts, 5.0, 3.0, 1.0) AS score,
                       snippet(sessions_fts, 1, '>>>', '<<<', '...', 32) AS snip
                FROM sessions_fts
                JOIN sessions ON sessions.rowid = sessions_fts.rowid
                WHERE sessions_fts MATCH ?
                ORDER BY score
                LIMIT ?
            """
            cur = await self._conn.execute(sql, (fts_expr, limit))
            rows = await cur.fetchall()

            results: list[SessionSearchResult] = []
            for row in rows:
                session = self._row_to_session(row)
                results.append(
                    SessionSearchResult(
                        session=session,
                        score=abs(row["score"]),
                        snippet=row["snip"] or "",
                    )
                )
            return results
        except Exception:
            logger.warning("search_sessions failed for query: %s", fts_expr, exc_info=True)
            return []

    # -- session counts ------------------------------------------------------

    async def count_sessions(self) -> int:
        """Return total number of indexed sessions."""
        cur = await self._conn.execute("SELECT COUNT(*) AS cnt FROM sessions")
        row = await cur.fetchone()
        return row["cnt"] if row else 0

    # -- check if session needs reindexing -----------------------------------

    async def needs_reindex(self, session_id: str, mtime: float, size: int) -> bool:
        """Check if a session file has changed since last indexing."""
        cur = await self._conn.execute(
            "SELECT source_mtime, source_size FROM sessions WHERE session_id = ?",
            (session_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return True
        return bool(row["source_mtime"] != mtime or row["source_size"] != size)

    # -- last indexed timestamp -----------------------------------------------

    async def last_indexed_at(self) -> datetime | None:
        """Return the most recent indexed_at timestamp across all sessions."""
        cur = await self._conn.execute("SELECT MAX(indexed_at) AS last FROM sessions")
        row = await cur.fetchone()
        if row is None or row["last"] is None:
            return None
        return _iso_to_dt(row["last"])

    # =========================================================================
    # EMBEDDING CACHE
    # =========================================================================

    async def get_cached_embedding(
        self,
        provider: str,
        model: str,
        content_hash: str,
    ) -> list[float] | None:
        """Retrieve a cached embedding, or ``None`` if not found."""
        cur = await self._conn.execute(
            "SELECT embedding, dims FROM embedding_cache "
            "WHERE provider = ? AND model = ? AND content_hash = ?",
            (provider, model, content_hash),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        try:
            return list(struct.unpack(f"<{row['dims']}f", row["embedding"]))
        except struct.error:
            logger.warning("Corrupted embedding cache entry for hash %s — skipping", content_hash)
            return None

    async def cache_embedding(
        self,
        provider: str,
        model: str,
        content_hash: str,
        embedding: list[float],
        dims: int,
    ) -> None:
        """Store an embedding in the cache."""
        blob = struct.pack(f"<{dims}f", *embedding)
        now = _dt_to_iso(utcnow())
        async with self._write_lock:
            await self._conn.execute(
                "INSERT OR REPLACE INTO embedding_cache "
                "(provider, model, content_hash, embedding, dims, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (provider, model, content_hash, blob, dims, now),
            )
            await self._conn.commit()

    # =========================================================================
    # VECTOR SEARCH
    # =========================================================================

    async def ensure_vec_table(self, dims: int) -> None:
        """Create the ``notes_vec`` virtual table if it doesn't exist.

        No-op when sqlite-vec is unavailable.
        """
        if not self._vec_loaded:
            logger.debug("sqlite-vec not available — skipping notes_vec creation")
            return
        async with self._write_lock:
            try:
                await self._conn.execute(
                    f"CREATE VIRTUAL TABLE IF NOT EXISTS notes_vec "
                    f"USING vec0(note_id TEXT PRIMARY KEY, embedding float[{dims}])"
                )
                await self._conn.commit()
            except Exception:
                logger.warning("Failed to create notes_vec table", exc_info=True)
                self._vec_loaded = False

    async def upsert_note_embedding(self, note_id: str, embedding: list[float]) -> None:
        """Insert or update the vector for a note."""
        blob = struct.pack(f"<{len(embedding)}f", *embedding)
        async with self._write_lock:
            if self._vec_loaded:
                # vec0 doesn't support ON CONFLICT — delete then insert.
                await self._conn.execute("DELETE FROM notes_vec WHERE note_id = ?", (note_id,))
                await self._conn.execute(
                    "INSERT INTO notes_vec (note_id, embedding) VALUES (?, ?)",
                    (note_id, blob),
                )
            else:
                # Fallback: Reuse embedding_cache table to store note vectors when
                # sqlite-vec is unavailable. Sentinel values '_vec'/'_fallback' distinguish
                # these from real API-response cache entries. The content_hash column
                # holds the note_id (not an actual content hash) in this case.
                # See also _search_vec_bruteforce() which queries these sentinel entries.
                now = _dt_to_iso(utcnow())
                dims = len(embedding)
                await self._conn.execute(
                    "INSERT OR REPLACE INTO embedding_cache "
                    "(provider, model, content_hash, embedding, dims, created_at) "
                    "VALUES ('_vec', '_fallback', ?, ?, ?, ?)",
                    (note_id, blob, dims, now),
                )
            await self._conn.commit()

    async def search_notes_vector(
        self,
        query_embedding: list[float],
        limit: int = 10,
    ) -> list[NoteSearchResult]:
        """Search notes by vector similarity.

        Uses sqlite-vec when available, otherwise falls back to brute-force
        cosine similarity over cached embeddings.
        """
        if self._vec_loaded:
            return await self._search_vec_native(query_embedding, limit)
        return await self._search_vec_bruteforce(query_embedding, limit)

    # -- native sqlite-vec search --------------------------------------------

    async def _search_vec_native(
        self,
        query_embedding: list[float],
        limit: int,
    ) -> list[NoteSearchResult]:
        """Vector search via sqlite-vec ``vec_distance_cosine``."""
        try:
            blob = struct.pack(f"<{len(query_embedding)}f", *query_embedding)
            sql = """
                SELECT v.note_id, vec_distance_cosine(v.embedding, ?) AS dist
                FROM notes_vec v
                JOIN notes n ON n.note_id = v.note_id
                WHERE n.relevance > 0.0
                ORDER BY dist ASC
                LIMIT ?
            """
            cur = await self._conn.execute(sql, (blob, limit))
            rows = await cur.fetchall()

            results: list[NoteSearchResult] = []
            for row in rows:
                note = await self.get_note(row["note_id"])
                if note is None:
                    continue
                # vec_distance_cosine returns 1 - cosine_similarity, ranging from
                # 0 (identical) to 2 (opposite). Clamp to avoid negative scores.
                score = max(0.0, 1.0 - row["dist"]) * note.relevance
                results.append(NoteSearchResult(note=note, score=score, snippet=""))
            return results
        except Exception:
            logger.warning("Native vector search failed", exc_info=True)
            return []

    # -- brute-force fallback ------------------------------------------------

    async def _search_vec_bruteforce(
        self,
        query_embedding: list[float],
        limit: int,
    ) -> list[NoteSearchResult]:
        """Brute-force cosine similarity over embeddings in the cache."""
        try:
            cur = await self._conn.execute(
                "SELECT content_hash AS note_id, embedding, dims "
                "FROM embedding_cache WHERE provider = '_vec' AND model = '_fallback'"
            )
            rows = await cur.fetchall()
            if not rows:
                return []

            scored: list[tuple[str, float]] = []
            for row in rows:
                try:
                    vec = list(struct.unpack(f"<{row['dims']}f", row["embedding"]))
                except struct.error:
                    logger.warning("Corrupted embedding in bruteforce search — skipping entry")
                    continue
                sim = _cosine_similarity(query_embedding, vec)
                scored.append((row["note_id"], sim))

            scored.sort(key=lambda t: t[1], reverse=True)

            results: list[NoteSearchResult] = []
            for note_id, sim in scored:
                if len(results) >= limit:
                    break
                note = await self.get_note(note_id)
                if note is None or note.relevance <= 0.0:
                    continue
                score = sim * note.relevance
                results.append(NoteSearchResult(note=note, score=score, snippet=""))
            return results
        except Exception:
            logger.warning("Brute-force vector search failed", exc_info=True)
            return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def content_hash(text: str) -> str:
    """SHA-256 hash of text content, for embedding cache keys.

    This is the canonical hashing function — server.py imports this
    rather than computing hashes inline.
    """
    return hashlib.sha256(text.encode()).hexdigest()

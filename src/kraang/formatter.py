"""Markdown output formatting for MCP tool responses."""

from __future__ import annotations

from datetime import datetime, timezone

from kraang.models import (
    Note,
    NoteSearchResult,
    Session,
    SessionSearchResult,
    TranscriptTurn,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _relative_time(dt: datetime) -> str:
    """Format a datetime as a relative time string (e.g. '2d ago')."""
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = now - dt
    days = delta.days

    if days == 0:
        hours = delta.seconds // 3600
        if hours == 0:
            minutes = delta.seconds // 60
            return f"{minutes}m ago" if minutes > 0 else "just now"
        return f"{hours}h ago"
    elif days == 1:
        return "1d ago"
    elif days < 30:
        return f"{days}d ago"
    elif days < 365:
        months = days // 30
        return f"{months}mo ago"
    else:
        years = days // 365
        return f"{years}y ago"


def _format_duration(seconds: int) -> str:
    """Format seconds as a human-readable duration."""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes}m {secs}s" if secs else f"{minutes}m"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m" if minutes else f"{hours}h"


def _format_tags(tags: list[str]) -> str:
    """Format tags as a pipe-separated string."""
    return " | ".join(tags) if tags else ""


def _format_date(dt: datetime) -> str:
    """Format a datetime for display."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# remember tool output
# ---------------------------------------------------------------------------


def format_remember_created(note: Note, similar: list[Note] | None = None) -> str:
    """Format output for a newly created note."""
    parts: list[str] = []
    tag_str = f" (tags: {_format_tags(note.tags)})" if note.tags else ""
    cat_str = f" | category: {note.category}" if note.category else ""
    parts.append(f'Created "{note.title}"{tag_str}{cat_str}')

    if similar:
        for s in similar:
            parts.append(
                f'Note: similar existing note "{s.title}"'
                " — use that exact title to update it instead."
            )

    return "\n".join(parts)


def format_remember_updated(note: Note) -> str:
    """Format output for an updated note."""
    tag_str = f" (tags: {_format_tags(note.tags)})" if note.tags else ""
    cat_str = f" | category: {note.category}" if note.category else ""
    return f'Updated "{note.title}"{tag_str}{cat_str}'


# ---------------------------------------------------------------------------
# recall tool output
# ---------------------------------------------------------------------------


def format_recall_results(
    query: str,
    notes: list[NoteSearchResult],
    sessions: list[SessionSearchResult],
) -> str:
    """Format combined search results as markdown."""
    if not notes and not sessions:
        return f'No results found for "{query}".'

    parts: list[str] = [f'## Results for "{query}"']

    if notes:
        parts.append(f"\n### Notes ({len(notes)} {'match' if len(notes) == 1 else 'matches'})\n")
        for nr in notes:
            n = nr.note
            cat_str = f" ({n.category})" if n.category else ""
            parts.append(f"**{n.title}**{cat_str}")
            tag_line = f"Tags: {', '.join(n.tags)}" if n.tags else ""
            date_line = f"Updated: {_format_date(n.updated_at)}"
            meta = " | ".join(filter(None, [tag_line, date_line]))
            if meta:
                parts.append(meta)
            snippet = nr.snippet or n.content[:200]
            parts.append(f"> {snippet}\n")

    if sessions:
        count = len(sessions)
        label = "match" if count == 1 else "matches"
        parts.append(f"\n### Sessions ({count} {label})\n")
        for sr in sessions:
            s = sr.session
            duration = _format_duration(s.duration_s)
            total_turns = s.user_turn_count + s.assistant_turn_count
            slug_str = f" {s.slug}" if s.slug else ""
            date = _format_date(s.started_at)
            parts.append(f"**{date}{slug_str}** ({duration}, {total_turns} turns)")
            meta_parts: list[str] = []
            if s.git_branch:
                meta_parts.append(f"Branch: `{s.git_branch}`")
            meta_parts.append(f"ID: `{s.session_id[:8]}`")
            parts.append(" | ".join(meta_parts))

            # Show summary snippet
            snippet = sr.snippet or s.summary[:200]
            if snippet:
                parts.append(f"> {snippet}\n")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# read_session tool output
# ---------------------------------------------------------------------------


def format_transcript(
    session: Session,
    turns: list[TranscriptTurn],
    max_turns: int = 0,
) -> str:
    """Format a session transcript as markdown."""
    parts: list[str] = []

    # Header
    slug_str = f": {session.slug}" if session.slug else ""
    parts.append(f"## Session{slug_str}")

    meta_parts = [f"Date: {_format_date(session.started_at)}"]
    if session.duration_s:
        meta_parts.append(f"Duration: {_format_duration(session.duration_s)}")
    if session.git_branch:
        meta_parts.append(f"Branch: {session.git_branch}")
    parts.append(" | ".join(meta_parts))
    parts.append(f"ID: {session.session_id}")
    parts.append("\n---\n")

    # Turns
    display_turns = turns[:max_turns] if max_turns > 0 else turns
    for turn in display_turns:
        ts_str = ""
        if turn.timestamp:
            try:
                ts = datetime.fromisoformat(turn.timestamp.replace("Z", "+00:00"))
                ts_str = f" ({ts.strftime('%H:%M')})"
            except (ValueError, TypeError):
                pass

        parts.append(f"**{turn.role}**{ts_str}:")
        if turn.text:
            # Truncate very long text blocks
            text = turn.text
            if len(text) > 2000:
                text = text[:2000] + "\n...(truncated)"
            parts.append(text)
        if turn.tool_calls:
            for tc in turn.tool_calls:
                parts.append(f"- {tc}")
        parts.append("")

    # Footer
    total_turns = len(turns)
    parts.append("---")
    footer_parts = [f"*{total_turns} turns"]
    if session.model:
        footer_parts.append(f"Model: {session.model}")
    parts.append(" | ".join(footer_parts) + "*")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# forget tool output
# ---------------------------------------------------------------------------


def format_forget(title: str, relevance: float) -> str:
    """Format output for the forget tool."""
    if relevance == 0.0:
        return f'Forgot "{title}" (relevance: 0.0, hidden from search)'
    return f'Forgot "{title}" (relevance: {relevance:.1f})'


# ---------------------------------------------------------------------------
# status tool output
# ---------------------------------------------------------------------------


def format_status(
    active_notes: int,
    forgotten_notes: int,
    session_count: int,
    last_indexed: datetime | None,
    recent_notes: list[Note],
    categories: dict[str, int],
    tags: dict[str, int],
    stale_notes: list[Note],
    embedding_status: str = "",
) -> str:
    """Format the status overview as markdown."""
    parts: list[str] = ["## Kraang Status\n"]

    # Counts
    total_notes = active_notes + forgotten_notes
    parts.append(
        f"**Notes:** {total_notes} total ({active_notes} active, {forgotten_notes} forgotten)"
    )
    indexed_str = _format_date(last_indexed) if last_indexed else "never"
    parts.append(f"**Sessions indexed:** {session_count} (last indexed: {indexed_str})")
    if embedding_status:
        parts.append(f"**Embeddings:** {embedding_status}")

    # Recent notes
    if recent_notes:
        parts.append("\n### Recent Notes (7 days)")
        for note in recent_notes:
            cat_str = f" ({note.category})" if note.category else ""
            parts.append(f"- {note.title}{cat_str} — {_relative_time(note.updated_at)}")

    # Categories
    if categories:
        parts.append("\n### Categories")
        cat_parts = [f"{cat} ({count})" for cat, count in categories.items()]
        parts.append(" | ".join(cat_parts))

    # Tags
    if tags:
        parts.append("\n### Tags")
        tag_parts = [f"{tag} ({count})" for tag, count in list(tags.items())[:15]]
        parts.append(" | ".join(tag_parts))

    # Stale notes
    if stale_notes:
        parts.append("\n### Stale (>30 days)")
        for note in stale_notes[:5]:
            cat_str = f" ({note.category})" if note.category else ""
            parts.append(f"- {note.title}{cat_str} — {_relative_time(note.updated_at)}")

    return "\n".join(parts)

"""Rich terminal formatting for CLI output."""

from __future__ import annotations

from datetime import UTC, datetime

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from kraang.models import Note, Session, TranscriptTurn

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _relative_time(dt: datetime) -> str:
    """Format a datetime as a relative time string."""
    now = datetime.now(UTC)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
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
    else:
        months = days // 30
        return f"{months}mo ago"


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


def _format_date(dt: datetime) -> str:
    """Format a datetime for display."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.strftime("%b %d %I:%M %p")


# ---------------------------------------------------------------------------
# Sessions table
# ---------------------------------------------------------------------------


def display_sessions(sessions: list[Session]) -> None:
    """Display sessions as a rich table."""
    if not sessions:
        console.print("[dim]No sessions found.[/dim]")
        return

    table = Table(title="Sessions", show_header=True, header_style="bold")
    table.add_column("ID", style="cyan", width=10)
    table.add_column("Started", width=18)
    table.add_column("Duration", width=10)
    table.add_column("Branch", width=20)
    table.add_column("Summary", no_wrap=False)

    for s in sessions:
        summary = s.summary[:60] + "..." if len(s.summary) > 60 else s.summary
        # Remove newlines from summary for table display
        summary = summary.replace("\n", " ")
        table.add_row(
            s.session_id[:8],
            _format_date(s.started_at),
            _format_duration(s.duration_s),
            s.git_branch or "-",
            summary,
        )

    console.print(table)


# ---------------------------------------------------------------------------
# Session transcript
# ---------------------------------------------------------------------------


def display_transcript(session: Session, turns: list[TranscriptTurn]) -> None:
    """Display a session transcript with rich panels."""
    # Header
    slug_str = f": {session.slug}" if session.slug else ""
    header = f"Session{slug_str}"
    meta_parts = [f"Date: {_format_date(session.started_at)}"]
    if session.duration_s:
        meta_parts.append(f"Duration: {_format_duration(session.duration_s)}")
    if session.git_branch:
        meta_parts.append(f"Branch: {session.git_branch}")
    if session.model:
        meta_parts.append(f"Model: {session.model}")
    meta = " | ".join(meta_parts)

    console.print(Panel(f"[bold]{header}[/bold]\n{meta}\nID: {session.session_id}"))

    for turn in turns:
        ts_str = ""
        if turn.timestamp:
            try:
                ts = datetime.fromisoformat(turn.timestamp.replace("Z", "+00:00"))
                ts_str = f" ({ts.strftime('%H:%M')})"
            except (ValueError, TypeError):
                pass

        style = "green" if turn.role == "User" else "blue"
        parts: list[str] = []
        if turn.text:
            text = turn.text
            if len(text) > 3000:
                text = text[:3000] + "\n...(truncated)"
            parts.append(text)
        if turn.tool_calls:
            for tc in turn.tool_calls:
                parts.append(f"  - {tc}")

        content = "\n".join(parts)
        console.print(
            Panel(
                content,
                title=f"[{style}]{turn.role}{ts_str}[/{style}]",
                title_align="left",
                border_style=style,
            )
        )

    total = len(turns)
    console.print(f"\n[dim]{total} turns | ID: {session.session_id}[/dim]")


# ---------------------------------------------------------------------------
# Search results
# ---------------------------------------------------------------------------


def display_search_results(
    query: str,
    notes: list[tuple[Note, float, str]],
    sessions: list[tuple[Session, float, str]],
) -> None:
    """Display combined search results."""
    if not notes and not sessions:
        console.print(f'[dim]No results found for "{query}".[/dim]')
        return

    console.print(f'\n[bold]Results for "{query}"[/bold]\n')

    if notes:
        console.print(f"[bold]Notes ({len(notes)} matches)[/bold]")
        for note, score, snippet in notes:
            cat_str = f" ({note.category})" if note.category else ""
            tag_str = ", ".join(note.tags) if note.tags else ""
            console.print(f"  [cyan]{note.title}[/cyan]{cat_str} [dim]score: {score:.2f}[/dim]")
            if tag_str:
                console.print(f"    Tags: {tag_str}")
            if snippet:
                console.print(f"    {snippet}")
            console.print()

    if sessions:
        console.print(f"[bold]Sessions ({len(sessions)} matches)[/bold]")
        for session, _score, snippet in sessions:
            slug_str = f" {session.slug}" if session.slug else ""
            console.print(
                f"  [cyan]{_format_date(session.started_at)}{slug_str}[/cyan] "
                f"({_format_duration(session.duration_s)}) [dim]ID: {session.session_id[:8]}[/dim]"
            )
            if snippet:
                console.print(f"    {snippet}")
            console.print()


# ---------------------------------------------------------------------------
# Notes list
# ---------------------------------------------------------------------------


def display_notes(notes: list[Note]) -> None:
    """Display notes as a rich table."""
    if not notes:
        console.print("[dim]No notes found.[/dim]")
        return

    table = Table(title="Notes", show_header=True, header_style="bold")
    table.add_column("Title", no_wrap=False)
    table.add_column("Category", width=15)
    table.add_column("Tags", width=25)
    table.add_column("Updated", width=12)
    table.add_column("Relevance", width=10)

    for note in notes:
        tags = ", ".join(note.tags) if note.tags else "-"
        rel_style = "dim" if note.relevance < 1.0 else ""
        rel_str = f"{note.relevance:.1f}" if note.relevance < 1.0 else "1.0"
        table.add_row(
            note.title,
            note.category or "-",
            tags,
            _relative_time(note.updated_at),
            f"[{rel_style}]{rel_str}[/{rel_style}]" if rel_style else rel_str,
        )

    console.print(table)


# ---------------------------------------------------------------------------
# Status display
# ---------------------------------------------------------------------------


def display_status(markdown_status: str) -> None:
    """Display the status overview using rich markdown rendering."""
    console.print(Markdown(markdown_status))


# ---------------------------------------------------------------------------
# Init summary
# ---------------------------------------------------------------------------


def display_init_summary(
    db_path: str,
    mcp_json_updated: bool,
    hook_configured: bool,
    sessions_indexed: int,
    gitignore_updated: bool = False,
    rules_configured: bool = False,
) -> None:
    """Display summary of kraang init."""
    parts = ["[bold green]kraang initialized![/bold green]\n"]
    parts.append(f"  Database: {db_path}")
    parts.append(f"  .mcp.json: {'updated' if mcp_json_updated else 'already configured'}")
    parts.append(f"  .gitignore: {'updated' if gitignore_updated else 'already configured'}")
    parts.append(f"  SessionEnd hook: {'configured' if hook_configured else 'already configured'}")
    parts.append(f"  Rules file: {'created' if rules_configured else 'already configured'}")
    if sessions_indexed > 0:
        parts.append(f"  Sessions indexed: {sessions_indexed}")
    else:
        parts.append("  Sessions indexed: 0 (no existing sessions)")
    console.print(Panel("\n".join(parts), title="Setup Complete"))

    console.print("\n  [bold]Next steps:[/bold]")
    console.print("    Start a new Claude Code session — kraang is ready to use.")
    console.print("    Run [cyan]kraang status[/cyan] to check your knowledge base.\n")

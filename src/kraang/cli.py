"""Typer CLI for kraang — init, serve, index, sessions, session, search, notes, status."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import typer

from kraang.config import find_project_root, resolve_db_path

logger = logging.getLogger("kraang.cli")

app = typer.Typer(
    name="kraang",
    help="A second brain for humans and their agents.",
    no_args_is_help=True,
)


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


def _backup_file(path: Path, kraang_dir: Path) -> Path:
    """Create a timestamped backup of *path* inside .kraang/backups/."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_dir = kraang_dir / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / f"{path.name}.{ts}.bak"
    shutil.copy2(path, backup)
    return backup


_KRAANG_RULES = """\
# Kraang — Second Brain

You have access to the **kraang** MCP server, a persistent knowledge base.
Use it to save knowledge worth preserving and to retrieve context \
from prior sessions when relevant.

## When to save notes

Use `remember` whenever you encounter:

- **Key decisions** — architecture choices, trade-offs considered, why option A was picked over B
- **Debugging breakthroughs** — root causes found, non-obvious fixes, "gotchas" for this codebase
- **Patterns & conventions** — coding patterns, naming conventions, or project-specific idioms
- **Environment & setup** — dependency quirks, config requirements, toolchain details
- **User preferences** — the user's preferred style, tools, or workflow patterns
- **Reusable knowledge** — anything that would save time if recalled in a future session

## How to write good notes

- Use a clear, searchable `title` (e.g., "pytest-asyncio requires auto mode in this repo")
- Write `content` that your future self (or another agent) can act on without extra context
- Add relevant `tags` (e.g., `["python", "testing", "pytest"]`)
- Use `category` to group by domain (e.g., `"debugging"`, `"architecture"`, `"setup"`)

## When to search

Use `recall` only when the user's message suggests prior work, \
decisions, or context that you don't have from the current conversation.

**Recall when you detect:**

- **Continuity signals** — "pick up where we left off," \
"continue with X," "back to X," "revisit," "let's resume"
- **Reference signals** — "like we did," "same approach as," \
"the pattern we used," "the way we handle"
- **History questions** — "why do we X," "what was the issue with," \
"how did we set up," "what did we decide"
- **Explicit requests** — "check your notes," "search memory," \
"do you remember"
- **Convention/preference queries** — "what's our convention for," \
"how do we usually," "what's the preferred way"
- **Recurring-problem signals** — "this broke again," \
"same error as before," "I'm hitting that issue again"

**Do NOT recall when:**

- Starting a session with a clearly new, self-contained task
- The request is a general coding/knowledge question unrelated to project history
- The task is purely forward-looking with no backward reference
- The current conversation already contains the needed context
- You are unsure — default to NOT recalling; the user can always ask you to search

## Housekeeping

- When you notice a note is outdated, update it with `remember` (same title)
- Use `forget` to downweight notes that are no longer relevant
- Use `status` to check for stale notes periodically
"""


# ---------------------------------------------------------------------------
# kraang init
# ---------------------------------------------------------------------------


@app.command()
def init(
    path: str = typer.Argument(None, help="Project root path (default: auto-detect)"),
) -> None:
    """Set up kraang for the current project."""
    root = Path(path) if path else find_project_root()
    root = root.resolve()

    from kraang.display import console, display_init_summary

    console.print(f"Initializing kraang in [bold]{root}[/bold]...\n")

    _markers = (".git", "pyproject.toml", "package.json", "Cargo.toml")
    if not any((root / m).exists() for m in _markers):
        console.print(
            "  [yellow]![/yellow] No project root marker found "
            "(.git, pyproject.toml, etc.). Using current directory.\n"
        )

    # 1. Create .kraang/ directory and database
    kraang_dir = root / ".kraang"
    db_path = kraang_dir / "kraang.db"
    db_existed = db_path.exists()

    try:
        kraang_dir.mkdir(exist_ok=True)

        async def _init_db() -> None:
            from kraang.store import SQLiteStore

            async with SQLiteStore(str(db_path)):
                pass  # Schema created on initialize

        _run(_init_db())
    except (PermissionError, OSError) as exc:
        console.print(f"  [red]✗[/red] Failed to create database: {exc}")
        raise typer.Exit(1) from None

    if db_existed:
        console.print("  [dim]=[/dim] Database already exists")
    else:
        console.print("  [green]+[/green] Database created")

    # 2. Add .kraang/ to .gitignore
    gitignore = root / ".gitignore"
    gitignore_updated = False
    if gitignore.exists():
        content = gitignore.read_text()
        if ".kraang/" not in content:
            with open(gitignore, "a") as f:
                if not content.endswith("\n"):
                    f.write("\n")
                f.write("\n# Kraang\n.kraang/\n")
            gitignore_updated = True
    else:
        gitignore.write_text("# Kraang\n.kraang/\n")
        gitignore_updated = True

    if gitignore_updated:
        console.print("  [green]+[/green] .gitignore updated")
    else:
        console.print("  [dim]=[/dim] .gitignore already configured")

    # 3. Create/merge .mcp.json
    mcp_json_path = root / ".mcp.json"
    mcp_config = {
        "mcpServers": {
            "kraang": {
                "command": "uvx",
                "args": ["kraang", "serve"],
                "env": {"KRAANG_DB_PATH": ".kraang/kraang.db"},
            }
        }
    }

    mcp_json_updated = False
    mcp_json_refreshed = False
    if mcp_json_path.exists():
        try:
            existing = json.loads(mcp_json_path.read_text())
            if "mcpServers" not in existing:
                existing["mcpServers"] = {}
            if "kraang" not in existing["mcpServers"]:
                existing["mcpServers"]["kraang"] = mcp_config["mcpServers"]["kraang"]
                mcp_json_path.write_text(json.dumps(existing, indent=2) + "\n")
                mcp_json_updated = True
            elif existing["mcpServers"]["kraang"] != mcp_config["mcpServers"]["kraang"]:
                existing["mcpServers"]["kraang"] = mcp_config["mcpServers"]["kraang"]
                mcp_json_path.write_text(json.dumps(existing, indent=2) + "\n")
                mcp_json_updated = True
                mcp_json_refreshed = True
        except (json.JSONDecodeError, KeyError):
            backup = _backup_file(mcp_json_path, kraang_dir)
            mcp_json_path.write_text(json.dumps(mcp_config, indent=2) + "\n")
            mcp_json_updated = True
            console.print(f"  [yellow]![/yellow] Corrupt .mcp.json backed up to {backup}")
    else:
        mcp_json_path.write_text(json.dumps(mcp_config, indent=2) + "\n")
        mcp_json_updated = True

    if mcp_json_refreshed:
        console.print("  [yellow]~[/yellow] .mcp.json updated (kraang config refreshed)")
    elif mcp_json_updated:
        console.print("  [green]+[/green] .mcp.json configured")
    else:
        console.print("  [dim]=[/dim] .mcp.json already configured")

    import os

    if os.environ.get("OPENAI_API_KEY"):
        console.print(
            "  [dim]Tip: Add OPENAI_API_KEY to .mcp.json env"
            " to enable semantic search in Claude Code[/dim]"
        )

    # 4. Create/merge .claude/settings.json with SessionEnd hook
    claude_dir = root / ".claude"
    claude_dir.mkdir(exist_ok=True)
    settings_path = claude_dir / "settings.json"

    hook_config = {
        "hooks": {
            "SessionEnd": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": "uvx kraang index --from-hook",
                            "timeout": 120,
                        }
                    ]
                }
            ]
        }
    }

    hook_configured = False
    kraang_hook_command = "uvx kraang index --from-hook"
    if settings_path.exists():
        try:
            existing = json.loads(settings_path.read_text())
            if "hooks" not in existing:
                existing["hooks"] = {}
            if "SessionEnd" not in existing["hooks"]:
                existing["hooks"]["SessionEnd"] = hook_config["hooks"]["SessionEnd"]
                settings_path.write_text(json.dumps(existing, indent=2) + "\n")
                hook_configured = True
            else:
                # Check if kraang hook already exists in SessionEnd entries
                has_kraang = False
                for entry in existing["hooks"]["SessionEnd"]:
                    for hook in entry.get("hooks", []):
                        if hook.get("command") == kraang_hook_command:
                            has_kraang = True
                            break
                    if has_kraang:
                        break
                if not has_kraang:
                    existing["hooks"]["SessionEnd"].append(hook_config["hooks"]["SessionEnd"][0])
                    settings_path.write_text(json.dumps(existing, indent=2) + "\n")
                    hook_configured = True
        except (json.JSONDecodeError, KeyError):
            backup = _backup_file(settings_path, kraang_dir)
            settings_path.write_text(json.dumps(hook_config, indent=2) + "\n")
            hook_configured = True
            console.print(f"  [yellow]![/yellow] Corrupt settings.json backed up to {backup}")
    else:
        settings_path.write_text(json.dumps(hook_config, indent=2) + "\n")
        hook_configured = True

    if hook_configured:
        console.print("  [green]+[/green] SessionEnd hook configured")
    else:
        console.print("  [dim]=[/dim] SessionEnd hook already configured")

    # 5. Run initial index
    async def _index() -> int:
        from kraang.indexer import index_sessions
        from kraang.store import SQLiteStore

        async with SQLiteStore(str(db_path)) as store:
            return await index_sessions(store, project_path=root)

    try:
        with console.status("[dim]Indexing existing sessions...[/dim]"):
            sessions_indexed = _run(_index())
    except Exception:
        sessions_indexed = 0
        console.print("  [yellow]![/yellow] Indexing failed (non-fatal, run 'kraang index' later)")

    # 6. Create .claude/rules/kraang.md
    rules_dir = claude_dir / "rules"
    rules_file = rules_dir / "kraang.md"
    rules_configured = False
    if not rules_file.exists():
        rules_dir.mkdir(parents=True, exist_ok=True)
        rules_file.write_text(_KRAANG_RULES)
        rules_configured = True
        console.print("  [green]+[/green] .claude/rules/kraang.md created")
    else:
        console.print("  [dim]=[/dim] .claude/rules/kraang.md already exists")

    display_init_summary(
        db_path=str(db_path),
        mcp_json_updated=mcp_json_updated,
        hook_configured=hook_configured,
        sessions_indexed=sessions_indexed,
        gitignore_updated=gitignore_updated,
        rules_configured=rules_configured,
    )


# ---------------------------------------------------------------------------
# kraang serve
# ---------------------------------------------------------------------------


@app.command()
def serve() -> None:
    """Run the MCP server (invoked by Claude Code)."""
    from kraang.server import main

    main()


# ---------------------------------------------------------------------------
# kraang index
# ---------------------------------------------------------------------------


@app.command()
def index(
    from_hook: bool = typer.Option(
        False, "--from-hook", help="Read session info from stdin (hook mode)"
    ),
    path: str = typer.Argument(None, help="Project root path (default: auto-detect)"),
) -> None:
    """Index/re-index conversation sessions for this project."""

    async def _do_index() -> int:
        from kraang.indexer import index_sessions
        from kraang.store import SQLiteStore

        root = Path(path) if path else find_project_root()
        root = root.resolve()
        db_path = resolve_db_path(root)

        if not db_path.exists():
            typer.echo(f"Database not found at {db_path}. Run 'kraang init' first.", err=True)
            raise typer.Exit(1)

        async with SQLiteStore(str(db_path)) as store:
            if from_hook:
                # Read session info from stdin
                try:
                    stdin_data = sys.stdin.read()
                    hook_info = json.loads(stdin_data)
                    transcript_path = hook_info.get("transcript_path", "")
                    if transcript_path and Path(transcript_path).exists():
                        return await index_sessions(
                            store,
                            project_path=hook_info.get("cwd", str(root)),
                            single_file=Path(transcript_path),
                        )
                    return 0
                except (json.JSONDecodeError, KeyError):
                    # Fallback: index all
                    return await index_sessions(store, project_path=root)
            else:
                return await index_sessions(store, project_path=root)

    count = _run(_do_index())
    if not from_hook:
        typer.echo(f"Indexed {count} session(s).")


# ---------------------------------------------------------------------------
# kraang sessions
# ---------------------------------------------------------------------------


@app.command()
def sessions(
    limit: int = typer.Option(20, "--limit", "-n", help="Number of sessions to show"),
    path: str = typer.Argument(None, help="Project root path (default: auto-detect)"),
) -> None:
    """List recent conversation sessions."""

    async def _list() -> None:
        from kraang.display import display_sessions
        from kraang.store import SQLiteStore

        root = Path(path) if path else find_project_root()
        db_path = resolve_db_path(root)

        if not db_path.exists():
            typer.echo(f"Database not found at {db_path}. Run 'kraang init' first.", err=True)
            raise typer.Exit(1)

        async with SQLiteStore(str(db_path)) as store:
            result = await store.list_sessions(limit=limit)
            display_sessions(result)

    _run(_list())


# ---------------------------------------------------------------------------
# kraang session <id>
# ---------------------------------------------------------------------------


@app.command()
def session(
    session_id: str = typer.Argument(..., help="Session ID or 8-char prefix"),
    max_turns: int = typer.Option(0, "--max-turns", "-n", help="Max turns to show (0 = all)"),
) -> None:
    """View a session transcript in detail."""

    async def _show() -> None:
        from kraang.config import encode_project_path
        from kraang.display import display_transcript
        from kraang.indexer import read_transcript
        from kraang.store import SQLiteStore

        root = find_project_root()
        db_path = resolve_db_path(root)

        if not db_path.exists():
            typer.echo(f"Database not found at {db_path}. Run 'kraang init' first.", err=True)
            raise typer.Exit(1)

        async with SQLiteStore(str(db_path)) as store:
            try:
                sess = await store.get_session(session_id)
            except ValueError as e:
                typer.echo(str(e), err=True)
                raise typer.Exit(1) from None
            if sess is None:
                typer.echo(f'Session "{session_id}" not found.', err=True)
                raise typer.Exit(1)

            # Find JSONL file
            encoded = encode_project_path(sess.project_path)
            sessions_dir = Path.home() / ".claude" / "projects" / encoded
            jsonl_path = sessions_dir / f"{sess.session_id}.jsonl"

            if not jsonl_path.exists():
                typer.echo(f"Transcript file not found: {jsonl_path}", err=True)
                raise typer.Exit(1)

            turns = read_transcript(jsonl_path)
            if max_turns > 0:
                turns = turns[:max_turns]
            display_transcript(sess, turns)

    _run(_show())


# ---------------------------------------------------------------------------
# kraang search
# ---------------------------------------------------------------------------


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, "--limit", "-n", help="Max results per type"),
) -> None:
    """Search notes and sessions."""

    async def _search() -> None:
        from kraang.display import display_search_results
        from kraang.hybrid import hybrid_search
        from kraang.search import build_fts_query
        from kraang.store import SQLiteStore

        root = find_project_root()
        db_path = resolve_db_path(root)

        if not db_path.exists():
            typer.echo(f"Database not found at {db_path}. Run 'kraang init' first.", err=True)
            raise typer.Exit(1)

        from kraang.embeddings import create_provider

        provider = await create_provider()

        async with SQLiteStore(str(db_path)) as store:
            note_results = await hybrid_search(store, provider, query, limit=limit)

            fts_expr = build_fts_query(query)
            session_results = []
            if fts_expr:
                session_results = await store.search_sessions(fts_expr, limit=limit)

            notes = [(r.note, r.score, r.snippet) for r in note_results]
            sessions = [(r.session, r.score, r.snippet) for r in session_results]

            display_search_results(query, notes, sessions)

    _run(_search())


# ---------------------------------------------------------------------------
# kraang notes
# ---------------------------------------------------------------------------


@app.command()
def notes(
    all_notes: bool = typer.Option(False, "--all", "-a", help="Include forgotten notes"),
    limit: int = typer.Option(50, "--limit", "-n", help="Max notes to show"),
) -> None:
    """List notes in the knowledge base."""

    async def _list() -> None:
        from kraang.display import display_notes
        from kraang.store import SQLiteStore

        root = find_project_root()
        db_path = resolve_db_path(root)

        if not db_path.exists():
            typer.echo(f"Database not found at {db_path}. Run 'kraang init' first.", err=True)
            raise typer.Exit(1)

        async with SQLiteStore(str(db_path)) as store:
            result = await store.list_notes(include_forgotten=all_notes, limit=limit)
            display_notes(result)

    _run(_list())


# ---------------------------------------------------------------------------
# kraang status
# ---------------------------------------------------------------------------


@app.command(name="status")
def status_cmd() -> None:
    """Show knowledge base health and statistics."""

    async def _status() -> None:
        from kraang.display import display_status
        from kraang.formatter import format_status
        from kraang.store import SQLiteStore

        root = find_project_root()
        db_path = resolve_db_path(root)

        if not db_path.exists():
            typer.echo(f"Database not found at {db_path}. Run 'kraang init' first.", err=True)
            raise typer.Exit(1)

        from kraang.embeddings import create_provider

        provider = await create_provider()

        if provider is not None:
            embedding_status = f"{provider.provider_id}/{provider.model} ({provider.dims} dims)"
        else:
            embedding_status = "disabled — set OPENAI_API_KEY to enable semantic search"

        async with SQLiteStore(str(db_path)) as store:
            active, forgotten = await store.count_notes()
            session_count = await store.count_sessions()
            last_indexed = await store.last_indexed_at()
            recent = await store.recent_notes(days=7, limit=10)
            categories = await store.category_counts()
            tags = await store.tag_counts()
            stale = await store.stale_notes(days=30)

            md = format_status(
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
            display_status(md)

    _run(_status())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    app()


if __name__ == "__main__":
    main()

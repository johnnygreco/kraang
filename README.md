<table>
  <tr>
    <td><img src="assets/kraang.jpeg" alt="Kraang" width="350"></td>
    <td><h1>Kraang</h1><b>Long-term memory for your agents.</b></td>
  </tr>
</table>

Kraang is an MCP (Model Context Protocol) server that gives AI assistants persistent memory and session indexing, backed by SQLite with FTS5 full-text search and optional hybrid semantic search. It stores knowledge notes, indexes conversation transcripts, and surfaces what matters via search.

## Why?

AI assistants forget everything between sessions. Kraang gives them persistent memory — decisions, debugging breakthroughs, patterns — so your next conversation picks up where the last one left off.

## Quick Start

The fastest way to get started is with `kraang init`:

```bash
uvx kraang init        # ephemeral — downloads on each run
uv tool install kraang # persistent — install once, use everywhere
kraang init
```

This creates a `.kraang/` directory, initializes the database, configures `.mcp.json`, adds `.kraang/` to `.gitignore`, sets up a `SessionEnd` hook for automatic session indexing, creates `.claude/rules/kraang.md` with agent usage guidelines, and indexes any existing sessions.

### Manual Configuration

Add to your MCP client configuration (e.g. Claude Code, Claude Desktop):

```json
{
  "mcpServers": {
    "kraang": {
      "command": "uvx",
      "args": ["kraang", "serve"],
      "env": { "KRAANG_DB_PATH": ".kraang/kraang.db" }
    }
  }
}
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `remember` | Save a note. Updates in place if the title exists, otherwise creates a new one. |
| `recall` | Search notes and conversation sessions. Scope to `"notes"`, `"sessions"`, or `"all"`. |
| `read_session` | Load a full conversation transcript by session ID (use `recall` to find sessions first). |
| `forget` | Downweight or hide a note by adjusting its relevance score (0.0 = hidden, 1.0 = full). |
| `context` | Auto-recall relevant memories and return safety-framed XML for context injection. |
| `status` | Get a knowledge base overview: note/session counts, recent activity, top tags. |

## CLI Commands

| Command | Description |
|---------|-------------|
| `kraang init [PATH]` | Set up kraang for the current project (database, .gitignore, config, hooks, rules, initial index). |
| `kraang serve` | Run the MCP server (invoked by Claude Code). |
| `kraang index [PATH] [--from-hook]` | Index or re-index conversation sessions for the project. |
| `kraang sessions [-n LIMIT] [PATH]` | List recent conversation sessions (default: 20). |
| `kraang session <id> [-n MAX_TURNS]` | View a session transcript in detail. |
| `kraang search <query> [-n LIMIT]` | Search notes and sessions (default: 10). |
| `kraang notes [-a] [-n LIMIT]` | List notes in the knowledge base (`-a` includes forgotten). |
| `kraang status` | Show knowledge base health and statistics. |

## Architecture

Kraang uses a layered architecture:

1. **Models** (`models.py`) -- Pydantic schemas for notes, sessions, and search results.
2. **Store** (`store.py`) -- SQLite backend with FTS5 full-text search, BM25 ranking, vector storage, and brute-force cosine fallback.
3. **Search** (`search.py`) -- Query parsing and FTS5 expression building.
4. **Hybrid** (`hybrid.py`) -- Hybrid vector + keyword search with weighted merge and FTS fallback.
5. **Embeddings** (`embeddings.py`) -- Embedding provider abstraction with OpenAI implementation, retry logic, and L2 normalization.
6. **Safety** (`safety.py`) -- Prompt injection detection and safe context formatting for LLM consumption.
7. **Indexer** (`indexer.py`) -- Reads Claude Code JSONL transcripts and indexes sessions.
8. **Server** (`server.py`) -- MCP server exposing 6 tools over stdio.
9. **CLI** (`cli.py`) -- Typer CLI for init, serve, index, and local queries.
10. **Formatter** (`formatter.py`) -- Markdown formatting for MCP tool responses.
11. **Display** (`display.py`) -- Rich console rendering for CLI commands.
12. **Config** (`config.py`) -- Project root detection, database path resolution, and title normalization.

## Semantic Search (optional)

Kraang supports hybrid semantic + keyword search for more intelligent recall. When enabled, queries like "that debugging trick from last week" work alongside exact keyword matches.

### Setup

Set your OpenAI API key to enable semantic search:

```bash
export OPENAI_API_KEY="sk-..."
kraang status  # verify: should show openai/text-embedding-3-small
```

When enabled, `remember` automatically embeds notes for semantic search. Existing notes can be re-saved to generate embeddings.

### How it works

- **Hybrid search**: Combines vector similarity (70%) with keyword matching (30%)
- **Graceful degradation**: No API key? Everything works with keyword search only
- **Embedding cache**: Content-addressed cache avoids redundant API calls
- **Safety framing**: The `context` tool wraps recalled memories with prompt injection protection

### Environment variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `OPENAI_API_KEY` | No | — | Enables semantic search via OpenAI embeddings |
| `KRAANG_DB_PATH` | No | `<project_root>/.kraang/kraang.db` | Override database location |

### Claude Code integration

When using `kraang init`, add `OPENAI_API_KEY` to the generated `.mcp.json` to enable semantic search in Claude Code:

```json
{
  "mcpServers": {
    "kraang": {
      "command": "uvx",
      "args": ["kraang", "serve"],
      "env": {
        "KRAANG_DB_PATH": ".kraang/kraang.db",
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

## Development

```bash
git clone https://github.com/johnnygreco/kraang.git && cd kraang
uv sync --extra dev
make install-hooks   # install pre-commit hooks (run once)
make test
make lint
```

Pre-commit hooks run automatically before each commit (ruff format, ruff check --fix, ty). Run manually:

```bash
uv run pre-commit run --all-files
```

Run the full check suite:

```bash
make coverage   # tests + coverage report
make format     # auto-format with ruff
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Kraang tools not showing up in Claude Code | Restart Claude Code after running `kraang init` |
| Sessions not being indexed automatically | Check that `.claude/settings.json` has the `SessionEnd` hook |
| Search returns nothing | Run `kraang status` to check counts, then `kraang index` to re-index |
| Need a fresh start | Delete `.kraang/` and re-run `kraang init` |

## License

Apache 2.0

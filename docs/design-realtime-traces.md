# Real-Time Traces for kraang

## Design Document

**Status:** Proposal
**Date:** 2025-02-21

---

## 1. Vision

kraang today is a notebook — the agent writes things down and looks them up.
With real-time traces, kraang becomes a **coach**. It watches the agent work,
recognizes when it's struggling, recalls how similar struggles were resolved,
and offers targeted advice at the moment it matters.

Session 1: the agent struggles with FTS5 triggers for 20 minutes.
Session 2: kraang says "last time you edited store.py, the FTS5 triggers needed
updating — check lines 52-117."
Session 3: the agent asks `check_insights(files=["store.py"])` before editing
and avoids the issue entirely.

By session 10, the project's common pitfalls are mapped and surfaced proactively.

This is not a logging tool. It is a learning system.

---

## 2. Architecture Overview

```
Claude Code writes JSONL
        |
        v
  ┌─────────────────────────────────────────────┐
  │           kraang watch (watcher)             │
  │                                              │
  │  File Tailer ──> JSONL Parser ──> Analyzer   │
  │       (poll/notify)    (incremental)    |    │
  │                                         v    │
  │                               Pattern Engine │
  │                               (6 detectors)  │
  └──────────────────┬──────────────────────────-┘
                     │ writes
                     v
              .kraang/kraang.db
              (same SQLite, WAL mode)
              ┌──────────────────┐
              │ trace_turns      │ ← NEW
              │ trace_tool_calls │ ← NEW
              │ trace_insights   │ ← NEW
              │ sessions         │ (existing)
              │ notes            │ (existing)
              └──────────────────┘
                     │ reads
                     v
  ┌──────────────────────────────────────────────┐
  │          MCP Server (existing + new tools)    │
  │                                               │
  │  remember, recall, forget, status             │
  │  + check_insights  ← NEW                      │
  └───────────────────────────────────────────────┘
```

**Key decisions:**

- **Same database.** Traces live alongside notes and sessions in
  `.kraang/kraang.db`. WAL mode with `busy_timeout=5000` already handles
  concurrent access. One DB means one connection pool, one lifecycle, one
  backup. Trace tables can be rebuilt from source JSONL at any time.

- **Python first, Rust later.** Ship the watcher in Python for speed of
  iteration. Port the hot path (file tailing + JSON parsing) to a standalone
  Rust binary when we know what works. The integration layer is SQLite either
  way — Rust writes, Python reads, no custom IPC.

- **Pull model for agent feedback.** MCP is request-response over stdio. There
  is no mechanism for the server to push messages into the conversation. The
  agent calls `check_insights` at natural decision points. Insights that prove
  durable get materialized as notes via the existing `remember` pathway.

---

## 3. What We Are Tracing

A coding agent session is a tree, not a pipeline. The right trace model has
exactly **three levels**:

| Level | What | Why |
|-------|------|-----|
| **Session** | Full session start to finish | Already exists as `sessions` table; we augment it with aggregate trace metrics |
| **Turn** | One user-prompt → assistant-response cycle | The atomic unit of "work attempted." This is the most important addition. |
| **Tool Call** | A single tool invocation within a turn | Where latency, errors, and file mutations actually happen |

We deliberately do NOT add a fourth "event" level for sub-tool-call things
(thinking blocks, API retries). The JSONL doesn't emit that granularity, and
three levels is enough for every diagnostic question we care about.

### What the JSONL gives us that we don't use today

The current `parse_jsonl` in `indexer.py` crushes the conversation tree into a
flat `Session` record — turn counts, concatenated text, tool list. That's fine
for search. It's completely wrong for diagnosis.

Fields the current indexer **ignores** that traces need:
- `uuid` / `parentUuid` — the causal chain between messages
- `requestId` — groups a user prompt with its assistant response
- Per-message `timestamp` — gives us turn-level and tool-level timing
- `tool_use.id` / `tool_result.tool_use_id` — links tool calls to their results
- `isSidechain` — subagent/parallel execution markers
- Error content in `tool_result` blocks

---

## 4. Data Model

### 4.1 Table: `trace_turns`

One row per user-prompt-to-response cycle. The core diagnostic unit.

```sql
CREATE TABLE IF NOT EXISTS trace_turns (
    turn_id          TEXT PRIMARY KEY,
    session_id       TEXT NOT NULL,
    turn_index       INTEGER NOT NULL,
    request_id       TEXT DEFAULT '',

    -- Timing
    started_at       TEXT NOT NULL,
    ended_at         TEXT NOT NULL,
    duration_ms      INTEGER NOT NULL DEFAULT 0,

    -- Content fingerprint (NOT full text — that stays in JSONL)
    user_prompt      TEXT NOT NULL DEFAULT '',
    assistant_summary TEXT NOT NULL DEFAULT '',

    -- Tool call aggregates
    tool_call_count  INTEGER NOT NULL DEFAULT 0,
    tool_error_count INTEGER NOT NULL DEFAULT 0,
    tools_used_json  TEXT NOT NULL DEFAULT '[]',

    -- File tracking
    files_read_json  TEXT NOT NULL DEFAULT '[]',
    files_written_json TEXT NOT NULL DEFAULT '[]',

    -- Anomaly flags (computed at ingestion)
    has_error        INTEGER NOT NULL DEFAULT 0,
    has_retry        INTEGER NOT NULL DEFAULT 0,
    is_sidechain     INTEGER NOT NULL DEFAULT 0,

    meta_json        TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX idx_tt_session ON trace_turns(session_id);
CREATE INDEX idx_tt_started ON trace_turns(started_at);
CREATE INDEX idx_tt_errors  ON trace_turns(has_error) WHERE has_error = 1;
```

### 4.2 Table: `trace_tool_calls`

One row per tool invocation. Where latency diagnosis lives.

```sql
CREATE TABLE IF NOT EXISTS trace_tool_calls (
    tool_call_id     TEXT PRIMARY KEY,
    turn_id          TEXT NOT NULL,
    session_id       TEXT NOT NULL,
    call_index       INTEGER NOT NULL,

    tool_name        TEXT NOT NULL,
    input_summary    TEXT NOT NULL DEFAULT '',
    file_path        TEXT DEFAULT '',

    started_at       TEXT NOT NULL,
    ended_at         TEXT DEFAULT '',
    duration_ms      INTEGER NOT NULL DEFAULT 0,

    is_error         INTEGER NOT NULL DEFAULT 0,
    error_text       TEXT DEFAULT '',
    result_size      INTEGER NOT NULL DEFAULT 0,

    meta_json        TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX idx_ttc_turn     ON trace_tool_calls(turn_id);
CREATE INDEX idx_ttc_session  ON trace_tool_calls(session_id);
CREATE INDEX idx_ttc_tool     ON trace_tool_calls(tool_name);
CREATE INDEX idx_ttc_error    ON trace_tool_calls(is_error) WHERE is_error = 1;
CREATE INDEX idx_ttc_file     ON trace_tool_calls(file_path) WHERE file_path != '';
CREATE INDEX idx_ttc_duration ON trace_tool_calls(duration_ms);
```

### 4.3 Table: `trace_insights`

Algorithmic observations from pattern detection, the bridge between raw traces
and agent intelligence.

```sql
CREATE TABLE IF NOT EXISTS trace_insights (
    insight_id       TEXT PRIMARY KEY,
    session_id       TEXT NOT NULL,
    pattern          TEXT NOT NULL,
    severity         TEXT NOT NULL DEFAULT 'info',
    scope            TEXT NOT NULL DEFAULT 'ephemeral',
    summary          TEXT NOT NULL,
    detail           TEXT NOT NULL DEFAULT '',
    suggestion       TEXT NOT NULL DEFAULT '',
    evidence_json    TEXT NOT NULL DEFAULT '[]',
    related_files_json TEXT NOT NULL DEFAULT '[]',
    fingerprint      TEXT NOT NULL DEFAULT '',
    created_at       TEXT NOT NULL,
    acknowledged     INTEGER NOT NULL DEFAULT 0,
    materialized_note_id TEXT DEFAULT NULL
);

CREATE INDEX idx_ti_session  ON trace_insights(session_id);
CREATE INDEX idx_ti_pattern  ON trace_insights(pattern);
CREATE INDEX idx_ti_severity ON trace_insights(severity);
CREATE INDEX idx_ti_fingerprint ON trace_insights(fingerprint);
```

### 4.4 Pydantic Models

```python
class TraceTurn(BaseModel):
    turn_id: str
    session_id: str
    turn_index: int
    request_id: str = ""
    started_at: datetime
    ended_at: datetime
    duration_ms: int = 0
    user_prompt: str = ""           # first 500 chars
    assistant_summary: str = ""     # first 500 chars
    tool_call_count: int = 0
    tool_error_count: int = 0
    tools_used: list[str] = []
    files_read: list[str] = []
    files_written: list[str] = []
    has_error: bool = False
    has_retry: bool = False
    is_sidechain: bool = False


class TraceToolCall(BaseModel):
    tool_call_id: str
    turn_id: str
    session_id: str
    call_index: int
    tool_name: str
    input_summary: str = ""
    file_path: str = ""
    started_at: datetime
    ended_at: datetime | None = None
    duration_ms: int = 0
    is_error: bool = False
    error_text: str = ""
    result_size: int = 0


class InsightSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class InsightScope(str, Enum):
    EPHEMERAL = "ephemeral"     # this session only
    DURABLE = "durable"         # persisted as note
    RECURRING = "recurring"     # seen across sessions


class TraceInsight(BaseModel):
    insight_id: str
    session_id: str
    pattern: str
    severity: InsightSeverity
    scope: InsightScope
    summary: str
    detail: str = ""
    suggestion: str = ""
    evidence: list[str] = []
    related_files: list[str] = []
    fingerprint: str = ""
    created_at: datetime
    acknowledged: bool = False
    materialized_note_id: str | None = None
```

---

## 5. The Watcher (`kraang watch`)

### 5.1 Architecture

The watcher is a new module `src/kraang/watcher.py` with three components:

**SessionTailer** — tails a single JSONL file, tracking byte offset, buffering
partial lines, parsing complete lines using the existing helpers from
`indexer.py`.

**TraceAnalyzer** — receives parsed entries and maintains incremental
`SessionState` per session. Reconstructs the turn/tool-call tree using `uuid`,
`parentUuid`, and `requestId` fields. Emits `TraceTurn` and `TraceToolCall`
objects.

**PatternEngine** — receives trace objects and runs six detectors (see
section 6). Emits `TraceInsight` objects.

### 5.2 File Watching Strategy

**Phase 1 (Python):** Simple polling. Stat each tracked JSONL file every
second. If `st_size` has grown, seek to stored offset, read new bytes, split on
`\n`, buffer any trailing partial line. This has zero new dependencies and is
perfectly adequate for 1-5 concurrent sessions.

**Phase 2 (Rust):** A standalone `kraang-trace` binary using the `notify` crate
for inotify/kqueue/FSEvents with a polling fallback. Communicates with Python
through the shared SQLite database — no IPC protocol. See section 9 for Rust
architecture.

### 5.3 Session Lifecycle Detection

- **New session:** A `.jsonl` file appears in `~/.claude/projects/{encoded-path}/`
  that wasn't present on last scan.
- **Active session:** File has been modified within the last 60 seconds.
- **Ended session:** No modification for 5 minutes. Perform a final full parse.
- **Pre-existing sessions:** When `kraang watch` starts, files modified within
  the last 10 minutes are treated as active. Tail from end (don't replay).

### 5.4 CLI Commands

```
kraang watch                    # Foreground dashboard (Rich Live)
kraang watch --daemon           # Background mode, log to .kraang/watch.log
kraang watch stop               # Stop background watcher
kraang watch status             # Is the watcher running?
kraang watch insights           # List recent insights (Rich table)
```

### 5.5 The Dashboard

When run in foreground, `kraang watch` shows a three-panel Rich Live display:

```
┌─[ kraang watch ]──── project: myapp ──── 14:32 ─────────────────┐
│                                                                   │
│  ACTIVE SESSIONS                                                  │
│  ───────────────────────────────────────────────────────────────── │
│  [*] 7a3f2b1c  feat/auth   "Add OAuth2 login"     12m active     │
│      Turn 14 | 8 tools | 3 files | Last: Edit src/auth/oauth.py  │
│                                                                   │
│  [*] 9e1d4c8a  main        "Fix CI pipeline"       3m active     │
│      Turn 4  | 2 tools | 0 files | Last: Bash pytest tests/ -v   │
│                                                                   │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  LIVE FEED                                                        │
│  ───────────────────────────────────────────────────────────────── │
│  14:32:18  7a3f..  Edit src/auth/oauth.py                         │
│  14:32:15  7a3f..  Read src/auth/config.py                        │
│  14:32:12  9e1d..  User: "Can you also fix the linting?"          │
│  14:32:08  7a3f..  Bash: rg "OAuth" src/                          │
│  14:31:55  7a3f..  User: "Now implement the callback handler"     │
│                                                                   │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  INSIGHTS                                                         │
│  ───────────────────────────────────────────────────────────────── │
│  [!] 7a3f.. edited src/auth/oauth.py 3 times in 8 min            │
│  [!] 9e1d.. pytest failed 3 times consecutively                   │
│                                                                   │
├───────────────────────────────────────────────────────────────────┤
│  q: quit    f: filter session    i: insights only                 │
└───────────────────────────────────────────────────────────────────┘
```

Color coding: green for user messages, blue for tool calls, yellow for warnings,
bold red for errors, dim for normal events. Most of the screen is calm; bright
spots pull your eye to what matters.

---

## 6. Pattern Detection Engine

Six detectors, each implementing a simple interface:

```python
class PatternDetector(Protocol):
    name: str
    def observe(self, entry: TraceEntry) -> TraceInsight | None: ...
    def summarize(self) -> list[TraceInsight]: ...
    def reset(self) -> None: ...
```

### 6.1 Retry Loops

**Rule:** Same tool name + substantially similar input appears 3+ times within
a sliding window of 10 assistant turns.

**Output:** "You've attempted `Edit src/auth.py` 3 times. Consider stepping
back to understand the error before retrying."

### 6.2 Circular Edits

**Rule:** A file is written, then written again to undo >60% of changes, then
written again. Same file appearing 4+ times in a window with cycling content.

**Output:** "Circular edit detected on `models.py`. You wrote, then reverted.
The approach may need rethinking."

### 6.3 Test-Fix Death Spirals

**Rule:** Sequence of (Bash test → error → Edit → Bash test → same error class)
repeated 2+ times. "Same error class" = matching test name or error type.

**Output:** "Test `test_login_flow` has failed 3 times with `AssertionError`.
Consider reading the test fixtures first."

### 6.4 Environment/Permission Failures

**Rule:** Bash returns errors matching `Permission denied`, `command not found`,
`ModuleNotFoundError`, `No such file or directory`.

**Output:** "Command failed with `ModuleNotFoundError: numpy`. This was also
seen in session `a3f8b2c1`."

### 6.5 Token Waste (Redundant Reads)

**Rule:** `Read` called on the same file 2+ times with no intervening
`Edit`/`Write` to that file.

**Output:** "`store.py` was read 3 times without edits. Consider noting key
sections to avoid re-reading."

### 6.6 Anti-Pattern Detection

**Rule:** Bash used with `cat`, `grep`, `head`, `tail`, `find`, `echo >` when
MCP tools (Read, Grep, Glob, Write) exist.

**Output:** "Bash `cat file.py` — the Read tool is more efficient."

### Cross-Session Learning

Each pattern gets a **fingerprint** — a hash of its essential characteristics
stripped of session-specific details:

- Retry loop on `pytest test_auth.py::test_login` → `retry:test_auth.py::test_login`
- Circular edit on `src/models.py` → `circular_edit:src/models.py`
- Environment error → `env:ModuleNotFoundError:numpy`

Fingerprints are tracked across sessions. When a pattern recurs 2+ times in
different sessions, it crosses the **materialization threshold** and gets
auto-created as a note via `upsert_note(category="trace-insight")`. This makes
it searchable via `recall` and visible in `status`.

**Insights start ephemeral and earn their way to permanence through
repetition.** One-off flukes don't pollute the knowledge base. Real patterns
get captured.

---

## 7. MCP Tool: `check_insights`

One new tool. Minimal surface area, maximum utility.

```python
@mcp.tool()
async def check_insights(
    context: str = "",
    files: list[str] | None = None,
    severity_min: str = "info",
    limit: int = 5,
) -> str:
    """Check for trace-derived insights and recommendations.

    Call at session start, before risky operations, or after errors.
    Returns actionable intelligence from current and past sessions.

    Args:
        context: What you're about to do (helps filter relevant insights).
        files: Files you're working with (surfaces file-specific patterns).
        severity_min: Minimum severity: "info", "warning", "critical".
        limit: Maximum insights to return.
    """
```

### Return format

```markdown
## Trace Insights

### Active (2 insights)

**WARNING: Retry loop detected**
`pytest tests/test_auth.py` failed 3 times with `AssertionError` in `test_login_flow`.
> Suggestion: Read the test fixture setup in conftest.py — this test depends
> on a mock database that may need resetting.
> Similar pattern resolved in session `a3f8b2c1` by fixing fixture teardown.

**INFO: Redundant file reads**
`src/kraang/store.py` read 3 times this session with no edits.
> Suggestion: Note key sections to avoid re-reading the full file.

### File History
- `auth.py`: Last 3 sessions editing this file also required updating `test_auth.py`
- `store.py`: Common gotcha — FTS5 triggers must stay in sync with schema changes
```

### Why this shape works

- **Zero-argument default is useful.** Returns unacknowledged insights from the
  current session plus critical cross-session warnings. The agent doesn't need
  to know what to ask for.

- **`context` enables pre-flight checks.** `check_insights(context="refactoring
  auth module", files=["src/auth.py"])` surfaces file-specific history.

- **Severity filtering prevents overload.** The agent can check only `"warning"`
  during fast-moving work, then do a full review at natural pause points.

### Avoiding information overload

Each insight gets a signal score:

```
signal = severity_weight * recency_decay * cross_session_boost * novelty
```

Where severity_weight: critical=3, warning=2, info=1. Recency decays over
hours. Cross-session recurrence amplifies. Already-acknowledged insights score
zero. Only the top `limit` insights are returned, capped at ~800 tokens.

Ephemeral insights auto-expire after 24 hours. Only materialized (durable)
insights survive as notes.

---

## 8. Integration with Existing kraang

### Status tool enhancement

The existing `status` tool output grows a "Trace Digest" section:

```markdown
### Trace Digest
**From last session:**
- WARNING: `test_login_flow` was failing when session ended
- INFO: 3 files edited without running tests

**Recurring patterns:**
- `store.py` edits require FTS5 trigger updates (seen in 4 sessions)
- `test_auth.py` fixtures need manual teardown (seen in 3 sessions)
```

### Rules file update

`.claude/rules/kraang.md` (generated by `kraang init`) gets a new section:

```markdown
## Trace Intelligence

Use `check_insights` at key moments:
- **Session start**: Call with no arguments to get the trace digest
- **Before editing**: Call with `files=["path/to/file.py"]` for file-specific patterns
- **After errors**: Call with `context="test failure"` for past resolutions
- **Before committing**: Call with `severity_min="warning"` for unresolved issues
```

### Indexer integration

`kraang index` gains trace extraction as a second pass. After `parse_jsonl`
produces a `Session`, a new `parse_trace` function extracts the turn/tool-call
tree from the same JSONL file and stores it. The watcher does this
incrementally in real-time; the indexer does it comprehensively at session end.

---

## 9. Rust Component (Phase 2)

### Why Rust, specifically

Python's `json.loads` on a 10MB JSONL file takes seconds; Rust's `serde_json`
handles it in milliseconds. But more importantly, Python cannot efficiently tail
files while serving MCP requests on the same event loop. A separate Rust process
solves this cleanly.

### Integration: standalone binary + shared SQLite

`kraang-trace` is a standalone Rust binary that watches JSONL files and writes
to the same `.kraang/kraang.db`. No PyO3, no custom IPC, no maturin. SQLite IS
the integration layer.

The Python MCP server reads what Rust writes. Two processes sharing a WAL-mode
SQLite database is exactly what WAL mode was designed for.

### Architecture

```
kraang-trace/
  Cargo.toml
  src/
    main.rs            -- arg parsing, signal handling, daemon setup
    watcher.rs         -- notify crate + polling fallback
    parser.rs          -- incremental JSONL line parsing (serde_json)
    session_state.rs   -- running session accumulator
    pattern.rs         -- sliding window pattern detectors
    db.rs              -- rusqlite writes (same schema as Python)
    types.rs           -- shared structs
```

### Key dependencies

```toml
[dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1"
chrono = { version = "0.4", features = ["serde"] }
notify = "7"
rusqlite = { version = "0.32", features = ["bundled"] }
clap = { version = "4", features = ["derive"] }
indexmap = "2"
log = "0.4"
env_logger = "0.11"
```

### No async runtime

This is a file watcher and database writer. No network I/O. Two threads:
- **Main thread:** `notify` event loop + periodic stat polling
- **Processing thread:** Read bytes, parse JSON, detect patterns, write to SQLite

Communication via `std::sync::mpsc`. No tokio, no async. ~930 lines of Rust
total.

### Distribution

Phase 2a: GitHub Releases with platform binaries. `kraang watch` auto-downloads
the correct binary on first use (like `ruff` does).

Phase 2b: PyPI binary package (`kraang-trace`) with platform-specific wheels.
`pip install kraang-trace` makes it available. `kraang watch` detects and
delegates to it. Falls back to Python polling if absent.

---

## 10. Implementation Plan

### Phase 1: Foundation (Python-only, shippable)

**1a. Data models and schema**
- Add `TraceTurn`, `TraceToolCall`, `TraceInsight` to `models.py`
- Add trace tables to `_SCHEMA` in `store.py`
- Add CRUD methods to `SQLiteStore`
- Tests

**1b. Trace extraction in indexer**
- Add `parse_trace()` to `indexer.py` — extracts turn/tool-call tree from JSONL
- Integrate into `index_sessions()` as a second pass
- Tests with extended JSONL fixtures

**1c. Pattern detection engine**
- New module `trace.py` with `TraceAnalyzer` and 6 detectors
- Both real-time mode (entry-at-a-time) and post-hoc mode (full file replay)
- Tests for each detector

**1d. MCP tool**
- Add `check_insights` to `server.py`
- Add formatting to `formatter.py`
- Enhance `status` with trace digest
- Update rules template
- Tests

**1e. Watcher and CLI**
- New module `watcher.py` with `SessionTailer`, polling, orchestration
- New module `watch_display.py` with Rich Live dashboard
- Add `kraang watch` command group to `cli.py`
- Daemon mode with PID file
- Tests

### Phase 2: Rust Accelerator

**2a. Core binary**
- Crate setup, types, parser (port `parse_jsonl` logic)
- File watcher with notify + polling fallback
- SQLite writer matching Python schema exactly

**2b. Pattern detection in Rust**
- Port the 6 detectors to Rust
- Write to `trace_insights` table

**2c. Integration**
- `kraang watch` detects and delegates to `kraang-trace` binary
- Auto-download from GitHub Releases
- Fallback to Python watcher

### Phase 3: Compound Intelligence

**3a. Cross-session learning**
- Fingerprint-based pattern tracking across sessions
- Materialization threshold (2+ cross-session occurrences → note)
- Auto-creation of `category="trace-insight"` notes

**3b. Insight refinement**
- Signal scoring algorithm
- Context budget compression (~800 token cap)
- Acknowledgment tracking and auto-expiry

---

## 11. Key Design Decisions

### Why not OpenTelemetry?

OTel is designed for distributed systems with spans crossing service boundaries.
A coding agent session is a single-process conversation. OTel would bring heavy
dependencies and concepts (trace IDs, span contexts, exporters) without value.
Our three-level model (session/turn/tool_call) is semantically cleaner and
specific to the problem.

### Why store truncated text instead of full content?

The JSONL source files are always available. Traces should be lean and
query-fast. 500-char summaries let you identify WHICH turn/tool to investigate;
then `read_session` (already existing) gives full detail.

### Why anomaly flags at ingestion time?

Pattern detection (circular edits, retry loops) requires scanning across
multiple rows. Doing this once at ingestion and storing boolean flags means
queries stay simple and indexed. The cost is baked-in logic, but these patterns
are stable and well-understood.

### Why same database, not separate?

One SQLiteStore, one connection pool, one lifecycle. Trace tables add perhaps
100-500 rows per session. With WAL mode and busy timeout, this is well within
SQLite's comfort zone for a project-scoped database. If it grows large later,
splitting is trivial.

### Why Python first?

Polling-based file tailing in Python is ~200 lines and has zero new
dependencies. It's fast enough for 1-5 concurrent sessions. Shipping Python
first lets us iterate on the trace model, pattern detectors, and UX before
committing to a Rust binary. The Rust component is an optimization, not a
prerequisite.

### Why one new MCP tool, not three?

`check_insights` handles everything: session-start digest (zero args), pre-edit
checks (`files=["..."]`), post-error analysis (`context="test failure"`), and
commit readiness (`severity_min="warning"`). Multiple tools would fragment the
interface and confuse the agent about when to call what.

---

## 12. Queries the Model Enables

```sql
-- "What went wrong in this session?"
SELECT * FROM trace_turns
WHERE session_id = ? AND has_error = 1
ORDER BY turn_index;

-- "What's slow?"
SELECT tool_name, AVG(duration_ms), MAX(duration_ms), COUNT(*)
FROM trace_tool_calls WHERE session_id = ?
GROUP BY tool_name ORDER BY AVG(duration_ms) DESC;

-- "Show me the retry loops"
SELECT t.user_prompt, tc.tool_name, tc.file_path, tc.is_error
FROM trace_turns t JOIN trace_tool_calls tc ON tc.turn_id = t.turn_id
WHERE t.session_id = ? AND t.has_retry = 1
ORDER BY t.turn_index, tc.call_index;

-- "Which files cause the most trouble?"
SELECT file_path, COUNT(*) as errors
FROM trace_tool_calls WHERE is_error = 1 AND file_path != ''
GROUP BY file_path ORDER BY errors DESC LIMIT 10;

-- "Recurring patterns for this project"
SELECT pattern, summary, COUNT(*) as occurrences
FROM trace_insights WHERE scope = 'recurring'
GROUP BY fingerprint ORDER BY occurrences DESC;
```

---

## 13. Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `src/kraang/models.py` | Modify | Add TraceTurn, TraceToolCall, TraceInsight, InsightSeverity, InsightScope |
| `src/kraang/store.py` | Modify | Add trace tables to schema, add CRUD/query methods |
| `src/kraang/indexer.py` | Modify | Add `parse_trace()`, integrate into `index_sessions()` |
| `src/kraang/trace.py` | Create | TraceAnalyzer + 6 pattern detectors |
| `src/kraang/watcher.py` | Create | SessionTailer, file polling, orchestration |
| `src/kraang/watch_display.py` | Create | Rich Live dashboard for `kraang watch` |
| `src/kraang/server.py` | Modify | Add `check_insights` MCP tool |
| `src/kraang/formatter.py` | Modify | Add insight formatting, trace digest in status |
| `src/kraang/cli.py` | Modify | Add `kraang watch` command group |
| `tests/test_trace.py` | Create | Tests for pattern detectors |
| `tests/test_watcher.py` | Create | Tests for file tailing and orchestration |
| `kraang-trace/` | Create (Phase 2) | Rust crate for high-performance watcher |

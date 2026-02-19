# OpenClaw Deep-Dive: Session Management, Sync Mechanisms & QMD Backend

## Table of Contents
1. [Session Management](#1-session-management)
2. [Session Sync Pipeline](#2-session-sync-pipeline)
3. [Memory File Sync](#3-memory-file-sync)
4. [Sync Infrastructure](#4-sync-infrastructure)
5. [QMD Backend](#5-qmd-backend)
6. [Search Manager Factory & Fallback](#6-search-manager-factory--fallback)
7. [Backend Configuration](#7-backend-configuration)
8. [Memory Configuration Schema](#8-memory-configuration-schema)
9. [Agent Workspace Scoping](#9-agent-workspace-scoping)
10. [Key Architectural Patterns](#10-key-architectural-patterns)

---

## 1. Session Management

### 1.1 Session Storage Architecture

Sessions are stored as a **JSON dictionary** (`sessions.json`) mapping session keys to `SessionEntry` objects. The store lives at `<stateDir>/agents/<agentId>/sessions/sessions.json`.

**File:** `src/config/sessions/store.ts`

```typescript
// Session store is a flat JSON file: Record<string, SessionEntry>
export function loadSessionStore(
  storePath: string,
  opts: LoadSessionStoreOptions = {},
): Record<string, SessionEntry> {
  // Check cache first (TTL: 45s by default)
  if (!opts.skipCache && isSessionStoreCacheEnabled()) {
    const cached = SESSION_STORE_CACHE.get(storePath);
    if (cached && isSessionStoreCacheValid(cached)) {
      const currentMtimeMs = getFileMtimeMs(storePath);
      if (currentMtimeMs === cached.mtimeMs) {
        return structuredClone(cached.store);
      }
    }
  }
  // ... reads JSON from disk with retry (Windows-safe)
}
```

Key design decisions:
- **In-memory cache with TTL** (45s default) to avoid re-reading from disk
- **mtime-based invalidation**: cache is invalidated when file modification time changes
- **Atomic writes via temp-file + rename** on all platforms (with special Windows retry logic)
- **Write lock queue**: serialized writes per store path using `acquireSessionWriteLock()`

### 1.2 Session Entry Structure

**File:** `src/config/sessions/types.ts`

```typescript
export type SessionEntry = {
  sessionId: string;
  updatedAt: number;
  sessionFile?: string;           // Path to JSONL transcript
  spawnedBy?: string;             // Parent session (for subagents)
  spawnDepth?: number;            // 0=main, 1=sub, 2=sub-sub
  chatType?: SessionChatType;     // 'direct' | 'group' | 'channel'
  providerOverride?: string;      // Model provider override
  modelOverride?: string;         // Model override
  inputTokens?: number;
  outputTokens?: number;
  totalTokens?: number;
  compactionCount?: number;       // Number of context compactions
  memoryFlushAt?: number;         // Timestamp of last memory flush
  label?: string;
  channel?: string;               // Source channel (slack, discord, etc.)
  deliveryContext?: DeliveryContext;
  // ... many more fields
};
```

### 1.3 Session Key System

**File:** `src/sessions/session-key-utils.ts`

Session keys are hierarchical, colon-delimited identifiers:

```
agent:<agentId>:<rest>
agent:<agentId>:cron:<cronName>:run:<runId>
agent:<agentId>:subagent:<parentKey>
acp:<...>
<channel>:<chatType>:<id>
```

Key functions:
- `parseAgentSessionKey()` - Extracts `agentId` and `rest` from `agent:X:Y` keys
- `isCronRunSessionKey()` - Matches `cron:*:run:*` patterns
- `isSubagentSessionKey()` - Detects subagent session chains
- `getSubagentDepth()` - Counts `:subagent:` segments in key hierarchy
- `resolveThreadParentSessionKey()` - Strips `:thread:` or `:topic:` suffix to get parent

### 1.4 Session Lifecycle

Sessions flow through this lifecycle:

1. **Creation**: A `SessionEntry` is created in `sessions.json` with a fresh `sessionId`
2. **Transcript initialization**: A JSONL file is created at the resolved session path
3. **Message appending**: User/assistant messages are appended as JSONL lines
4. **Metadata updates**: `updateLastRoute()` / `recordSessionMetaFromInbound()` update delivery info
5. **Memory indexing**: Session JSONL files are indexed into the memory search system
6. **Maintenance**: Stale entries pruned (30 day default), capped (500 entries), rotated (10MB)

### 1.5 Session Transcripts

**File:** `src/config/sessions/transcript.ts`

Transcripts are JSONL files. Each session file starts with a header line:

```json
{"type":"session","version":"...","id":"<sessionId>","timestamp":"...","cwd":"..."}
```

Messages are appended as JSONL records:

```typescript
export async function appendAssistantMessageToSessionTranscript(params) {
  // 1. Load session store, find entry by sessionKey
  // 2. Ensure session file header exists
  // 3. Open SessionManager, append message
  // 4. Emit transcript update event
  emitSessionTranscriptUpdate(sessionFile);
}
```

### 1.6 Session Transcript Events

**File:** `src/sessions/transcript-events.ts`

A simple pub-sub mechanism that bridges transcript writes to memory sync:

```typescript
const SESSION_TRANSCRIPT_LISTENERS = new Set<SessionTranscriptListener>();

export function onSessionTranscriptUpdate(listener: SessionTranscriptListener): () => void {
  SESSION_TRANSCRIPT_LISTENERS.add(listener);
  return () => SESSION_TRANSCRIPT_LISTENERS.delete(listener);
}

export function emitSessionTranscriptUpdate(sessionFile: string): void {
  const update = { sessionFile: trimmed };
  for (const listener of SESSION_TRANSCRIPT_LISTENERS) {
    listener(update);
  }
}
```

This is the **critical bridge** between session writes and memory indexing. When a transcript is written, the event fires, and the memory manager's session listener picks it up.

### 1.7 Session Maintenance

**File:** `src/config/sessions/store.ts`

The session store has built-in maintenance:

| Feature | Default | Config Path |
|---------|---------|-------------|
| Pruning stale entries | 30 days | `session.maintenance.pruneAfter` |
| Max entries cap | 500 | `session.maintenance.maxEntries` |
| File rotation | 10 MB | `session.maintenance.rotateBytes` |
| Maintenance mode | `"warn"` | `session.maintenance.mode` |

In `"warn"` mode, it only logs when the active session would be evicted. In enforcement mode, it prunes/caps/rotates on every save.

### 1.8 Send Policy

**File:** `src/sessions/send-policy.ts`

Controls which sessions are allowed to send messages. Uses a rule-based matching system:

```typescript
export function resolveSendPolicy(params): SessionSendPolicyDecision {
  // 1. Check per-session override
  // 2. Evaluate policy rules (match by channel, chatType, keyPrefix, rawKeyPrefix)
  // 3. Fall back to default ("allow")
}
```

### 1.9 Input Provenance

**File:** `src/sessions/input-provenance.ts`

Tracks where user input came from:

```typescript
export type InputProvenanceKind = "external_user" | "inter_session" | "internal_system";

export type InputProvenance = {
  kind: InputProvenanceKind;
  sourceSessionKey?: string;
  sourceChannel?: string;
  sourceTool?: string;
};
```

This is used to differentiate between human input, cross-session messages, and system-generated input.

### 1.10 Session File Paths

**File:** `src/config/sessions/paths.ts`

Session transcripts are stored at:
```
<stateDir>/agents/<agentId>/sessions/<sessionId>.jsonl
<stateDir>/agents/<agentId>/sessions/<sessionId>-topic-<topicId>.jsonl
```

The path resolution handles:
- Legacy absolute paths (migrated to relative)
- Cross-agent session references via sibling directories
- Session ID validation (alphanumeric + dots/dashes, max 128 chars)

---

## 2. Session Sync Pipeline

### 2.1 Session File Parsing

**File:** `src/memory/session-files.ts`

JSONL session files are parsed into searchable text:

```typescript
export type SessionFileEntry = {
  path: string;        // Relative path: "sessions/<filename>"
  absPath: string;     // Absolute filesystem path
  mtimeMs: number;     // File modification time
  size: number;        // File size in bytes
  hash: string;        // SHA-256 of content + lineMap
  content: string;     // Flattened text: "User: ...\nAssistant: ..."
  lineMap: number[];   // Maps content lines → original JSONL line numbers
};
```

The extraction process:

```typescript
export async function buildSessionEntry(absPath: string): Promise<SessionFileEntry | null> {
  const raw = await fs.readFile(absPath, "utf-8");
  const lines = raw.split("\n");
  for (let jsonlIdx = 0; jsonlIdx < lines.length; jsonlIdx++) {
    // Parse each JSONL line
    // Filter: only type === "message", role === "user" | "assistant"
    // Extract text content (string or text blocks)
    // Normalize whitespace
    // Redact sensitive text
    // Format as "User: ..." or "Assistant: ..."
    // Build lineMap for citation mapping
  }
  return { path, absPath, mtimeMs, size, hash, content, lineMap };
}
```

Key details:
- Only `user` and `assistant` messages are extracted (system messages ignored)
- Text blocks (`{type: "text", text: "..."}`) are concatenated
- Sensitive text is redacted using `redactSensitiveText(text, { mode: "tools" })`
- A `lineMap` maps flattened content lines back to original JSONL line numbers (for citations)
- The `hash` includes both content AND lineMap for change detection

### 2.2 Listing Session Files

```typescript
export async function listSessionFilesForAgent(agentId: string): Promise<string[]> {
  const dir = resolveSessionTranscriptsDirForAgent(agentId);
  const entries = await fs.readdir(dir, { withFileTypes: true });
  return entries
    .filter((entry) => entry.isFile())
    .filter((name) => name.endsWith(".jsonl"))
    .map((name) => path.join(dir, name));
}
```

### 2.3 Session Sync Orchestration

**File:** `src/memory/sync-session-files.ts`

```typescript
export async function syncSessionFiles(params: {
  agentId: string;
  db: DatabaseSync;
  needsFullReindex: boolean;
  progress?: SyncProgressState;
  batchEnabled: boolean;
  concurrency: number;
  dirtyFiles: Set<string>;
  // ...
}) {
  const files = await listSessionFilesForAgent(params.agentId);
  const activePaths = new Set(files.map((file) => sessionPathForFile(file)));
  const indexAll = params.needsFullReindex || params.dirtyFiles.size === 0;

  // Build tasks: for each file, check if dirty, build entry, index if changed
  const tasks = files.map((absPath) => async () => {
    if (!indexAll && !params.dirtyFiles.has(absPath)) {
      return; // Skip unchanged files
    }
    const entry = await buildSessionEntry(absPath);
    if (!entry) return;
    await indexFileEntryIfChanged({
      db, source: "sessions", needsFullReindex, entry, indexFile
    });
  });

  await params.runWithConcurrency(tasks, params.concurrency);

  // Remove stale paths from the index
  deleteStaleIndexedPaths({
    db, source: "sessions", activePaths, vectorTable, ftsTable, ...
  });
}
```

### 2.4 Session Delta Tracking

**File:** `src/memory/manager-sync-ops.ts`

The manager uses delta tracking to avoid unnecessary re-indexing:

```typescript
// Listener for transcript update events
protected ensureSessionListener() {
  this.sessionUnsubscribe = onSessionTranscriptUpdate((update) => {
    if (!this.isSessionFileForAgent(sessionFile)) return;
    this.scheduleSessionDirty(sessionFile);
  });
}

// Debounce: 5 seconds after last event
private scheduleSessionDirty(sessionFile: string) {
  this.sessionPendingFiles.add(sessionFile);
  if (this.sessionWatchTimer) return;
  this.sessionWatchTimer = setTimeout(() => {
    void this.processSessionDeltaBatch();
  }, 5000); // SESSION_DIRTY_DEBOUNCE_MS
}
```

Delta thresholds control when re-indexing happens:

```typescript
// Configurable thresholds (defaults):
deltaBytes: 100_000,     // 100KB of new content
deltaMessages: 50,       // 50 new JSONL lines

// The system tracks:
// - pendingBytes: cumulative bytes added since last sync
// - pendingMessages: cumulative new lines since last sync
// Re-index triggers when EITHER threshold is hit
```

---

## 3. Memory File Sync

### 3.1 Memory File Discovery

**File:** `src/memory/internal.ts`

Memory files are discovered from:
1. `MEMORY.md` in workspace root
2. `memory.md` in workspace root (alternative)
3. `memory/**/*.md` directory tree
4. Extra paths from config (`memorySearch.extraPaths`)

```typescript
export async function listMemoryFiles(
  workspaceDir: string,
  extraPaths?: string[],
): Promise<string[]> {
  // 1. Check MEMORY.md, memory.md (root files)
  // 2. Walk memory/ directory for *.md files
  // 3. Walk extra paths (directories → recursive, files → add directly)
  // 4. Deduplicate by realpath
  // Symlinks are SKIPPED throughout
}
```

### 3.2 Memory File Indexing

**File:** `src/memory/sync-memory-files.ts`

```typescript
export async function syncMemoryFiles(params) {
  const files = await listMemoryFiles(params.workspaceDir, params.extraPaths);
  const fileEntries = await Promise.all(
    files.map(async (file) => buildFileEntry(file, params.workspaceDir)),
  );
  const activePaths = new Set(fileEntries.map((entry) => entry.path));

  // Index each file (skip if hash unchanged)
  const tasks = fileEntries.map((entry) => async () => {
    await indexFileEntryIfChanged({
      db, source: "memory", needsFullReindex, entry, indexFile
    });
  });
  await params.runWithConcurrency(tasks, params.concurrency);

  // Remove stale entries
  deleteStaleIndexedPaths({ db, source: "memory", activePaths, ... });
}
```

### 3.3 File Watch System

**File:** `src/memory/manager-sync-ops.ts`

Memory files are watched via **chokidar**:

```typescript
protected ensureWatcher() {
  const watchPaths = new Set([
    path.join(this.workspaceDir, "MEMORY.md"),
    path.join(this.workspaceDir, "memory.md"),
    path.join(this.workspaceDir, "memory", "**", "*.md"),
    // + extra paths from config
  ]);

  this.watcher = chokidar.watch(Array.from(watchPaths), {
    ignoreInitial: true,
    ignored: shouldIgnoreMemoryWatchPath, // Skips .git, node_modules, etc.
    awaitWriteFinish: {
      stabilityThreshold: this.settings.sync.watchDebounceMs, // 1500ms default
      pollInterval: 100,
    },
  });

  const markDirty = () => {
    this.dirty = true;
    this.scheduleWatchSync();
  };
  this.watcher.on("add", markDirty);
  this.watcher.on("change", markDirty);
  this.watcher.on("unlink", markDirty);
}
```

Ignored directories: `.git`, `node_modules`, `.pnpm-store`, `.venv`, `venv`, `.tox`, `__pycache__`

---

## 4. Sync Infrastructure

### 4.1 Index Change Detection

**File:** `src/memory/sync-index.ts`

```typescript
export async function indexFileEntryIfChanged<TEntry extends { path: string; hash: string }>(
  params
): Promise<void> {
  const record = params.db
    .prepare(`SELECT hash FROM files WHERE path = ? AND source = ?`)
    .get(params.entry.path, params.source);

  if (!params.needsFullReindex && record?.hash === params.entry.hash) {
    return; // Hash unchanged, skip
  }

  await params.indexFile(params.entry);
}
```

### 4.2 Stale Entry Cleanup

**File:** `src/memory/sync-stale.ts`

After indexing, any paths in the DB that no longer exist on disk are cleaned up:

```typescript
export function deleteStaleIndexedPaths(params) {
  const staleRows = params.db
    .prepare(`SELECT path FROM files WHERE source = ?`)
    .all(params.source);

  for (const stale of staleRows) {
    if (params.activePaths.has(stale.path)) continue;
    // Delete from: files, vector table, chunks, FTS table
    params.db.prepare(`DELETE FROM files WHERE path = ? AND source = ?`).run(...);
    params.db.prepare(`DELETE FROM ${vectorTable} WHERE id IN (SELECT id FROM chunks WHERE ...)`).run(...);
    params.db.prepare(`DELETE FROM chunks WHERE path = ? AND source = ?`).run(...);
    // FTS cleanup if enabled
  }
}
```

### 4.3 Sync Progress Tracking

**File:** `src/memory/sync-progress.ts`

```typescript
export type SyncProgressState = {
  completed: number;
  total: number;
  label?: string;
  report: (update) => void;
};

export function bumpSyncProgressTotal(progress, delta, label?) {
  progress.total += delta;
  progress.report({ completed: progress.completed, total: progress.total, label });
}

export function bumpSyncProgressCompleted(progress, delta = 1, label?) {
  progress.completed += delta;
  progress.report({ completed: progress.completed, total: progress.total, label });
}
```

### 4.4 Full Sync Orchestration

**File:** `src/memory/manager-sync-ops.ts` → `runSync()`

The full sync flow:

```
1. Load vector extension (sqlite-vec)
2. Read metadata (model, provider, chunk settings)
3. Determine if full reindex needed:
   - Force flag
   - No existing meta
   - Model/provider/chunk settings changed
   - Vector newly available
4. If full reindex:
   a. Create temp database
   b. Seed embedding cache from old DB
   c. Sync memory files (full)
   d. Sync session files (full)
   e. Write meta
   f. Atomic swap: temp DB → target DB
5. If incremental:
   a. Sync dirty memory files
   b. Sync dirty session files (if delta thresholds met)
6. On embedding error: activate fallback provider, retry full reindex
```

### 4.5 Sync Triggers

| Trigger | Memory Files | Session Files | Notes |
|---------|:---:|:---:|-------|
| `session-start` | Yes | **No** | Only syncs memory, not sessions |
| `watch` | Yes | **No** | Chokidar file change |
| `search` | If dirty | If dirty | On-demand before search |
| `session-delta` | No | Yes | Delta threshold hit |
| `interval` | Yes | Yes | Timer-based (configurable minutes) |
| `manual`/force | Yes | Yes | Explicit `sync()` call |

---

## 5. QMD Backend

### 5.1 What is QMD?

QMD is an **external CLI tool** (`qmd`) that provides document indexing and semantic search. OpenClaw integrates it as an alternative memory backend via the `QmdMemoryManager` class. QMD stores its own index in an SQLite database and supports commands like:

- `qmd collection add <path> --name <name> --mask <pattern>` - Register a file collection
- `qmd collection list --json` - List existing collections
- `qmd update` - Re-scan all collections and update the index
- `qmd embed` - Generate/refresh embeddings
- `qmd query <text> --json -n <limit>` - Semantic search
- `qmd search <text> --json -n <limit>` - Keyword search
- `qmd vsearch <text> --json -n <limit>` - Vector search

### 5.2 QMD Manager Architecture

**File:** `src/memory/qmd-manager.ts`

```typescript
export class QmdMemoryManager implements MemorySearchManager {
  // Per-agent state isolation:
  private readonly qmdDir: string;          // <stateDir>/agents/<agentId>/qmd
  private readonly xdgConfigHome: string;   // <qmdDir>/xdg-config
  private readonly xdgCacheHome: string;    // <qmdDir>/xdg-cache
  private readonly indexPath: string;       // <xdgCacheHome>/qmd/index.sqlite

  // Collection tracking:
  private readonly collectionRoots = new Map<string, CollectionRoot>();
  private readonly sources = new Set<MemorySource>();

  // Session export state:
  private readonly exportedSessionState = new Map<string, { hash; mtimeMs; target }>();

  // Update scheduling:
  private updateTimer: NodeJS.Timeout | null = null;
  private pendingUpdate: Promise<void> | null = null;
}
```

### 5.3 QMD Initialization

```typescript
private async initialize(mode: QmdManagerMode): Promise<void> {
  this.bootstrapCollections();
  if (mode === "status") return; // Skip setup for status queries

  // Create XDG directories
  await fs.mkdir(this.xdgConfigHome, { recursive: true });
  await fs.mkdir(this.xdgCacheHome, { recursive: true });

  // Symlink shared ML models to avoid re-downloading per agent
  await this.symlinkSharedModels();

  // Create/verify QMD collections
  await this.ensureCollections();

  // Boot sync (optionally wait for completion)
  if (this.qmd.update.onBoot) {
    const bootRun = this.runUpdate("boot", true);
    if (this.qmd.update.waitForBootSync) {
      await bootRun;
    }
  }

  // Start periodic update timer
  if (this.qmd.update.intervalMs > 0) {
    this.updateTimer = setInterval(() => {
      void this.runUpdate("interval");
    }, this.qmd.update.intervalMs);
  }
}
```

### 5.4 QMD Collection Management

Collections are scoped per-agent with sanitized names:

```typescript
// Default collections (when includeDefaultMemory=true):
// memory-root-<agentId>  → <workspace>/MEMORY.md
// memory-alt-<agentId>   → <workspace>/memory.md
// memory-dir-<agentId>   → <workspace>/memory/**/*.md

// Custom collections from config:
// custom-1-<agentId>     → user-configured paths

// Session collection (when sessions.enabled=true):
// sessions-<agentId>     → <qmdDir>/sessions/**/*.md
```

Collections are idempotently registered via `qmd collection add`. If a collection's path or pattern changed, it's removed and re-added.

### 5.5 QMD Session Export

When QMD session indexing is enabled, sessions are **exported as Markdown files** for QMD to index:

```typescript
private async exportSessions(): Promise<void> {
  const files = await listSessionFilesForAgent(this.agentId);
  for (const sessionFile of files) {
    const entry = await buildSessionEntry(sessionFile);
    if (cutoff && entry.mtimeMs < cutoff) continue; // Retention check

    const target = path.join(exportDir, `${basename}.md`);
    // Write only if hash/mtime changed
    if (!state || state.hash !== entry.hash || state.mtimeMs !== entry.mtimeMs) {
      await fs.writeFile(target, this.renderSessionMarkdown(entry), "utf-8");
    }
  }
  // Clean up .md files for deleted sessions
}

private renderSessionMarkdown(entry: SessionFileEntry): string {
  return `# Session ${basename}\n\n${entry.content}\n`;
}
```

Key design: QMD can't read JSONL directly, so sessions are **converted to Markdown** and placed in a separate directory that QMD indexes as a collection.

### 5.6 QMD Update Cycle

```typescript
private async runUpdate(reason: string, force?: boolean): Promise<void> {
  // 1. Export sessions to Markdown (if enabled)
  if (this.sessionExporter) {
    await this.exportSessions();
  }

  // 2. Run `qmd update` to scan collections
  await this.runQmd(["update"], { timeoutMs: this.qmd.update.updateTimeoutMs });

  // 3. Run `qmd embed` (if due based on embedIntervalMs)
  const shouldEmbed = force || this.lastEmbedAt === null ||
    (Date.now() - this.lastEmbedAt > embedIntervalMs);
  if (shouldEmbed) {
    await this.runQmd(["embed"], { timeoutMs: this.qmd.update.embedTimeoutMs });
    this.lastEmbedAt = Date.now();
  }

  this.lastUpdateAt = Date.now();
  this.docPathCache.clear();
}
```

Update debouncing: `debounceMs` (15s default) prevents too-frequent updates.

### 5.7 QMD Search

```typescript
async search(query, opts?) {
  // 1. Check scope (deny non-DM sessions by default)
  if (!this.isScopeAllowed(opts?.sessionKey)) return [];

  // 2. Wait briefly for pending update (500ms max)
  await this.waitForPendingUpdateBeforeSearch();

  // 3. Execute search based on searchMode
  // "search" (default) → `qmd search <query> --json -n <limit> -c <collections>`
  // "vsearch"          → `qmd vsearch <query> --json -n <limit> -c <collections>`
  // "query"            → `qmd query <query> --json -n <limit>` (per-collection for multi)

  // 4. Parse JSON results
  const parsed = parseQmdQueryJson(result.stdout, result.stderr);

  // 5. Resolve document locations (docid → file path + source)
  // 6. Extract snippet lines, apply minScore filter
  // 7. Clamp results by maxInjectedChars budget
  return this.clampResultsByInjectedChars(results.slice(0, limit));
}
```

### 5.8 QMD Query Parser

**File:** `src/memory/qmd-query-parser.ts`

Handles parsing QMD's JSON output with robustness:

```typescript
export function parseQmdQueryJson(stdout: string, stderr: string): QmdQueryResult[] {
  // Handle "no results found" markers in stdout/stderr
  // Try direct JSON.parse
  // Fall back to extracting first JSON array from noisy output
  // Throw on invalid JSON
}

export type QmdQueryResult = {
  docid?: string;    // Document hash/ID
  score?: number;    // Relevance score
  file?: string;     // File path
  snippet?: string;  // Matching text excerpt
  body?: string;     // Full document body
};
```

### 5.9 QMD Scope Control

**File:** `src/memory/qmd-scope.ts`

QMD search is **restricted by default** to direct messages only:

```typescript
// Default scope: deny everything except DMs
const DEFAULT_QMD_SCOPE = {
  default: "deny",
  rules: [{ action: "allow", match: { chatType: "direct" } }],
};

export function isQmdScopeAllowed(scope, sessionKey?): boolean {
  // Parse session key → extract channel, chatType
  // Evaluate rules: match by channel, chatType, keyPrefix, rawKeyPrefix
  // Return rule action or fallback default
}
```

This prevents QMD memory from leaking into group chats unless explicitly configured.

### 5.10 QMD Error Recovery

The QMD manager handles several failure modes:
- **Null-byte collection metadata**: Detected via `ENOTDIR` + NUL markers, triggers collection rebuild
- **Unsupported CLI flags**: Falls back from `search`/`vsearch` to `query` mode
- **SQLite busy**: Returns friendly error when QMD index is locked
- **Output overflow**: Rejects if stdout/stderr exceed 200KB

---

## 6. Search Manager Factory & Fallback

### 6.1 Factory Pattern

**File:** `src/memory/search-manager.ts`

```typescript
export async function getMemorySearchManager(params: {
  cfg: OpenClawConfig;
  agentId: string;
  purpose?: "default" | "status";
}): Promise<MemorySearchManagerResult> {
  const resolved = resolveMemoryBackendConfig(params);

  if (resolved.backend === "qmd" && resolved.qmd) {
    // Try QMD first
    const primary = await QmdMemoryManager.create({...});
    if (primary) {
      // Wrap in FallbackMemoryManager
      const wrapper = new FallbackMemoryManager({
        primary,
        fallbackFactory: async () => {
          return await MemoryIndexManager.get(params);
        }
      });
      QMD_MANAGER_CACHE.set(cacheKey, wrapper);
      return { manager: wrapper };
    }
  }

  // Fall back to built-in
  const manager = await MemoryIndexManager.get(params);
  return { manager };
}
```

### 6.2 Fallback Manager

```typescript
class FallbackMemoryManager implements MemorySearchManager {
  private primaryFailed = false;

  async search(query, opts?) {
    if (!this.primaryFailed) {
      try {
        return await this.deps.primary.search(query, opts);
      } catch (err) {
        this.primaryFailed = true;
        this.lastError = err.message;
        // Evict from cache so next request retries QMD fresh
        this.evictCacheEntry();
      }
    }
    // Try built-in index as fallback
    const fallback = await this.ensureFallback();
    return await fallback.search(query, opts);
  }
}
```

Key behavior:
- QMD is primary, built-in is fallback
- **One failure switches permanently** to fallback for that manager instance
- Cache eviction ensures the next request creates a fresh QMD manager
- All interface methods (search, readFile, status, sync, probe) follow the same pattern

### 6.3 Caching

```typescript
const QMD_MANAGER_CACHE = new Map<string, MemorySearchManager>();

function buildQmdCacheKey(agentId: string, config: ResolvedQmdConfig): string {
  return `${agentId}:${stableSerialize(config)}`;
}
```

Managers are cached per `agentId + config hash`. The cache key uses stable JSON serialization with sorted keys.

---

## 7. Backend Configuration

### 7.1 Backend Resolution

**File:** `src/memory/backend-config.ts`

```typescript
export function resolveMemoryBackendConfig(params: {
  cfg: OpenClawConfig;
  agentId: string;
}): ResolvedMemoryBackendConfig {
  const backend = params.cfg.memory?.backend ?? "builtin";  // Default: builtin
  const citations = params.cfg.memory?.citations ?? "auto";

  if (backend !== "qmd") {
    return { backend: "builtin", citations };
  }

  // Resolve QMD configuration
  const workspaceDir = resolveAgentWorkspaceDir(params.cfg, params.agentId);
  const collections = [
    ...resolveDefaultCollections(includeDefaultMemory, workspaceDir, nameSet, agentId),
    ...resolveCustomPaths(qmdCfg?.paths, workspaceDir, nameSet, agentId),
  ];

  return {
    backend: "qmd",
    citations,
    qmd: { command, searchMode, collections, sessions, update, limits, scope },
  };
}
```

### 7.2 Resolved QMD Config

```typescript
export type ResolvedQmdConfig = {
  command: string;                          // "qmd" (default)
  searchMode: "query" | "search" | "vsearch"; // "search" (default)
  collections: ResolvedQmdCollection[];
  sessions: ResolvedQmdSessionConfig;
  update: {
    intervalMs: number;       // 5 minutes
    debounceMs: number;       // 15 seconds
    onBoot: boolean;          // true
    waitForBootSync: boolean; // false
    embedIntervalMs: number;  // 60 minutes
    commandTimeoutMs: number; // 30 seconds
    updateTimeoutMs: number;  // 120 seconds
    embedTimeoutMs: number;   // 120 seconds
  };
  limits: {
    maxResults: number;       // 6
    maxSnippetChars: number;  // 700
    maxInjectedChars: number; // 4000
    timeoutMs: number;        // 4000
  };
  includeDefaultMemory: boolean;
  scope?: SessionSendPolicyConfig;
};
```

---

## 8. Memory Configuration Schema

### 8.1 Top-Level Memory Config

**File:** `src/config/types.memory.ts`

```typescript
export type MemoryConfig = {
  backend?: "builtin" | "qmd";
  citations?: "auto" | "on" | "off";
  qmd?: MemoryQmdConfig;
};

export type MemoryQmdConfig = {
  command?: string;               // QMD binary path
  searchMode?: "query" | "search" | "vsearch";
  includeDefaultMemory?: boolean; // Include MEMORY.md, memory/ by default
  paths?: MemoryQmdIndexPath[];   // Custom paths to index
  sessions?: {
    enabled?: boolean;
    exportDir?: string;
    retentionDays?: number;
  };
  update?: {
    interval?: string;            // Duration string: "5m"
    debounceMs?: number;
    onBoot?: boolean;
    waitForBootSync?: boolean;
    embedInterval?: string;       // Duration string: "60m"
    commandTimeoutMs?: number;
    updateTimeoutMs?: number;
    embedTimeoutMs?: number;
  };
  limits?: {
    maxResults?: number;
    maxSnippetChars?: number;
    maxInjectedChars?: number;
    timeoutMs?: number;
  };
  scope?: SessionSendPolicyConfig;
};
```

### 8.2 Per-Agent Memory Search Config

**File:** `src/agents/memory-search.ts`

```typescript
export type ResolvedMemorySearchConfig = {
  enabled: boolean;
  sources: Array<"memory" | "sessions">;       // Default: ["memory"]
  extraPaths: string[];
  provider: "openai" | "local" | "gemini" | "voyage" | "auto";
  remote?: {
    baseUrl?: string;
    apiKey?: string;
    headers?: Record<string, string>;
    batch?: { enabled; wait; concurrency; pollIntervalMs; timeoutMinutes };
  };
  experimental: {
    sessionMemory: boolean;                    // Default: false
  };
  fallback: "openai" | "gemini" | "local" | "voyage" | "none";
  model: string;
  local: { modelPath?; modelCacheDir? };
  store: {
    driver: "sqlite";
    path: string;                              // <stateDir>/memory/<agentId>.sqlite
    vector: { enabled: boolean; extensionPath? };
  };
  chunking: {
    tokens: number;    // 400 (default)
    overlap: number;   // 80 (default)
  };
  sync: {
    onSessionStart: boolean;   // true
    onSearch: boolean;         // true
    watch: boolean;            // true
    watchDebounceMs: number;   // 1500
    intervalMinutes: number;   // 0 (disabled)
    sessions: {
      deltaBytes: number;     // 100_000
      deltaMessages: number;  // 50
    };
  };
  query: {
    maxResults: number;        // 6
    minScore: number;          // 0.35
    hybrid: {
      enabled: boolean;        // true
      vectorWeight: number;    // 0.7
      textWeight: number;      // 0.3
      candidateMultiplier: number; // 4
      mmr: { enabled; lambda };
      temporalDecay: { enabled; halfLifeDays };
    };
  };
  cache: { enabled: boolean; maxEntries? };
};
```

Config resolution merges `agents.defaults.memorySearch` with per-agent `agents.list[].memorySearch` overrides.

---

## 9. Agent Workspace Scoping

### 9.1 Agent ID Resolution

**File:** `src/agents/agent-scope.ts`

Agents are identified by their `id` in config, normalized to lowercase:

```typescript
export function resolveSessionAgentId(params: {
  sessionKey?: string;
  config?: OpenClawConfig;
}): string {
  const defaultAgentId = resolveDefaultAgentId(params.config ?? {});
  const parsed = parseAgentSessionKey(normalizedSessionKey);
  return parsed?.agentId ? normalizeAgentId(parsed.agentId) : defaultAgentId;
}
```

### 9.2 Workspace Directory Resolution

```typescript
export function resolveAgentWorkspaceDir(cfg: OpenClawConfig, agentId: string) {
  // 1. Per-agent config: agents.list[].workspace
  const configured = resolveAgentConfig(cfg, id)?.workspace;
  if (configured) return resolveUserPath(configured);

  // 2. Default agent: agents.defaults.workspace or CWD
  if (id === defaultAgentId) {
    const fallback = cfg.agents?.defaults?.workspace;
    if (fallback) return resolveUserPath(fallback);
    return resolveDefaultAgentWorkspaceDir(process.env);
  }

  // 3. Non-default agents: <stateDir>/workspace-<agentId>
  return path.join(stateDir, `workspace-${id}`);
}
```

### 9.3 Memory Isolation Per Agent

Each agent gets its own:
- **Workspace directory**: Determines which `MEMORY.md` / `memory/` files to index
- **SQLite database**: `<stateDir>/memory/<agentId>.sqlite`
- **Session transcripts directory**: `<stateDir>/agents/<agentId>/sessions/`
- **QMD state directory**: `<stateDir>/agents/<agentId>/qmd/`
- **QMD collection names**: Scoped with `<base>-<agentId>` suffix

### 9.4 Memory Search Config Per Agent

```typescript
export function resolveMemorySearchConfig(cfg, agentId): ResolvedMemorySearchConfig | null {
  const defaults = cfg.agents?.defaults?.memorySearch;
  const overrides = resolveAgentConfig(cfg, agentId)?.memorySearch;
  return mergeConfig(defaults, overrides, agentId);
}
```

This allows different agents to have completely different memory configurations: different embedding providers, different chunk sizes, different sync schedules.

---

## 10. Key Architectural Patterns

### 10.1 Dual Backend with Transparent Fallback

```
┌─────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│ Search Call  │────>│ FallbackMemoryManager│────>│ QmdMemoryManager│
│             │     │  (if backend=qmd)    │     │  (Primary)      │
└─────────────┘     │                      │     └─────────────────┘
                    │  On QMD failure:     │
                    │                      │     ┌─────────────────┐
                    │  Switch to ──────────│────>│MemoryIndexManager│
                    │                      │     │  (Fallback)     │
                    └──────────────────────┘     └─────────────────┘
```

### 10.2 Event-Driven Session → Memory Pipeline

```
Session Write
    │
    ▼
emitSessionTranscriptUpdate(sessionFile)
    │
    ▼
onSessionTranscriptUpdate listener
    │
    ▼
scheduleSessionDirty(sessionFile)  [debounce 5s]
    │
    ▼
processSessionDeltaBatch()
    │
    ├── Check deltaBytes threshold (100KB)
    ├── Check deltaMessages threshold (50)
    │
    ▼ (if either threshold met)
sync({ reason: "session-delta" })
    │
    ├── buildSessionEntry() → parse JSONL → extract text → hash
    ├── indexFileEntryIfChanged() → skip if hash same
    └── deleteStaleIndexedPaths() → clean up deleted sessions
```

### 10.3 Safe Reindexing Pattern

```
1. Create temp SQLite database
2. Seed embedding cache from old DB
3. Run full memory + session sync into temp DB
4. Write metadata
5. Close both databases
6. Atomic swap: temp → target (with backup)
7. Reopen target database
8. On failure: restore original DB from backup
```

### 10.4 Configuration Cascade

```
Global defaults (hardcoded)
    ▼
agents.defaults.memorySearch (shared baseline)
    ▼
agents.list[N].memorySearch (per-agent overrides)
    ▼
ResolvedMemorySearchConfig (final merged config)
```

### 10.5 Key Constants Summary

| Constant | Value | Purpose |
|----------|-------|---------|
| Default chunk tokens | 400 | Markdown chunk size |
| Default chunk overlap | 80 | Overlap between chunks |
| Watch debounce | 1500ms | File change debounce |
| Session dirty debounce | 5000ms | Session transcript debounce |
| Session delta bytes | 100,000 | Bytes threshold for re-index |
| Session delta messages | 50 | Lines threshold for re-index |
| Default max results | 6 | Search result limit |
| Default min score | 0.35 | Minimum similarity score |
| Hybrid vector weight | 0.7 | Vector vs text balance |
| Hybrid text weight | 0.3 | Text search weight |
| QMD update interval | 5min | Periodic index update |
| QMD embed interval | 60min | Embedding refresh |
| QMD debounce | 15s | Update debounce |
| QMD max results | 6 | QMD search limit |
| QMD max snippet chars | 700 | Snippet truncation |
| QMD max injected chars | 4000 | Total context budget |
| Session prune after | 30 days | Session maintenance |
| Session max entries | 500 | Session store cap |
| Session store TTL | 45s | In-memory cache TTL |

### 10.6 Interface Contract

**File:** `src/memory/types.ts`

```typescript
export interface MemorySearchManager {
  search(query, opts?): Promise<MemorySearchResult[]>;
  readFile(params): Promise<{ text; path }>;
  status(): MemoryProviderStatus;
  sync?(params?): Promise<void>;
  probeEmbeddingAvailability(): Promise<MemoryEmbeddingProbeResult>;
  probeVectorAvailability(): Promise<boolean>;
  close?(): Promise<void>;
}

export type MemorySearchResult = {
  path: string;       // Relative file path
  startLine: number;  // Snippet start line
  endLine: number;    // Snippet end line
  score: number;      // Relevance score [0, 1]
  snippet: string;    // Matching text excerpt
  source: "memory" | "sessions";
  citation?: string;  // Optional citation reference
};
```

Both `MemoryIndexManager` and `QmdMemoryManager` implement this interface, making the backend swap transparent to callers.

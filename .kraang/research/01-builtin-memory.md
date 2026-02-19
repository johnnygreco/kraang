# OpenClaw Built-in Memory System — Deep Technical Analysis

> Source: `https://github.com/openclaw/openclaw` — `src/memory/` directory
> Date: 2026-02-19

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Hybrid Search](#2-hybrid-search)
3. [Embedding System](#3-embedding-system)
4. [Chunking Strategy](#4-chunking-strategy)
5. [MMR Re-ranking](#5-mmr-re-ranking)
6. [Temporal Decay](#6-temporal-decay)
7. [SQLite Schema](#7-sqlite-schema)
8. [Vector Search (sqlite-vec)](#8-vector-search-sqlite-vec)
9. [Manager Architecture](#9-manager-architecture)
10. [Query Expansion (FTS-only Mode)](#10-query-expansion-fts-only-mode)
11. [Configuration Defaults](#11-configuration-defaults)
12. [Key Takeaways for Kraang](#12-key-takeaways-for-kraang)

---

## 1. Architecture Overview

OpenClaw's built-in memory system is a **file-based, locally-indexed semantic search engine** that operates on Markdown files. The architecture has these layers:

```
User Query
    │
    ▼
MemoryIndexManager (manager.ts)              ← Singleton per agent+workspace
    ├── search()                             ← Entry point
    │     ├── Vector Search (sqlite-vec)     ← Cosine similarity via extension
    │     ├── Keyword Search (FTS5/BM25)     ← Full-text search
    │     ├── mergeHybridResults()           ← Weighted linear combination
    │     ├── Temporal Decay                 ← Optional recency scoring
    │     └── MMR Re-ranking                 ← Optional diversity re-ranking
    │
    ├── sync()                               ← File watching + indexing
    │     ├── listMemoryFiles()              ← Discover MEMORY.md, memory/
    │     ├── chunkMarkdown()                ← Split into overlapping chunks
    │     └── embedBatch()                   ← Embed chunks via provider
    │
    └── SQLite DB                            ← Per-agent .sqlite file
          ├── files table                    ← File metadata + hash tracking
          ├── chunks table                   ← Chunk text + embeddings (JSON)
          ├── chunks_vec (virtual)           ← sqlite-vec float32 vectors
          ├── chunks_fts (virtual)           ← FTS5 full-text index
          └── embedding_cache table          ← Cross-session embedding cache
```

### Class Hierarchy

```
MemoryManagerSyncOps (manager-sync-ops.ts)    ← Abstract: DB init, file watching, sync lifecycle
    └── MemoryManagerEmbeddingOps (manager-embedding-ops.ts) ← Abstract: embedding, batching, indexing
        └── MemoryIndexManager (manager.ts)   ← Concrete: search, status, public API
```

### Backend Selection

The `search-manager.ts` module provides a **backend abstraction** supporting two modes:
- **`builtin`** (default): The `MemoryIndexManager` analyzed here
- **`qmd`**: An external CLI-based backend (`QmdMemoryManager`) with automatic fallback to builtin if QMD fails

```typescript
// search-manager.ts — Fallback wrapper
class FallbackMemoryManager implements MemorySearchManager {
  // Tries QMD primary, falls back to builtin MemoryIndexManager on failure
  async search(query, opts) {
    if (!this.primaryFailed) {
      try { return await this.deps.primary.search(query, opts); }
      catch (err) {
        this.primaryFailed = true;
        await this.deps.primary.close?.();
        this.evictCacheEntry();
      }
    }
    const fallback = await this.ensureFallback(); // → MemoryIndexManager.get()
    return await fallback.search(query, opts);
  }
}
```

---

## 2. Hybrid Search

**File:** `hybrid.ts`

The hybrid search combines **vector (semantic) search** and **keyword (BM25) search** using a **weighted linear combination**.

### Default Weights

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vectorWeight` | **0.7** | Weight for vector/semantic similarity |
| `textWeight` | **0.3** | Weight for BM25 keyword relevance |
| `candidateMultiplier` | **4** | Fetch `maxResults × 4` candidates from each source |
| `minScore` | **0.35** | Minimum hybrid score threshold |
| `maxResults` | **6** | Maximum results returned |

### Scoring Formula

```
hybridScore = vectorWeight × vectorScore + textWeight × textScore
```

where:
- `vectorScore` = `1 - cosine_distance` (from sqlite-vec) or `cosineSimilarity()` (fallback)
- `textScore` = `1 / (1 + bm25_rank)` (normalized from FTS5's BM25 rank)

### BM25 Rank Normalization

```typescript
// hybrid.ts
export function bm25RankToScore(rank: number): number {
  const normalized = Number.isFinite(rank) ? Math.max(0, rank) : 999;
  return 1 / (1 + normalized);  // Maps rank to (0, 1] — lower rank = higher score
}
```

### FTS Query Construction

Raw queries are tokenized into Unicode word tokens and combined with AND:

```typescript
// hybrid.ts
export function buildFtsQuery(raw: string): string | null {
  const tokens = raw.match(/[\p{L}\p{N}_]+/gu)
    ?.map((t) => t.trim()).filter(Boolean) ?? [];
  if (tokens.length === 0) return null;
  const quoted = tokens.map((t) => `"${t.replaceAll('"', "")}"`);
  return quoted.join(" AND ");
  // "how to deploy API" → '"how" AND "to" AND "deploy" AND "API"'
}
```

### Merge Algorithm

```typescript
// hybrid.ts — mergeHybridResults()
// 1. Union results by chunk ID into a Map
for (const r of params.vector) {
  byId.set(r.id, { ...r, vectorScore: r.vectorScore, textScore: 0 });
}
for (const r of params.keyword) {
  const existing = byId.get(r.id);
  if (existing) { existing.textScore = r.textScore; }
  else { byId.set(r.id, { ...r, vectorScore: 0, textScore: r.textScore }); }
}

// 2. Compute hybrid score
const merged = Array.from(byId.values()).map((entry) => ({
  ...entry,
  score: vectorWeight * entry.vectorScore + textWeight * entry.textScore,
}));

// 3. Apply temporal decay (if enabled)
const decayed = await applyTemporalDecayToHybridResults({ results: merged, ... });

// 4. Sort by score descending
const sorted = decayed.toSorted((a, b) => b.score - a.score);

// 5. Apply MMR re-ranking (if enabled)
if (mmrConfig.enabled) {
  return applyMMRToHybridResults(sorted, mmrConfig);
}
return sorted;
```

### FTS-Only Mode

When no embedding provider is available (no API keys), the system **gracefully degrades to FTS-only mode**:

```typescript
// manager.ts — search() in FTS-only mode
if (!this.provider) {
  // Extract keywords from conversational queries
  const keywords = extractKeywords(cleaned);
  const searchTerms = keywords.length > 0 ? keywords : [cleaned];

  // Search each keyword separately and merge, keeping highest score per chunk
  const resultSets = await Promise.all(
    searchTerms.map((term) => this.searchKeyword(term, candidates))
  );
  // Deduplicate + sort by score
}
```

---

## 3. Embedding System

**File:** `embeddings.ts`

### Provider Abstraction

```typescript
export type EmbeddingProvider = {
  id: string;
  model: string;
  maxInputTokens?: number;
  embedQuery: (text: string) => Promise<number[]>;
  embedBatch: (texts: string[]) => Promise<number[][]>;
};
```

### Supported Providers

| Provider | Default Model | Max Input Tokens |
|----------|---------------|------------------|
| **OpenAI** | `text-embedding-3-small` | 8192 |
| **Gemini** | `gemini-embedding-001` | 2048 |
| **Voyage** | `voyage-4-large` | 32000 |
| **Local** (node-llama-cpp) | `embeddinggemma-300m-qat-Q8_0.gguf` | N/A |

### Auto-Detection Priority

When `provider: "auto"` (default):

1. **Local first** — only if a local model file path is configured AND the file exists on disk
2. **OpenAI** — try if API key available
3. **Gemini** — try if API key available
4. **Voyage** — try if API key available
5. **FTS-only mode** — if all fail due to missing API keys, return `provider: null`

```typescript
// embeddings.ts — auto mode
if (canAutoSelectLocal(options)) {
  try { return createProvider("local"); } catch { /* store error, continue */ }
}
for (const provider of ["openai", "gemini", "voyage"]) {
  try { return createProvider(provider); }
  catch (err) {
    if (isMissingApiKeyError(err)) continue;  // Auth errors → try next
    throw err;  // Network errors → fatal
  }
}
return { provider: null };  // FTS-only mode
```

### Embedding Normalization

All embeddings are L2-normalized after generation:

```typescript
// embeddings.ts
function sanitizeAndNormalizeEmbedding(vec: number[]): number[] {
  const sanitized = vec.map((value) => (Number.isFinite(value) ? value : 0));
  const magnitude = Math.sqrt(sanitized.reduce((sum, value) => sum + value * value, 0));
  if (magnitude < 1e-10) return sanitized;
  return sanitized.map((value) => value / magnitude);
}
```

### Embedding Cache

Embeddings are cached in an SQLite table keyed by `(provider, model, provider_key, content_hash)`:

```typescript
// manager-embedding-ops.ts
private loadEmbeddingCache(hashes: string[]): Map<string, number[]> {
  // Batch query: SELECT hash, embedding FROM embedding_cache
  // WHERE provider = ? AND model = ? AND provider_key = ? AND hash IN (...)
}

private upsertEmbeddingCache(entries: Array<{ hash, embedding }>) {
  // INSERT ... ON CONFLICT DO UPDATE SET embedding=excluded.embedding, ...
}
```

The `provider_key` is a hash of the provider's configuration (base URL, model, non-auth headers) so cache entries are invalidated when the provider config changes.

Cache pruning deletes oldest entries (by `updated_at`) when count exceeds `maxEntries`:

```typescript
protected pruneEmbeddingCacheIfNeeded(): void {
  // DELETE FROM embedding_cache WHERE rowid IN (
  //   SELECT rowid FROM embedding_cache ORDER BY updated_at ASC LIMIT excess
  // )
}
```

### Batch Processing

Chunks are batched for embedding with a **token budget of 8000 per batch**:

```typescript
// manager-embedding-ops.ts
const EMBEDDING_BATCH_MAX_TOKENS = 8000;

private buildEmbeddingBatches(chunks: MemoryChunk[]): MemoryChunk[][] {
  // Accumulate chunks until batch exceeds 8000 estimated tokens
  // Oversized individual chunks get their own batch
}
```

### Retry Logic

Embedding calls include exponential backoff with jitter for rate limits:

```typescript
const EMBEDDING_RETRY_MAX_ATTEMPTS = 3;
const EMBEDDING_RETRY_BASE_DELAY_MS = 500;
const EMBEDDING_RETRY_MAX_DELAY_MS = 8000;

// Retryable: /(rate_limit|too many requests|429|resource has been exhausted|5\d\d|cloudflare)/i
// Delay doubles each attempt with 20% jitter
```

### Timeouts

| Context | Remote | Local |
|---------|--------|-------|
| Query (single embed) | 60s | 5 min |
| Batch embed | 2 min | 10 min |

### Batch API Support

For large indexing jobs, OpenClaw supports **provider-specific Batch APIs**:
- **OpenAI Batch API** (`batch-openai.ts`)
- **Gemini Batch Embedding** (`batch-gemini.ts`)
- **Voyage Batch API** (`batch-voyage.ts`)

Batch failures are tracked with a **circuit breaker** (limit: 2 failures before disabling batch mode):

```typescript
const BATCH_FAILURE_LIMIT = 2;
// After 2 failures, falls back to non-batch embedding permanently for the session
```

---

## 4. Chunking Strategy

**File:** `internal.ts`

### Default Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tokens` | **400** | Target chunk size in tokens |
| `overlap` | **80** | Overlap between adjacent chunks in tokens |

### Token-to-Character Approximation

**4 characters per token** (rough UTF-8 estimate):

```typescript
const maxChars = Math.max(32, chunking.tokens * 4);      // 400 × 4 = 1600 chars
const overlapChars = Math.max(0, chunking.overlap * 4);   // 80 × 4 = 320 chars
```

### Algorithm

```typescript
export function chunkMarkdown(
  content: string,
  chunking: { tokens: number; overlap: number },
): MemoryChunk[] {
  const lines = content.split("\n");
  const maxChars = Math.max(32, chunking.tokens * 4);
  const overlapChars = Math.max(0, chunking.overlap * 4);
  const chunks: MemoryChunk[] = [];
  let current: Array<{ line: string; lineNo: number }> = [];
  let currentChars = 0;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i] ?? "";
    const lineNo = i + 1;

    // Split very long lines into maxChars segments
    const segments = line.length === 0 ? [""]
      : Array.from({ length: Math.ceil(line.length / maxChars) },
          (_, j) => line.slice(j * maxChars, (j + 1) * maxChars));

    for (const segment of segments) {
      const lineSize = segment.length + 1;  // +1 for newline

      // Flush when chunk would exceed maxChars
      if (currentChars + lineSize > maxChars && current.length > 0) {
        flush();          // → create chunk from current lines
        carryOverlap();   // → keep last ~overlapChars worth of lines
      }
      current.push({ line: segment, lineNo });
      currentChars += lineSize;
    }
  }
  flush();  // Final chunk
  return chunks;
}
```

Key behaviors:
1. **Line-oriented**: Chunks break at line boundaries (never mid-line, except for extremely long lines)
2. **Overlap via carry-forward**: After flushing, the last N lines (up to `overlapChars` characters) are carried into the next chunk
3. **SHA-256 hash**: Each chunk gets a content hash for cache keying and deduplication
4. **Line tracking**: Each chunk records `startLine` and `endLine` (1-indexed) for citation

### Chunk Enforcement for Provider Limits

After chunking, an additional pass enforces per-provider input limits:

```typescript
// embedding-chunk-limits.ts
export function enforceEmbeddingMaxInputTokens(provider, chunks): MemoryChunk[] {
  const maxInputTokens = resolveEmbeddingMaxInputTokens(provider);
  for (const chunk of chunks) {
    if (estimateUtf8Bytes(chunk.text) <= maxInputTokens) {
      out.push(chunk);
    } else {
      // Split oversized chunk into sub-chunks within the provider's limit
      for (const text of splitTextToUtf8ByteLimit(chunk.text, maxInputTokens)) {
        out.push({ ...chunk, text, hash: hashText(text) });
      }
    }
  }
}
```

---

## 5. MMR Re-ranking

**File:** `mmr.ts`

Maximal Marginal Relevance (MMR) balances **relevance** with **diversity** in search results. It is **disabled by default** (opt-in).

### Default Config

```typescript
export const DEFAULT_MMR_CONFIG: MMRConfig = {
  enabled: false,    // Must be explicitly enabled
  lambda: 0.7,      // 0 = max diversity, 1 = max relevance
};
```

### Algorithm

Based on Carbonell & Goldstein (1998):

```
MMR = λ × normalized_relevance − (1−λ) × max_similarity_to_selected
```

```typescript
export function mmrRerank<T extends MMRItem>(items: T[], config): T[] {
  // 1. Pre-tokenize all items for Jaccard similarity
  const tokenCache = new Map<string, Set<string>>();
  for (const item of items) {
    tokenCache.set(item.id, tokenize(item.content));
  }

  // 2. Normalize scores to [0, 1]
  const scoreRange = maxScore - minScore;
  const normalizeScore = (score) => scoreRange === 0 ? 1 : (score - minScore) / scoreRange;

  // 3. Greedy iterative selection
  const selected: T[] = [];
  const remaining = new Set(items);

  while (remaining.size > 0) {
    let bestItem = null, bestMMRScore = -Infinity;

    for (const candidate of remaining) {
      const relevance = normalizeScore(candidate.score);
      const maxSim = maxSimilarityToSelected(candidate, selected, tokenCache);
      const mmrScore = lambda * relevance - (1 - lambda) * maxSim;

      if (mmrScore > bestMMRScore ||
          (mmrScore === bestMMRScore && candidate.score > bestItem?.score)) {
        bestMMRScore = mmrScore;
        bestItem = candidate;
      }
    }
    selected.push(bestItem);
    remaining.delete(bestItem);
  }
  return selected;
}
```

### Similarity Metric

MMR uses **Jaccard similarity** on lowercased alphanumeric tokens (NOT embedding vectors):

```typescript
export function tokenize(text: string): Set<string> {
  const tokens = text.toLowerCase().match(/[a-z0-9_]+/g) ?? [];
  return new Set(tokens);
}

export function jaccardSimilarity(setA: Set<string>, setB: Set<string>): number {
  // |A ∩ B| / |A ∪ B|
  let intersectionSize = 0;
  for (const token of smaller) {
    if (larger.has(token)) intersectionSize++;
  }
  return intersectionSize / (setA.size + setB.size - intersectionSize);
}
```

This is a deliberate design choice — using text-level Jaccard avoids requiring stored embedding vectors for the diversity calculation, making it fast and applicable even in FTS-only mode.

---

## 6. Temporal Decay

**File:** `temporal-decay.ts`

Temporal decay applies an **exponential decay multiplier** based on document age. **Disabled by default** (opt-in).

### Default Config

```typescript
export const DEFAULT_TEMPORAL_DECAY_CONFIG: TemporalDecayConfig = {
  enabled: false,
  halfLifeDays: 30,    // Score halves every 30 days
};
```

### Decay Formula

```
decayedScore = originalScore × e^(-λ × ageInDays)

where λ = ln(2) / halfLifeDays
```

```typescript
export function toDecayLambda(halfLifeDays: number): number {
  return Math.LN2 / halfLifeDays;  // 0.0231 for 30-day half-life
}

export function calculateTemporalDecayMultiplier(params: {
  ageInDays: number;
  halfLifeDays: number;
}): number {
  const lambda = toDecayLambda(params.halfLifeDays);
  return Math.exp(-lambda * Math.max(0, params.ageInDays));
}
```

### Decay Multiplier Examples (30-day half-life)

| Age | Multiplier | Effect |
|-----|-----------|--------|
| 0 days | 1.000 | Full score |
| 7 days | 0.851 | ~85% |
| 30 days | 0.500 | 50% (one half-life) |
| 60 days | 0.250 | 25% (two half-lives) |
| 90 days | 0.125 | 12.5% |

### Timestamp Extraction Priority

```
1. Date from file path: memory/2024-01-15.md → Jan 15, 2024
   (Regex: /(?:^|\/)memory\/(\d{4})-(\d{2})-(\d{2})\.md$/)

2. Evergreen files exempt: MEMORY.md, memory.md, memory/topic.md → NO DECAY
   (Only dated files like memory/YYYY-MM-DD.md decay)

3. File mtime fallback: fs.stat(absPath).mtimeMs
```

```typescript
// Evergreen memory files (MEMORY.md, topic files) are immune to decay
function isEvergreenMemoryPath(filePath: string): boolean {
  if (normalized === "MEMORY.md" || normalized === "memory.md") return true;
  if (!normalized.startsWith("memory/")) return false;
  return !DATED_MEMORY_PATH_RE.test(normalized);  // Only dated files decay
}
```

### Integration with Hybrid Search

Temporal decay is applied **after** the weighted linear combination but **before** MMR re-ranking:

```
Vector + Keyword → Hybrid Score → Temporal Decay → Sort → MMR Re-rank
```

---

## 7. SQLite Schema

**File:** `memory-schema.ts`

### Core Tables

```sql
-- Metadata store
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

-- Tracked files (one row per indexed file)
CREATE TABLE IF NOT EXISTS files (
  path TEXT PRIMARY KEY,
  source TEXT NOT NULL DEFAULT 'memory',   -- 'memory' or 'sessions'
  hash TEXT NOT NULL,                       -- SHA-256 of file content
  mtime INTEGER NOT NULL,
  size INTEGER NOT NULL
);

-- Chunk store (one row per text chunk)
CREATE TABLE IF NOT EXISTS chunks (
  id TEXT PRIMARY KEY,                      -- SHA-256 of source:path:lines:hash:model
  path TEXT NOT NULL,
  source TEXT NOT NULL DEFAULT 'memory',
  start_line INTEGER NOT NULL,
  end_line INTEGER NOT NULL,
  hash TEXT NOT NULL,                       -- SHA-256 of chunk text
  model TEXT NOT NULL,                      -- Embedding model used
  text TEXT NOT NULL,                       -- Raw chunk text
  embedding TEXT NOT NULL,                  -- JSON array of float32 values
  updated_at INTEGER NOT NULL
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);
```

### Embedding Cache Table

```sql
CREATE TABLE IF NOT EXISTS embedding_cache (
  provider TEXT NOT NULL,          -- 'openai', 'local', 'gemini', 'voyage'
  model TEXT NOT NULL,             -- e.g., 'text-embedding-3-small'
  provider_key TEXT NOT NULL,      -- Hash of provider config (base URL + headers)
  hash TEXT NOT NULL,              -- SHA-256 of input text
  embedding TEXT NOT NULL,         -- JSON float array
  dims INTEGER,                    -- Embedding dimensions
  updated_at INTEGER NOT NULL,
  PRIMARY KEY (provider, model, provider_key, hash)
);
CREATE INDEX IF NOT EXISTS idx_embedding_cache_updated_at ON embedding_cache(updated_at);
```

### FTS5 Virtual Table

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  text,                    -- Full-text indexed column
  id UNINDEXED,
  path UNINDEXED,
  source UNINDEXED,
  model UNINDEXED,
  start_line UNINDEXED,
  end_line UNINDEXED
);
```

### Vector Virtual Table (sqlite-vec)

Created dynamically when first embedding dimensions are known:

```sql
-- Created by ensureVectorTable(dimensions)
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
  id TEXT PRIMARY KEY,
  embedding float[{dimensions}]    -- e.g., float[1536] for OpenAI
);
```

### Meta Storage

The `meta` table stores a JSON blob with index configuration for change detection:

```typescript
type MemoryIndexMeta = {
  model: string;
  provider: string;
  providerKey?: string;
  chunkTokens: number;
  chunkOverlap: number;
  vectorDims?: number;
};
// Stored under key "memory_index_meta_v1"
```

When meta changes (e.g., switching embedding models), a **full reindex** is triggered.

---

## 8. Vector Search (sqlite-vec)

**File:** `sqlite-vec.ts`, `manager-search.ts`

### Extension Loading

```typescript
// sqlite-vec.ts
export async function loadSqliteVecExtension(params: {
  db: DatabaseSync;
  extensionPath?: string;
}): Promise<{ ok: boolean; extensionPath?: string; error?: string }> {
  const sqliteVec = await import("sqlite-vec");
  const extensionPath = resolvedPath ?? sqliteVec.getLoadablePath();
  params.db.enableLoadExtension(true);
  sqliteVec.load(params.db);  // or db.loadExtension() for custom paths
}
```

Key: Uses Node.js built-in `node:sqlite` (`DatabaseSync`) with the `sqlite-vec` npm package as a loadable extension.

### Vector Search Query

```typescript
// manager-search.ts — searchVector()
const rows = db.prepare(`
  SELECT c.id, c.path, c.start_line, c.end_line, c.text, c.source,
         vec_distance_cosine(v.embedding, ?) AS dist
    FROM chunks_vec v
    JOIN chunks c ON c.id = v.id
   WHERE c.model = ?${sourceFilter}
   ORDER BY dist ASC
   LIMIT ?
`).all(vectorToBlob(queryVec), providerModel, ...sourceParams, limit);

// Score = 1 - distance (cosine distance → cosine similarity)
return rows.map((row) => ({
  ...row,
  score: 1 - row.dist,
  snippet: truncateUtf16Safe(row.text, SNIPPET_MAX_CHARS),
}));
```

### Fallback (no sqlite-vec)

If the sqlite-vec extension is unavailable, falls back to **brute-force cosine similarity in JavaScript**:

```typescript
// manager-search.ts — searchVector() fallback
const candidates = listChunks({ db, providerModel, sourceFilter });
const scored = candidates.map((chunk) => ({
  chunk,
  score: cosineSimilarity(queryVec, chunk.embedding),  // Pure JS implementation
}));
return scored.toSorted((a, b) => b.score - a.score).slice(0, limit);
```

### Cosine Similarity Implementation

```typescript
// internal.ts
export function cosineSimilarity(a: number[], b: number[]): number {
  const len = Math.min(a.length, b.length);
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < len; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}
```

### Vector Data Format

Vectors are stored as **Float32Array buffers** when passing to sqlite-vec:

```typescript
const vectorToBlob = (embedding: number[]): Buffer =>
  Buffer.from(new Float32Array(embedding).buffer);
```

But in the `chunks` table, embeddings are stored as **JSON strings** (for the JS fallback path).

---

## 9. Manager Architecture

**File:** `manager.ts`, `manager-sync-ops.ts`

### Singleton Cache

One `MemoryIndexManager` per unique `(agentId, workspaceDir, settings)` tuple:

```typescript
const INDEX_CACHE = new Map<string, MemoryIndexManager>();

static async get(params): Promise<MemoryIndexManager | null> {
  const key = `${agentId}:${workspaceDir}:${JSON.stringify(settings)}`;
  const existing = INDEX_CACHE.get(key);
  if (existing) return existing;

  const providerResult = await createEmbeddingProvider({ ... });
  const manager = new MemoryIndexManager({ cacheKey: key, ... });
  INDEX_CACHE.set(key, manager);
  return manager;
}
```

### File Watching (Chokidar)

The manager watches for changes to memory files using Chokidar:

```typescript
// manager-sync-ops.ts
// Watches: MEMORY.md, memory/, extra configured paths
// Ignores: .git, node_modules, .pnpm-store, .venv, venv, .tox, __pycache__
// Debounce: 1500ms (configurable via watchDebounceMs)
```

### Sync Lifecycle

```typescript
async sync(params?) {
  if (this.syncing) return this.syncing;  // Deduplication: one sync at a time
  this.syncing = this.runSync(params).finally(() => { this.syncing = null; });
}
```

Sync triggers:
1. **On session start** (`sync.onSessionStart: true`)
2. **On search** (`sync.onSearch: true`) — if dirty flag is set
3. **On file change** (via Chokidar watcher)
4. **On interval** (`sync.intervalMinutes`)
5. **Session transcript deltas** — when session transcripts accumulate enough new data

### Search Flow

```typescript
async search(query, opts?) {
  // 1. Warm session (trigger sync if needed)
  void this.warmSession(opts?.sessionKey);

  // 2. Sync if dirty
  if (this.settings.sync.onSearch && (this.dirty || this.sessionsDirty)) {
    void this.sync({ reason: "search" });
  }

  // 3. Determine candidate count
  const candidates = min(200, max(1, floor(maxResults * candidateMultiplier)));

  // 4. FTS-only mode (no provider)
  if (!this.provider) {
    return ftsOnlySearch(cleaned, candidates, maxResults, minScore);
  }

  // 5. Hybrid mode: parallel vector + keyword search
  const keywordResults = hybrid.enabled
    ? await this.searchKeyword(cleaned, candidates)
    : [];
  const queryVec = await this.embedQueryWithTimeout(cleaned);
  const vectorResults = hasVector
    ? await this.searchVector(queryVec, candidates)
    : [];

  // 6. Merge and return
  const merged = await this.mergeHybridResults({
    vector: vectorResults, keyword: keywordResults,
    vectorWeight, textWeight, mmr, temporalDecay,
  });
  return merged.filter((e) => e.score >= minScore).slice(0, maxResults);
}
```

### Source Filtering

The system supports two memory sources:
- **`memory`**: Traditional `MEMORY.md` and `memory/` directory files
- **`sessions`**: Session transcript files (experimental)

Source filtering is applied as SQL WHERE clauses on every query.

### Change Detection

Files are tracked by content hash. On sync, each file is compared to the stored hash:

```typescript
// sync-index.ts — indexFileEntryIfChanged()
// If hash matches → skip (already indexed)
// If hash changed or force reindex → re-chunk + re-embed
```

Full reindex is triggered when:
- Embedding provider/model changes
- Chunk size or overlap changes
- Provider key changes
- First sync after manager creation

---

## 10. Query Expansion (FTS-only Mode)

**File:** `query-expansion.ts`

When operating without embeddings (FTS-only mode), conversational queries are expanded to keyword-based FTS queries:

```typescript
// query-expansion.ts
export function extractKeywords(query: string): string[] {
  const tokens = tokenize(query);  // Split into words + CJK characters/bigrams
  return tokens.filter((token) =>
    !STOP_WORDS_EN.has(token) &&
    !STOP_WORDS_ZH.has(token) &&
    isValidKeyword(token)
  );
}
// "that thing we discussed about the API" → ["discussed", "api"]
// "之前讨论的那个方案" → ["讨论", "方案"]
```

Features:
- **English stop word removal** (~80 words)
- **Chinese stop word removal** (~60 words)
- **CJK character n-grams** (unigrams + bigrams for Chinese text)
- **Short token filtering** (English words < 3 chars dropped)
- **Optional LLM-based expansion** (`expandQueryWithLlm()` — not yet used in core)

---

## 11. Configuration Defaults

**File:** `src/agents/memory-search.ts`

Complete default configuration:

```typescript
{
  enabled: true,
  sources: ["memory"],                    // Can add "sessions" (experimental)
  extraPaths: [],                         // Additional markdown directories
  provider: "auto",                       // Auto-detect: local → openai → gemini → voyage → fts-only
  fallback: "none",
  model: "",                              // Provider-specific default

  store: {
    driver: "sqlite",
    path: "~/.openclaw/state/memory/{agentId}.sqlite",
    vector: { enabled: true },
  },

  chunking: {
    tokens: 400,                          // ~1600 chars per chunk
    overlap: 80,                          // ~320 chars overlap
  },

  sync: {
    onSessionStart: true,
    onSearch: true,
    watch: true,
    watchDebounceMs: 1500,
    intervalMinutes: 0,                   // Disabled by default
    sessions: {
      deltaBytes: 100_000,               // Bytes of new transcript before re-sync
      deltaMessages: 50,                 // Messages before re-sync
    },
  },

  query: {
    maxResults: 6,
    minScore: 0.35,
    hybrid: {
      enabled: true,
      vectorWeight: 0.7,
      textWeight: 0.3,
      candidateMultiplier: 4,
      mmr: { enabled: false, lambda: 0.7 },
      temporalDecay: { enabled: false, halfLifeDays: 30 },
    },
  },

  cache: {
    enabled: true,
    maxEntries: undefined,                // Unlimited by default
  },
}
```

---

## 12. Key Takeaways for Kraang

### What OpenClaw Does Well

1. **Graceful degradation**: FTS-only mode when no embedding provider is available — search still works, just with keyword matching
2. **Hybrid search with tunable weights**: 70/30 vector/keyword split is a good starting point
3. **Embedding cache**: Content-hash-based caching across sessions saves significant API cost
4. **Provider abstraction**: Clean `EmbeddingProvider` interface supports multiple backends
5. **Singleton manager pattern**: Prevents duplicate DB connections and watchers
6. **Change detection via content hash**: Only re-embeds when content actually changes
7. **Batch API support**: For large initial indexing, uses provider batch APIs with circuit breaker
8. **Source filtering**: Clean separation of memory files vs session transcripts

### Design Decisions Worth Noting

1. **No semantic chunking**: Uses simple line-based chunking with character count, NOT markdown-aware semantic chunking (no heading/section boundaries)
2. **4 chars/token estimate**: Rough approximation; could under-count for CJK or over-count for code
3. **Embeddings stored as JSON**: The `chunks` table stores embeddings as JSON strings (for JS fallback), while `chunks_vec` stores them as Float32 blobs — some storage duplication
4. **MMR uses Jaccard, not embedding similarity**: Faster but less semantically aware for diversity
5. **Temporal decay on hybrid score, not per-source**: Applied uniformly after the weighted combination
6. **BM25 normalization is simplistic**: `1/(1+rank)` doesn't account for BM25 score magnitude, just relative ordering
7. **FTS query is AND-only**: `"token1" AND "token2"` — no fuzzy matching or OR fallback for partial matches
8. **No re-ranking with cross-encoders**: The system uses bi-encoder (embedding) similarity only

### Opportunities for Kraang

1. **Markdown-aware chunking**: Split on headings (##, ###) for more coherent chunks
2. **Tag/category-aware search**: OpenClaw has no metadata filtering — Kraang's tag system could add faceted search
3. **Cross-encoder re-ranking**: Add a lightweight reranker for top-K results
4. **Adaptive hybrid weights**: Learn optimal weights from user feedback
5. **Incremental FTS updates**: OpenClaw deletes + re-inserts all FTS entries for a file — could be optimized
6. **Configurable BM25 parameters**: FTS5 supports tuning b and k1 parameters
7. **Embedding dimension reduction**: Use Matryoshka embeddings or PCA for smaller/faster vectors

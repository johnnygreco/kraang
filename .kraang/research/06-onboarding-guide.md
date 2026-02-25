# Kraang Memory Upgrade — Onboarding Guide

**Branch:** `rethinking-memory`
**Date:** 2026-02-19
**Status:** Implementation complete + review hardening pass, 261 tests passing, all committed

---

## 1. What We Did (Executive Summary)

We studied the [OpenClaw](https://github.com/openclaw/openclaw) memory system in depth, extracted its best ideas, and integrated them into kraang. The upgrade adds **semantic search**, **hybrid retrieval**, and **prompt injection protection** — while preserving kraang's simplicity and keeping everything optional.

Before this work, kraang had:
- SQLite FTS5 keyword search
- Manual note management (`remember` / `recall` / `forget` / `status`)
- Session indexing from Claude Code JSONL transcripts

After this work, kraang also has:
- **Hybrid search** — vector (semantic) + keyword, weighted merge with FTS fallback
- **Embedding provider** — OpenAI text-embedding-3-small with caching, smart retry, and input validation
- **`context` tool** — auto-recall with prompt injection detection and safety framing
- **Graceful degradation everywhere** — no API key? No `sqlite-vec`? Network down mid-query? Works fine, FTS-only.
- **Embedding feedback** — `remember` output shows `[embedded]` or `[FTS only]`
- **CLI parity** — `kraang search` and `kraang status` now use hybrid search and show embedding info

### Review Hardening Pass

After initial implementation, a 7-agent review panel evaluated the code across Code Quality, Robustness, Functional Correctness, User Experience, Agent Experience, Test Strategy, and Simplicity. The review identified 98 findings. A 4-agent implementation team then addressed all findings in 5 commits:

1. **Removed ~530 lines of dead code** — `temporal_decay.py`, `mmr.py`, `extract_keywords()`, `prune_embedding_cache()` were implemented but never wired into any pipeline. Deleted per YAGNI; trivial to recreate.
2. **Hardened the search pipeline** — smart retry (no retry on 4xx), FTS fallback when embedding API fails, lowered `min_score` from 0.35→0.15 to fix FTS-only result filtering regression, proper type safety.
3. **Improved store robustness** — cosine score clamping, corrupted BLOB handling, brute-force filter fix, clearer naming.
4. **Improved UX** — differentiated status messages, embedding feedback in `remember`, injection detection wired into `context`, CLI hybrid search, `kraang init` hints about API key.
5. **Updated docs** — README documents semantic search, research docs corrected.

---

## 2. Architecture Overview

```
User/Agent
    │
    ▼
┌──────────────────┐
│   MCP Server     │  server.py — 6 tools
│   (FastMCP)      │  remember, recall, context, read_session, forget, status
└──────┬───────────┘
       │
       ├──────────────────────────────┐
       ▼                              ▼
┌──────────────┐                    ┌──────────────┐
│ hybrid.py    │                    │  safety.py   │
│ Vector+FTS   │                    │  Injection   │
│ merge        │                    │  protection  │
└──────┬───────┘                    └──────────────┘
       │
       ├──────────────────┐
       ▼                  ▼
┌──────────────┐  ┌──────────────┐
│ embeddings.py│  │  store.py    │
│ OpenAI       │  │  SQLite +    │
│ provider     │  │  FTS5 + vec  │
└──────────────┘  └──────────────┘
```

### Data flow for `recall("asyncio")`

1. `server.recall()` acquires the store and embedding provider singletons (via `asyncio.Lock`-protected lazy init)
2. Calls `hybrid_search(store, provider, "asyncio")`
3. `hybrid.py` embeds the query, then runs **vector search** and **FTS search** in parallel (`asyncio.gather`)
   - If `embed_query` fails, falls back to FTS-only (logs warning, does not crash)
4. Merges results by `note_id`, computing weighted scores: `0.7 * vec + 0.3 * fts`
5. Filters by `_MIN_SCORE` threshold (0.15)
6. Formatted as markdown and returned

**Future integration point:** After step 5, temporal decay and MMR diversity re-ranking could be applied. See section 9 for the pattern.

### Data flow for `context("asyncio")`

Same as recall, but:
- Results are checked against `looks_like_injection()` (flagged notes are logged as warnings)
- Output goes through `safety.format_recalled_context()` which wraps results in `<relevant-memories>` XML with an "untrusted historical data" warning and HTML-escaped content

---

## 3. Source Files

### `src/kraang/embeddings.py` (~160 lines)

**Purpose:** Embedding provider abstraction with OpenAI implementation.

Key components:
- **`EmbeddingProvider`** — `Protocol` defining the interface: `provider_id`, `model`, `dims`, `embed_query()`, `embed_batch()`. Docstring requires implementations to return L2-normalized vectors.
- **`OpenAIEmbeddingProvider`** — Calls `POST https://api.openai.com/v1/embeddings` with `text-embedding-3-small` (1536 dims)
  - Uses `httpx.AsyncClient` (async HTTP), client reused across retry attempts
  - L2 normalization via `_l2_normalize()` — logs warning if zero vector returned
  - Smart retry: only retries 429/5xx and network errors; raises immediately on 4xx (bad API key = instant failure, not 3 retries)
  - Input validation: `embed_query`/`embed_batch` reject empty or whitespace-only strings
  - Sorts response by `index` field to preserve input order
- **`create_provider()`** — Factory that reads `OPENAI_API_KEY` from env, returns `None` if missing

**Design decisions:**
- Protocol-based, not abstract class — any object with the right shape works. The Protocol is kept despite having one implementation because multiple providers (Ollama, local models) are explicitly planned, and it documents the L2 normalization contract.
- `httpx` rather than the `openai` SDK — fewer dependencies, more control over retry/timeout
- L2 normalization at the provider level — cosine similarity on normalized vectors is just dot product
- Returns `None` (not raises) when no key — enables graceful degradation

### `src/kraang/hybrid.py` (~130 lines)

**Purpose:** Weighted combination of vector and keyword search results.

Key components:
- **Module constants** (replacing the former `HybridConfig` dataclass):
  - `_VECTOR_WEIGHT = 0.7` — semantic similarity weight
  - `_TEXT_WEIGHT = 0.3` — keyword/BM25 weight
  - `_MIN_SCORE = 0.15` — threshold to filter low-quality results (lowered from 0.35 to prevent FTS-only result regression)
  - `_CANDIDATE_MULTIPLIER = 4` — over-fetch factor: fetch limit×N per source before merge
- **`_MergedEntry`** — `TypedDict` for type-safe merge dictionary (eliminates `type: ignore` comments)
- **`bm25_score_to_normalized(score)`** — Maps BM25 scores (unbounded positive reals) to (0, 1) via `score / (1 + score)`
- **`hybrid_search()`** — The main entry point:
  - If `provider is None`, falls back to FTS-only
  - If `provider.embed_query()` fails, falls back to FTS-only (logs warning)
  - Otherwise: embeds query, runs vector + FTS in parallel, merges by `note_id`
  - Store parameter properly typed as `SQLiteStore` via `TYPE_CHECKING` guard

**Design decisions:**
- 0.7/0.3 split favoring vectors — semantic understanding matters more than exact keyword matches
- `_MIN_SCORE = 0.15` — a FTS-only match scores at most ~0.27 (0.3 × 0.91). The old threshold of 0.35 silently dropped strong keyword matches when embeddings were enabled. 0.15 is low enough to include FTS-only hits.
- Module constants instead of `HybridConfig` dataclass — no user-facing mechanism to pass config; the dataclass was dead optionality. Constants can be promoted to a config when a tuning mechanism exists.
- `TYPE_CHECKING` guard for `SQLiteStore` — avoids circular import while preserving full type checking. `TypeError` raised (not `assert`) if wrong type passed.

### `src/kraang/safety.py` (~65 lines)

**Purpose:** Prompt injection detection and safe context formatting.

Key components:
- **`__all__`** — explicit export list
- **`INJECTION_PATTERNS`** — 8 compiled regexes matching known injection patterns
- **`looks_like_injection(text)`** — Returns `True` if text matches any pattern. Now wired into the `context` tool (flagged notes logged as warnings).
- **`escape_for_prompt(text)`** — HTML-escapes `& < > " '`
- **`format_recalled_context(results)`** — Wraps search results in safety-framed XML

**Design decisions:**
- Detection is conservative (low false-positive rate) — notes are flagged via logging, not blocked
- `looks_like_injection` is called in the `context` tool; not yet called in `remember` (could be added to flag notes at write time)
- HTML escaping + safety framing = defense-in-depth

---

## 4. Modified Files

### `src/kraang/store.py` (~930 lines)

New schema additions:
- **`embedding_cache` table** — Content-addressed cache for embedding API responses

New methods on `SQLiteStore`:

| Method | Purpose |
|--------|---------|
| `get_cached_embedding()` | Retrieve cached embedding by provider/model/content_hash. Handles corrupted BLOBs gracefully (returns `None`, logs warning). |
| `cache_embedding()` | Store an embedding BLOB in the cache |
| `ensure_vec_table()` | Create `notes_vec` virtual table (sqlite-vec), no-op if unavailable |
| `upsert_note_embedding()` | Store/update vector for a note (vec table or fallback). Sentinel values `'_vec'`/`'_fallback'` distinguish fallback entries from real cache entries. |
| `search_notes_vector()` | Vector similarity search (dispatches to native or brute-force) |
| `_search_vec_native()` | sqlite-vec `vec_distance_cosine` search. Scores clamped with `max(0.0, ...)`. |
| `_search_vec_bruteforce()` | Brute-force cosine similarity. Iterates until `limit` valid results (doesn't pre-slice). Handles corrupted BLOBs. |

Infrastructure:
- `_SQLITE_VEC_IMPORTABLE` flag (module level) — whether `import sqlite_vec` succeeded
- `_vec_loaded` instance flag — whether the extension actually loaded into this connection
- `_cosine_similarity()` — Pure Python cosine similarity helper
- `content_hash()` — SHA-256 for embedding cache keys (canonical implementation, imported by `server.py`)
- Embeddings stored as `BLOB` via `struct.pack("<Nf", ...)` (little-endian float32)

**Key design decisions:**
- **BLOB storage, not JSON** — 6KB vs 15-30KB per 1536-dim vector
- **Dual storage strategy** — native `vec0` when sqlite-vec available, brute-force fallback otherwise. Sentinel values `'_vec'`/`'_fallback'` in the embedding_cache table distinguish fallback vector entries from real API-response cache entries (documented in code).
- **Score clamping** — `vec_distance_cosine` returns [0, 2]; `max(0.0, 1.0 - dist)` prevents negative scores
- **Corrupted BLOB safety** — `struct.unpack` failures are caught, logged, and skipped (one bad entry doesn't crash search)
- **sqlite-vec failure at warning level** — visible with default INFO logging (was debug, invisible)

### `src/kraang/server.py` (~400 lines)

Key changes:
- **`asyncio.Lock`-protected lazy init** — `_get_store()` and `_get_provider()` use a shared lock to prevent concurrent MCP tool calls from creating duplicate connections
- **Transient failure retry** — `_provider_init_attempted` is NOT set on exception, so a brief DNS failure doesn't permanently disable semantic search
- **`content_hash()` import** — uses canonical `store.content_hash()` instead of inline hashlib
- **Differentiated status messages** — "disabled — set OPENAI_API_KEY..." vs "disabled — install with: pip install kraang[embeddings]"
- **Embedding feedback** — `remember` output shows `[embedded]` or `[FTS only]`
- **Injection detection** — `context` tool checks results against `looks_like_injection()`, logs warnings for flagged notes
- **TODO breadcrumb** — comment in `recall()` marking the integration point for future temporal_decay/mmr

### `src/kraang/search.py` (~100 lines)

- `extract_keywords()` and `STOP_WORDS` were deleted (dead code, never imported)
- Only `build_fts_query()` and related FTS5 query construction remain

### `src/kraang/formatter.py` (~290 lines)

- `format_remember_created()` and `format_remember_updated()` accept `embedded: bool` parameter
- Output appends `[embedded]` or `[FTS only]` indicator

### `src/kraang/cli.py` (~500 lines)

- `search` command uses `hybrid_search` when embeddings available (was FTS-only)
- `status` command shows embedding provider info
- `init` command hints about adding `OPENAI_API_KEY` to `.mcp.json`

### `src/kraang/__init__.py`

- `HybridConfig` removed from exports (replaced by module constants)

### `pyproject.toml`

- Optional dependency group `embeddings` with `httpx>=0.27` and `sqlite-vec>=0.1.6`

---

## 5. Test Coverage

**261 tests total**, all passing in ~2s.

### Test files

| File | Tests | What it covers |
|------|-------|----------------|
| `test_embeddings.py` | 26 | Provider creation, L2 normalization, smart retry (429 retried, 401 not), empty string validation, batch embedding, zero vector warning |
| `test_hybrid.py` | 16 | BM25 normalization, FTS fallback (no provider, failing provider), merge logic, min_score filtering, type safety, vector-only path |
| `test_safety.py` | 37 | Injection detection (8 patterns + negatives), HTML escaping, context formatting, empty results |
| `test_store.py` | 70 | Full CRUD, FTS search, embedding cache (get/cache/corrupt), vector search (brute-force, forgotten note filtering), cosine similarity edge cases, corrupted tags_json fallback |
| `test_server.py` | 42 | All 6 tools, context (XML, safety, max_results, error type), recall (FTS fallback), remember (embedding feedback), status (differentiated messages), provider fallback |
| `test_formatter.py` | 40 | All formatting functions, embedding indicators |
| `test_search.py` | 17 | FTS query building, edge cases |
| `test_models.py` | 13 | Data model validation |

### Testing strategy

- All embedding tests mock the OpenAI API — no real API calls. Shared `FakeClient`/`FakeResponse` helpers (refactored from duplicated pattern).
- The `store` and `populated_store` fixtures provide in-memory SQLite databases
- Tests verify **graceful degradation paths**: no provider, broken provider, no sqlite-vec, corrupted BLOBs, corrupted JSON
- `_search_vec_native` (sqlite-vec path) is not tested in CI since sqlite-vec may not be available. The brute-force fallback is thoroughly tested.

---

## 6. Key Design Decisions & Rationale

### What we adopted from OpenClaw

| Feature | OpenClaw approach | Our approach | Why |
|---------|-------------------|--------------|-----|
| Hybrid search | Vector + BM25, weighted merge | Same concept, simpler implementation | Core value — much better recall for semantic queries |
| Embedding cache | LanceDB with content dedup | SQLite table with SHA-256 content hash | Keeps everything in one DB |
| Injection protection | Multi-layer (detect + escape + frame) | Same pattern, wired into `context` tool | Essential for context-injection features |
| Auto-recall | `before_agent_start` hook | `context` tool (agent-initiated) | MCP tools are the natural interface |

### What we did NOT adopt (and why)

| Feature | Why we skipped it |
|---------|-------------------|
| **LanceDB** | External dependency. SQLite + sqlite-vec keeps everything in one file. |
| **Auto-capture** | Too noisy — manual `remember` is more intentional and higher quality. |
| **Chunking** | Kraang notes are already small units — chunking adds complexity with no benefit. |
| **Duplicate detection (0.95 cosine)** | Title-based upsert is simpler: same title = same note, updated in place. |
| **Temporal decay** | Implemented but deleted as dead code. Trivial to re-add when wired in. See section 9. |
| **MMR diversity** | Same — implemented, deleted, re-add when needed. |

### What we removed during review (and why)

| Removed | Rationale |
|---------|-----------|
| `temporal_decay.py` + tests | Never wired in. 38-line module trivial to recreate. YAGNI. |
| `mmr.py` + tests | Never wired in. 71-line module trivial to recreate. YAGNI. |
| `extract_keywords()` + `STOP_WORDS` | Never imported by any module. |
| `prune_embedding_cache()` | Never called. Add back when a caller exists. |
| `HybridConfig` dataclass | No user-facing mechanism to pass config. Replaced with module constants. |

### Why graceful degradation matters so much

```
OPENAI_API_KEY present? → Yes → httpx installed? → Yes → sqlite-vec? → Yes → Full hybrid search
                                                                       → No  → Brute-force cosine fallback
                                                    → No → ImportError caught, FTS-only
                         → No → Provider returns None, FTS-only

embed_query fails? → FTS-only fallback (logged, not crashed)
Corrupted BLOB?    → Skipped with warning, search continues
```

---

## 7. Module Dependency Graph

```
server.py
  ├── embeddings.py (try/except import)
  ├── hybrid.py
  │     ├── embeddings.py (EmbeddingProvider type via TYPE_CHECKING)
  │     ├── models.py (NoteSearchResult)
  │     └── search.py (build_fts_query)
  ├── safety.py (for context tool: format_recalled_context, looks_like_injection)
  │     └── models.py (NoteSearchResult)
  ├── store.py (content_hash import)
  │     ├── models.py
  │     └── config.py
  ├── search.py (build_fts_query)
  ├── formatter.py
  └── config.py
```

---

## 8. Configuration & Environment

### Environment variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `OPENAI_API_KEY` | No | (empty) | Enables semantic search via OpenAI embeddings |
| `KRAANG_DB_PATH` | No | `~/.kraang/kraang.db` | Override database location |

### Optional dependencies

```bash
pip install kraang[embeddings]
# or
uv pip install kraang[embeddings]
```

This installs:
- `httpx>=0.27` — Async HTTP client for OpenAI API
- `sqlite-vec>=0.1.6` — SQLite extension for vector similarity

### Hybrid search tuning

Search parameters are module constants in `hybrid.py`:

```python
_VECTOR_WEIGHT = 0.7      # Semantic similarity weight (0-1)
_TEXT_WEIGHT = 0.3         # BM25 keyword match weight (0-1)
_MIN_SCORE = 0.15          # Discard results below this hybrid score
_CANDIDATE_MULTIPLIER = 4  # Over-fetch factor: fetch limit×N per source before merge
```

To tune these, edit the constants directly. If user-facing configuration is needed in the future, promote these to a dataclass or environment variables.

---

## 9. What's Next (Suggested Improvements)

### Re-implement temporal decay + MMR and wire into the pipeline

These modules were deleted as dead code. They're trivial to recreate and the integration point is marked with a TODO in `server.py`'s `recall()` function:

```python
# In server.py recall() or context(), after hybrid_search returns:
from kraang.temporal_decay import apply_temporal_decay
from kraang.mmr import mmr_rerank

results = await hybrid_search(store, provider, query, limit=limit * 2)
results = apply_temporal_decay(results)  # exponential decay, 30-day half-life
results = mmr_rerank(results)            # diversity via Jaccard on text tokens
results = results[:limit]
```

The original implementations were:
- `temporal_decay`: `exp(-ln(2)/half_life * age_days)` with "evergreen"/"pinned" tag exemptions
- `mmr`: Greedy MMR selection (Carbonell & Goldstein, 1998) with `lambda_=0.7`

### Injection detection on `remember`

`looks_like_injection()` is wired into `context` but not `remember`. Adding it to `remember` would flag suspicious notes at write time:

```python
if looks_like_injection(content):
    logger.warning("Possible injection in note '%s'", title)
    # Still save it, but maybe add a "[flagged]" tag
```

### Embedding backfill CLI command

Existing notes don't have embeddings. A CLI command would be useful:

```bash
kraang embed-all  # embed all notes that don't have vectors yet
```

### Embedding cache pruning

`prune_embedding_cache()` was deleted. If the cache grows large, re-implement and wire into a startup routine or periodic maintenance step. The schema supports LRU eviction by `created_at`.

### More embedding providers

The `EmbeddingProvider` protocol makes it easy to add providers:
- **Ollama** — local inference, similar API to OpenAI
- **Local models** (e.g., sentence-transformers) — no API key needed
- **Anthropic** — if/when they offer an embeddings API

To add a provider:
1. Implement the `EmbeddingProvider` protocol (`provider_id`, `model`, `dims`, `embed_query`, `embed_batch`)
2. Ensure `embed_query`/`embed_batch` return L2-normalized vectors (use `_l2_normalize`)
3. Add detection logic in `create_provider()` (e.g., check `OLLAMA_BASE_URL` env var)
4. Update `pyproject.toml` optional deps if needed

### BM25 normalization asymmetry

The Functional Correctness reviewer noted that BM25 scores are normalized *after* relevance multiplication, meaning the saturation function dampens relevance's impact for FTS results vs linear impact for vector results. This is a known trade-off, not a bug — but if search quality tuning is needed, this is the place to look. See the review report for details.

### Update CLAUDE.md

The global CLAUDE.md instructions should be updated to mention the `context` tool so agents know to use it.

---

## 10. How to Verify

```bash
# Run all tests
uv run python -m pytest -q

# Run specific test suites
uv run python -m pytest tests/test_embeddings.py -v
uv run python -m pytest tests/test_hybrid.py -v
uv run python -m pytest tests/test_safety.py -v

# See commit history
git log --oneline

# Test with real embeddings (requires API key)
export OPENAI_API_KEY="sk-..."
pip install kraang[embeddings]
kraang serve  # start the MCP server
```

---

## 11. Commit History

```
ba278bf Update docs: README semantic search section, research doc corrections
3fc26cc Improve server UX: status messages, embedding feedback, CLI hybrid search
400992c Improve store robustness: score clamping, BLOB safety, naming clarity
6b956f9 Harden search pipeline: retry logic, min_score, type safety, FTS fallback
ca06230 Remove dead code: temporal_decay, mmr, extract_keywords, unused patterns
eea5cce Add hybrid semantic search, embeddings, and prompt safety
```

---

## 12. File Inventory

### Source files
- `src/kraang/embeddings.py` — Embedding provider (~160 lines)
- `src/kraang/hybrid.py` — Hybrid search (~130 lines)
- `src/kraang/safety.py` — Injection protection (~65 lines)
- `src/kraang/store.py` — SQLite store with embedding cache and vector search (~930 lines)
- `src/kraang/server.py` — MCP server, 6 tools (~400 lines)
- `src/kraang/search.py` — FTS5 query building (~100 lines)
- `src/kraang/formatter.py` — Output formatting (~290 lines)
- `src/kraang/cli.py` — CLI commands (~500 lines)

### Test files
- `tests/test_embeddings.py` — 26 tests
- `tests/test_hybrid.py` — 16 tests
- `tests/test_safety.py` — 37 tests
- `tests/test_store.py` — 70 tests
- `tests/test_server.py` — 42 tests
- `tests/test_formatter.py` — 40 tests
- `tests/test_search.py` — 17 tests
- `tests/test_models.py` — 13 tests
- **Total: 261 tests**

### Research documents (`.kraang/research/`)
- `01-builtin-memory.md` — OpenClaw built-in memory analysis
- `02-memory-plugins.md` — OpenClaw memory plugin analysis
- `03-sessions-sync.md` — OpenClaw session/sync analysis
- `04-openclaw-memory-spec.md` — Synthesis & comparison spec
- `05-implementation-plan.md` — Pre-implementation plan (has divergence disclaimer)
- `06-onboarding-guide.md` — This document

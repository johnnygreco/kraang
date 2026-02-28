"""Shared fixtures and factories for kraang tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kraang.store import SQLiteStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_db_path(tmp_path: Path) -> Path:
    """Return a temporary SQLite database path (file doesn't exist yet)."""
    return tmp_path / "test_kraang.db"


@pytest.fixture()
async def store(tmp_db_path: Path):
    """Yield an initialized SQLiteStore backed by a temp database."""
    s = SQLiteStore(str(tmp_db_path))
    await s.initialize()
    yield s
    await s.close()


@pytest.fixture()
async def populated_store(store):
    """A store pre-loaded with a sample corpus of notes."""
    for title, content, tags, category in SAMPLE_NOTES:
        await store.upsert_note(title=title, content=content, tags=tags, category=category)
    return store


# ---------------------------------------------------------------------------
# Sample corpus
# ---------------------------------------------------------------------------

SAMPLE_NOTES: list[tuple[str, str, list[str], str]] = [
    (
        "Python asyncio patterns",
        "Event loops, coroutines, and tasks are the core asyncio primitives.",
        ["python", "async", "programming"],
        "engineering",
    ),
    (
        "SQLite FTS5 guide",
        "FTS5 is a full-text search extension for SQLite using BM25 ranking.",
        ["sqlite", "search", "database"],
        "engineering",
    ),
    (
        "Weekly grocery list",
        "Milk, eggs, bread, avocados, chicken, rice, and vegetables.",
        ["shopping", "food"],
        "personal",
    ),
    (
        "MCP protocol overview",
        "Model Context Protocol enables LLMs to interact with external tools and data sources.",
        ["mcp", "ai", "protocol"],
        "engineering",
    ),
    (
        "Meditation practice notes",
        "Focus on breath for 10 minutes. Body scan technique helps with relaxation.",
        ["health", "mindfulness"],
        "wellness",
    ),
    (
        "Project Kraang architecture",
        "Long-term memory MCP server with SQLite FTS5 backend. Pydantic models for contracts.",
        ["kraang", "architecture", "mcp"],
        "engineering",
    ),
    (
        "Book: Thinking Fast and Slow",
        "System 1 is fast and intuitive. System 2 is slow and deliberate. Cognitive biases.",
        ["books", "psychology", "reading"],
        "learning",
    ),
    (
        "Docker compose tips",
        "Use volumes for persistent data. Networks for service communication. Health checks.",
        ["docker", "devops", "infrastructure"],
        "engineering",
    ),
    (
        "Birthday gift ideas",
        "Smart watch, cookbook, hiking gear, board game, concert tickets.",
        ["gifts", "personal"],
        "personal",
    ),
    (
        "Pydantic v2 migration",
        "model_validator replaces root_validator. ConfigDict replaces Config.",
        ["python", "pydantic", "migration"],
        "engineering",
    ),
    (
        "Morning routine",
        "Wake at 6am, meditate 10min, exercise 30min, cold shower, healthy breakfast.",
        ["health", "routine", "productivity"],
        "wellness",
    ),
    (
        "REST API design principles",
        "Use nouns for resources, HTTP verbs for actions. Pagination, filtering, versioning.",
        ["api", "rest", "design"],
        "engineering",
    ),
    (
        "Investment portfolio review",
        "60% index funds, 20% bonds, 10% international, 10% alternatives. Rebalance quarterly.",
        ["finance", "investing"],
        "finance",
    ),
    (
        "Garden planting schedule",
        "Tomatoes in March, herbs in April, squash in May. Water daily in summer.",
        ["garden", "plants"],
        "personal",
    ),
    (
        "Machine learning fundamentals",
        "Supervised vs unsupervised learning. Neural networks, gradient descent, backpropagation.",
        ["ml", "ai", "programming"],
        "learning",
    ),
]


# ---------------------------------------------------------------------------
# JSONL fixtures
# ---------------------------------------------------------------------------

SAMPLE_JSONL_ENTRIES = [
    {
        "type": "user",
        "sessionId": "test-session-001",
        "cwd": "/test/project",
        "gitBranch": "main",
        "timestamp": "2026-02-10T14:30:00Z",
        "isMeta": False,
        "isSidechain": False,
        "userType": "external",
        "message": {
            "role": "user",
            "content": "How do I configure FTS5 search in SQLite?",
        },
        "uuid": "msg-001",
        "parentUuid": None,
        "version": "2.1.38",
    },
    {
        "type": "assistant",
        "sessionId": "test-session-001",
        "cwd": "/test/project",
        "gitBranch": "main",
        "timestamp": "2026-02-10T14:31:00Z",
        "slug": "test-gathering-dove",
        "isSidechain": False,
        "userType": "external",
        "message": {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Let me look at the FTS5 configuration.",
                },
                {
                    "type": "tool_use",
                    "id": "tool-001",
                    "name": "Read",
                    "input": {"file_path": "/test/project/src/search.py"},
                },
            ],
        },
        "uuid": "msg-002",
        "parentUuid": "msg-001",
        "requestId": "req-001",
        "version": "2.1.38",
    },
    {
        "type": "user",
        "sessionId": "test-session-001",
        "cwd": "/test/project",
        "gitBranch": "main",
        "timestamp": "2026-02-10T14:31:30Z",
        "isMeta": False,
        "isSidechain": False,
        "userType": "external",
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tool-001",
                    "content": "file content here...",
                }
            ],
        },
        "uuid": "msg-003",
        "parentUuid": "msg-002",
        "version": "2.1.38",
    },
    {
        "type": "assistant",
        "sessionId": "test-session-001",
        "cwd": "/test/project",
        "gitBranch": "main",
        "timestamp": "2026-02-10T14:32:00Z",
        "isSidechain": False,
        "userType": "external",
        "message": {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "The root cause was missing porter stemming tokenizer."
                    " Add tokenize='porter unicode61' to FTS5.",
                },
            ],
        },
        "uuid": "msg-004",
        "parentUuid": "msg-003",
        "requestId": "req-002",
        "version": "2.1.38",
    },
    {
        "type": "user",
        "sessionId": "test-session-001",
        "cwd": "/test/project",
        "gitBranch": "main",
        "timestamp": "2026-02-10T14:33:00Z",
        "isMeta": False,
        "isSidechain": False,
        "userType": "external",
        "message": {
            "role": "user",
            "content": "Thanks, that fixed it!",
        },
        "uuid": "msg-005",
        "parentUuid": "msg-004",
        "version": "2.1.38",
    },
]


@pytest.fixture()
def sample_jsonl_file(tmp_path: Path) -> Path:
    """Create a sample JSONL session file."""
    jsonl_path = tmp_path / "test-session-001.jsonl"
    with open(jsonl_path, "w") as f:
        for entry in SAMPLE_JSONL_ENTRIES:
            f.write(json.dumps(entry) + "\n")
    return jsonl_path


@pytest.fixture()
def sample_jsonl_with_noise(tmp_path: Path) -> Path:
    """Create a JSONL file with noise entries that should be filtered."""
    entries = [
        {"type": "file-history-snapshot", "messageId": "x", "snapshot": {}},
        {"type": "progress", "data": {}},
        {
            "type": "user",
            "sessionId": "test-noise",
            "cwd": "/test",
            "timestamp": "2026-02-10T14:30:00Z",
            "isMeta": True,
            "isSidechain": False,
            "userType": "external",
            "message": {
                "role": "user",
                "content": "<local-command-caveat>some meta</local-command-caveat>",
            },
            "uuid": "n1",
            "parentUuid": None,
            "gitBranch": "main",
            "version": "2.1.38",
        },
        {
            "type": "user",
            "sessionId": "test-noise",
            "cwd": "/test",
            "timestamp": "2026-02-10T14:31:00Z",
            "isMeta": False,
            "isSidechain": False,
            "userType": "external",
            "message": {
                "role": "user",
                "content": "<system-reminder>reminder text</system-reminder>",
            },
            "uuid": "n2",
            "parentUuid": "n1",
            "gitBranch": "main",
            "version": "2.1.38",
        },
        {
            "type": "user",
            "sessionId": "test-noise",
            "cwd": "/test",
            "timestamp": "2026-02-10T14:32:00Z",
            "isMeta": False,
            "isSidechain": False,
            "userType": "external",
            "message": {
                "role": "user",
                "content": "This is the actual user question about search",
            },
            "uuid": "n3",
            "parentUuid": "n2",
            "gitBranch": "main",
            "version": "2.1.38",
        },
        {
            "type": "assistant",
            "sessionId": "test-noise",
            "cwd": "/test",
            "timestamp": "2026-02-10T14:33:00Z",
            "isSidechain": False,
            "userType": "external",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Here is the answer."}],
            },
            "uuid": "n4",
            "parentUuid": "n3",
            "requestId": "r1",
            "gitBranch": "main",
            "version": "2.1.38",
        },
    ]
    jsonl_path = tmp_path / "test-noise.jsonl"
    with open(jsonl_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    return jsonl_path


@pytest.fixture()
def env_db_path(tmp_db_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Set KRAANG_DB_PATH env var to the tmp db path."""
    monkeypatch.setenv("KRAANG_DB_PATH", str(tmp_db_path))
    return tmp_db_path

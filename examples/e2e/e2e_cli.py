#!/usr/bin/env python3
"""End-to-end tests for the kraang CLI (subprocess-based).

Exercises: kraang init, status, notes (empty), insert notes via API,
search, notes (populated).
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PASS = 0
FAIL = 0


def run_cli(args: list[str], *, env: dict[str, str], cwd: str | Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run a kraang CLI command and return the result."""
    return subprocess.run(
        ["kraang", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd,
    )


def ok(name: str) -> None:
    global PASS
    PASS += 1
    print(f"[PASS] {name}")


def fail(name: str, reason: str) -> None:
    global FAIL
    FAIL += 1
    print(f"[FAIL] {name}: {reason}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

tmpdir = tempfile.mkdtemp(prefix="kraang_e2e_cli_")
project_dir = Path(tmpdir) / "project"
project_dir.mkdir()

# find_project_root() walks up from cwd looking for .git
(project_dir / ".git").mkdir()

db_path = project_dir / ".kraang" / "kraang.db"

# Build env: inherit PATH so kraang binary is found, set KRAANG_DB_PATH
env = {
    "PATH": os.environ["PATH"],
    "HOME": os.environ.get("HOME", "/tmp"),
    "KRAANG_DB_PATH": str(db_path),
    # Suppress TERM issues in subprocess
    "TERM": "dumb",
}

# Also propagate VIRTUAL_ENV / PYTHONPATH if set (needed for editable installs)
for key in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME"):
    if key in os.environ:
        env[key] = os.environ[key]

try:
    # ------------------------------------------------------------------
    # Test 1: kraang init
    # ------------------------------------------------------------------
    r = run_cli(["init"], env=env, cwd=str(project_dir))
    if r.returncode != 0:
        fail("init: exit code", f"expected 0, got {r.returncode}\nstderr: {r.stderr}")
    else:
        ok("init: exit code 0")

    if db_path.exists():
        ok("init: .kraang/kraang.db created")
    else:
        fail("init: .kraang/kraang.db created", "file not found")

    mcp_json = project_dir / ".mcp.json"
    if mcp_json.exists():
        data = json.loads(mcp_json.read_text())
        if "kraang" in data.get("mcpServers", {}):
            ok("init: .mcp.json has kraang server")
        else:
            fail("init: .mcp.json has kraang server", f"got {data}")
    else:
        fail("init: .mcp.json created", "file not found")

    gitignore = project_dir / ".gitignore"
    if gitignore.exists() and ".kraang/" in gitignore.read_text():
        ok("init: .gitignore includes .kraang/")
    else:
        fail("init: .gitignore includes .kraang/", "missing entry")

    # ------------------------------------------------------------------
    # Test 2: kraang status
    # ------------------------------------------------------------------
    r = run_cli(["status"], env=env, cwd=str(project_dir))
    if r.returncode != 0:
        fail("status: exit code", f"expected 0, got {r.returncode}\nstderr: {r.stderr}")
    else:
        ok("status: exit code 0")

    output = r.stdout.lower()
    if "note" in output or "session" in output or "knowledge" in output or "status" in output:
        ok("status: output contains stats")
    else:
        fail("status: output contains stats", f"got: {r.stdout[:200]}")

    # ------------------------------------------------------------------
    # Test 3: kraang notes (empty)
    # ------------------------------------------------------------------
    r = run_cli(["notes"], env=env, cwd=str(project_dir))
    if r.returncode != 0:
        fail("notes (empty): exit code", f"expected 0, got {r.returncode}\nstderr: {r.stderr}")
    else:
        ok("notes (empty): exit code 0")

    # ------------------------------------------------------------------
    # Test 4: Insert notes via Python API
    # ------------------------------------------------------------------
    from kraang.store import SQLiteStore

    async def _insert_notes() -> None:
        async with SQLiteStore(str(db_path)) as store:
            await store.upsert_note(
                title="Python asyncio",
                content="Event loops and coroutines for async programming",
                tags=["python", "async"],
                category="engineering",
            )
            await store.upsert_note(
                title="SQLite FTS5",
                content="Full text search with SQLite FTS5 extension",
                tags=["sqlite", "search"],
                category="database",
            )
            await store.upsert_note(
                title="CLI testing patterns",
                content="Use subprocess.run for end-to-end CLI tests",
                tags=["testing", "cli"],
                category="testing",
            )

    try:
        asyncio.run(_insert_notes())
        ok("insert notes via API")
    except Exception as exc:
        fail("insert notes via API", str(exc))

    # ------------------------------------------------------------------
    # Test 5: kraang search
    # ------------------------------------------------------------------
    r = run_cli(["search", "asyncio"], env=env, cwd=str(project_dir))
    if r.returncode != 0:
        fail("search: exit code", f"expected 0, got {r.returncode}\nstderr: {r.stderr}")
    else:
        ok("search: exit code 0")

    if "asyncio" in r.stdout.lower():
        ok("search: results contain query term")
    else:
        fail("search: results contain query term", f"got: {r.stdout[:200]}")

    # ------------------------------------------------------------------
    # Test 6: kraang notes (populated)
    # ------------------------------------------------------------------
    r = run_cli(["notes"], env=env, cwd=str(project_dir))
    if r.returncode != 0:
        fail("notes (populated): exit code", f"expected 0, got {r.returncode}\nstderr: {r.stderr}")
    else:
        ok("notes (populated): exit code 0")

    stdout_lower = r.stdout.lower()
    if "asyncio" in stdout_lower or "python" in stdout_lower:
        ok("notes (populated): lists inserted notes")
    else:
        fail("notes (populated): lists inserted notes", f"got: {r.stdout[:300]}")

finally:
    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    shutil.rmtree(tmpdir, ignore_errors=True)

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
total = PASS + FAIL
print(f"\n{PASS}/{total} passed, {FAIL}/{total} failed")
sys.exit(1 if FAIL else 0)

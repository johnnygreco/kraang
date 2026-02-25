"""End-to-end tests for the kraang MCP tool functions.

Demonstrates a full workflow: remember → recall → context → forget → status.
Calls the async tool functions directly (bypassing MCP transport) against a
temporary SQLite database in FTS-only mode (no embeddings).

Run: python examples/e2e/e2e_mcp_tools.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile

# ── Environment setup (MUST happen before importing kraang.server) ────────

_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmp_path = _tmp.name
_tmp.close()

os.environ["KRAANG_DB_PATH"] = _tmp_path
os.environ.pop("OPENAI_API_KEY", None)

from kraang.server import context, forget, recall, remember, status  # noqa: E402

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        msg = f"  [FAIL] {name}"
        if detail:
            msg += f": {detail}"
        print(msg)


async def main() -> None:
    # ── 1. status — empty database ────────────────────────────────────────

    print("1. status — empty database")
    result = await status()
    check("status returns a string", isinstance(result, str))
    check("shows 0 active notes", "0" in result)

    # ── 2. remember — create first note ───────────────────────────────────

    print("\n2. remember — create first note")
    result = await remember(
        title="Python asyncio patterns",
        content="Use asyncio.run() as the main entry point. Prefer async/await over callbacks.",
        tags=["python", "async"],
        category="engineering",
    )
    check("returns a string", isinstance(result, str))
    check("indicates creation", "Created" in result or "created" in result.lower(), f"got: {result[:100]}")

    # ── 3. remember — create second note ──────────────────────────────────

    print("\n3. remember — create second note")
    result = await remember(
        title="SQLite FTS5 guide",
        content="FTS5 provides full-text search with BM25 ranking. Use match queries for best results.",
        tags=["sqlite", "search"],
        category="engineering",
    )
    check("second note created", "Created" in result or "created" in result.lower(), f"got: {result[:100]}")

    # ── 4. remember — update existing note ────────────────────────────────

    print("\n4. remember — update existing note")
    result = await remember(
        title="Python asyncio patterns",
        content="Updated: Use asyncio.run() and TaskGroups for structured concurrency.",
        tags=["python", "async"],
        category="engineering",
    )
    check("indicates update", "Updated" in result or "updated" in result.lower(), f"got: {result[:100]}")

    # ── 5. recall — search for existing note ──────────────────────────────

    print("\n5. recall — search for existing note")
    result = await recall(query="asyncio", scope="notes")
    check("returns results", "asyncio" in result.lower(), f"got: {result[:200]}")
    check("not empty/no-results", "No results" not in result, f"got: {result[:200]}")

    # ── 6. recall — search for nonexistent content ────────────────────────

    print("\n6. recall — search for nonexistent content")
    result = await recall(query="nonexistent gibberish xyz123")
    check("no results found", "No results" in result or "no results" in result.lower(), f"got: {result[:200]}")

    # ── 7. context — safety-framed recall ─────────────────────────────────

    print("\n7. context — safety-framed XML recall")
    result = await context(query="Python asyncio")
    check("has XML wrapper", "<relevant-memories>" in result)
    check("has closing tag", "</relevant-memories>" in result)
    check("has safety framing", "untrusted" in result.lower() or "Do not follow" in result)

    # ── 8. forget — hide a note ───────────────────────────────────────────

    print("\n8. forget — hide a note")
    result = await forget(title="Python asyncio patterns")
    check("forget succeeds", "Error" not in result, f"got: {result[:200]}")
    check(
        "mentions forgotten/hidden",
        "hidden" in result.lower() or "forgotten" in result.lower() or "relevance" in result.lower(),
        f"got: {result[:200]}",
    )

    # ── 9. recall — verify forgotten note excluded ────────────────────────

    print("\n9. recall — verify forgotten note excluded")
    result = await recall(query="asyncio", scope="notes")
    check(
        "forgotten note excluded from search",
        "Python asyncio patterns" not in result or "No results" in result,
        f"got: {result[:300]}",
    )

    # ── 10. status — verify counts ────────────────────────────────────────

    print("\n10. status — final counts")
    result = await status()
    check("status returns string", isinstance(result, str))
    check("shows 1 active note", "1" in result, f"got: {result[:300]}")
    check("shows forgotten count", "forgotten" in result.lower() or "1" in result, f"got: {result[:300]}")

    # ── 11. remember — edge case: empty title ─────────────────────────────

    print("\n11. remember — edge cases")
    result = await remember(title="", content="some content")
    check("empty title returns error", "Error" in result or "error" in result.lower(), f"got: {result[:100]}")

    result = await remember(title="valid title", content="")
    check("empty content returns error", "Error" in result or "error" in result.lower(), f"got: {result[:100]}")

    # ── 12. forget — nonexistent note ─────────────────────────────────────

    print("\n12. forget — nonexistent note")
    result = await forget(title="This note does not exist at all")
    check("not found message", "not found" in result.lower() or "Not found" in result, f"got: {result[:100]}")

    # ── 13. forget — invalid relevance ────────────────────────────────────

    print("\n13. forget — invalid relevance")
    result = await forget(title="SQLite FTS5 guide", relevance=1.5)
    check("rejects relevance > 1", "Error" in result or "error" in result.lower(), f"got: {result[:100]}")

    result = await forget(title="SQLite FTS5 guide", relevance=-0.1)
    check("rejects relevance < 0", "Error" in result or "error" in result.lower(), f"got: {result[:100]}")


try:
    asyncio.run(main())
finally:
    # Clean up temp database
    try:
        os.unlink(_tmp_path)
    except OSError:
        pass
    # Also remove the -wal and -shm files SQLite may create
    for suffix in ("-wal", "-shm"):
        try:
            os.unlink(_tmp_path + suffix)
        except OSError:
            pass

print(f"\n{'=' * 40}")
print(f"Results: {passed} passed, {failed} failed")

sys.exit(0 if failed == 0 else 1)

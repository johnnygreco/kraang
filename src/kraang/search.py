"""Search query processing for kraang — FTS5 query building and sanitization."""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# FTS5 special characters that need to be stripped from bare terms.
# ---------------------------------------------------------------------------

_FTS5_SPECIAL = re.compile(r"[\^\*\(\)\{\}\[\]:]")

# Match a quoted phrase (double quotes).
_QUOTED_PHRASE = re.compile(r'"[^"]*"')

# Recognised boolean operators that FTS5 understands natively.
_BOOLEAN_OPS = {"AND", "OR", "NOT"}


# ---------------------------------------------------------------------------
# Query processing
# ---------------------------------------------------------------------------


def sanitize_fts_query(raw: str) -> str:
    """Escape special FTS5 characters and prepare *raw* for a MATCH clause.

    - Strips FTS5 special chars from bare terms
    - Preserves quoted phrases intact
    - Returns ``""`` when the input is all whitespace / punctuation
    """
    if not raw or not raw.strip():
        return ""

    # If there's an odd number of quotes, strip ALL quotes to avoid FTS5 errors.
    if raw.count('"') % 2 != 0:
        raw = raw.replace('"', "")
        if not raw.strip():
            return ""

    # Pull out quoted phrases first, replacing with placeholders.
    phrases: list[str] = []

    def _stash(m: re.Match[str]) -> str:
        phrases.append(m.group(0))
        return f"\x00{len(phrases) - 1}\x00"

    text = _QUOTED_PHRASE.sub(_stash, raw)

    # Strip FTS5 special chars from the remaining bare text.
    text = _FTS5_SPECIAL.sub(" ", text)

    # Restore quoted phrases.
    for idx, phrase in enumerate(phrases):
        text = text.replace(f"\x00{idx}\x00", phrase)

    # Collapse whitespace.
    result = " ".join(text.split())
    return result


def build_fts_query(raw: str) -> str:
    """Convert a natural-language query string to FTS5 query syntax.

    * Individual bare words are wrapped in double quotes.
    * Quoted phrases are preserved verbatim.
    * ``AND`` / ``OR`` / ``NOT`` operators are kept as-is.
    * ``tag:xyz`` prefixes are stripped (handled as metadata filters).
    * Returns ``""`` if nothing useful remains.
    """
    # Strip tag:xyz prefixes before sanitization (colon is a special char).
    stripped = re.sub(r"\btag:\S+", "", raw, flags=re.IGNORECASE)

    sanitized = sanitize_fts_query(stripped)
    if not sanitized:
        return ""

    # Tokenise: pull out quoted phrases and individual words.
    tokens: list[tuple[str, int, str]] = []

    # Extract quoted phrases first.
    for match in _QUOTED_PHRASE.finditer(sanitized):
        tokens.append(("phrase", match.start(), match.group(0)))

    # Extract bare words (everything outside quotes).
    no_quotes = _QUOTED_PHRASE.sub(lambda m: " " * len(m.group(0)), sanitized)
    for match in re.finditer(r"\S+", no_quotes):
        word = match.group(0)
        tokens.append(("word", match.start(), word))

    # Sort by original position to preserve order.
    tokens.sort(key=lambda t: t[1])

    parts: list[str] = []
    for kind, _pos, value in tokens:
        if kind == "phrase" or value in _BOOLEAN_OPS:
            parts.append(value)
        else:
            parts.append(f'"{value}"')

    result = " ".join(parts)
    return result

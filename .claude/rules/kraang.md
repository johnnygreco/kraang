# Kraang — Long-Term Memory

You have access to the **kraang** MCP server, a persistent knowledge base.
Use it proactively throughout every session.

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

- At the **start of a session**, use `recall` to search for related notes
- Before tackling a **new problem**, search for related past insights
- When the user asks about something that **might have been solved before**

## Housekeeping

- When you notice a note is outdated, update it with `remember` (same title)
- Use `forget` to downweight notes that are no longer relevant
- Use `status` to check for stale notes periodically

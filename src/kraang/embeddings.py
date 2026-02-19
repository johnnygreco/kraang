"""Embedding provider abstraction for kraang — optional semantic search."""

from __future__ import annotations

import asyncio
import logging
import math
import os
from typing import Protocol, runtime_checkable

logger = logging.getLogger("kraang.embeddings")


# ---------------------------------------------------------------------------
# L2 normalisation helper
# ---------------------------------------------------------------------------


def _l2_normalize(vec: list[float]) -> list[float]:
    """Normalise a vector to unit length (L2 norm)."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        logger.warning(
            "Zero vector returned from embedding API — will not match any queries"
        )
        return vec
    return [x / norm for x in vec]


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Embedding provider interface.

    Implementations must return L2-normalized vectors (unit length)
    so that cosine similarity equals dot product.
    """

    @property
    def provider_id(self) -> str: ...

    @property
    def model(self) -> str: ...

    @property
    def dims(self) -> int: ...

    async def embed_query(self, text: str) -> list[float]: ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "text-embedding-3-small"
_DEFAULT_DIMS = 1536
_MAX_RETRIES = 3
_BASE_DELAY = 0.5  # seconds
_MAX_DELAY = 8.0  # seconds
_TIMEOUT = 60.0  # seconds


class OpenAIEmbeddingProvider:
    """OpenAI text-embedding-3-small via httpx (async)."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    @property
    def provider_id(self) -> str:
        return "openai"

    @property
    def model(self) -> str:
        return _DEFAULT_MODEL

    @property
    def dims(self) -> int:
        return _DEFAULT_DIMS

    # -- public API ----------------------------------------------------------

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")
        results = await self._call_api([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        if not texts:
            return []
        empty = [t for t in texts if not t or not t.strip()]
        if empty:
            raise ValueError("Cannot embed empty text")
        return await self._call_api(texts)

    # -- internals -----------------------------------------------------------

    async def _call_api(self, texts: list[str]) -> list[list[float]]:
        """Call the OpenAI embeddings endpoint with retry + exponential backoff."""
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "httpx is required for OpenAI embeddings. "
                "Install it with: pip install kraang[embeddings]"
            ) from exc

        last_exc: Exception | None = None
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            for attempt in range(1, _MAX_RETRIES + 1):
                try:
                    resp = await client.post(
                        "https://api.openai.com/v1/embeddings",
                        headers={
                            "Authorization": f"Bearer {self._api_key}",
                            "Content-Type": "application/json",
                        },
                        json={"input": texts, "model": self.model},
                    )
                    resp.raise_for_status()
                    data = resp.json()

                    # Sort by index to preserve input order.
                    items = sorted(data["data"], key=lambda d: d["index"])
                    return [_l2_normalize(item["embedding"]) for item in items]

                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code in (429, 500, 502, 503, 529):
                        last_exc = exc
                    else:
                        raise  # Non-retryable (401, 403, 400, etc.)
                except (httpx.ConnectError, httpx.TimeoutException, OSError) as exc:
                    last_exc = exc
                except Exception:
                    raise  # Don't retry unexpected errors

                if attempt < _MAX_RETRIES:
                    delay = min(_BASE_DELAY * (2 ** (attempt - 1)), _MAX_DELAY)
                    logger.warning(
                        "OpenAI embedding attempt %d/%d failed: %s — retrying in %.1fs",
                        attempt,
                        _MAX_RETRIES,
                        last_exc,
                        delay,
                    )
                    await asyncio.sleep(delay)

        raise RuntimeError(f"OpenAI embedding failed after {_MAX_RETRIES} attempts") from last_exc


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


async def create_provider() -> EmbeddingProvider | None:
    """Create an embedding provider from environment variables.

    Returns ``None`` if no API key is configured (FTS-only mode).
    """
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        logger.debug("OPENAI_API_KEY not set — semantic search disabled")
        return None
    return OpenAIEmbeddingProvider(api_key)

"""Embedding provider abstraction for kraang — semantic search."""

from __future__ import annotations

import asyncio
import logging
import math
import os
from typing import Protocol, runtime_checkable

import httpx
from fastembed import TextEmbedding

logger = logging.getLogger("kraang.embeddings")


# ---------------------------------------------------------------------------
# L2 normalisation helper
# ---------------------------------------------------------------------------


def _l2_normalize(vec: list[float]) -> list[float]:
    """Normalise a vector to unit length (L2 norm)."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        logger.warning("Zero vector returned from embedding API — will not match any queries")
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
        self._client: httpx.AsyncClient | None = None

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

    def _get_client(self) -> httpx.AsyncClient:
        """Return a reusable httpx client (lazy singleton)."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=_TIMEOUT)
        return self._client

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
        last_exc: Exception | None = None
        client = self._get_client()
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
# Local (fastembed) provider
# ---------------------------------------------------------------------------

_LOCAL_DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5-Q"

_KNOWN_DIMS: dict[str, int] = {
    "nomic-ai/nomic-embed-text-v1.5-Q": 768,
    "nomic-ai/nomic-embed-text-v1.5": 768,
    "BAAI/bge-small-en-v1.5": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}


class LocalEmbeddingProvider:
    """Local text embeddings via fastembed (ONNX-based, runs on CPU)."""

    def __init__(self, model: str = _LOCAL_DEFAULT_MODEL) -> None:
        self._model_name = model
        self._dims = _KNOWN_DIMS.get(model)
        self._engine: TextEmbedding | None = None
        self._init_lock = asyncio.Lock()

    async def _get_engine(self) -> TextEmbedding:
        """Lazily initialise the engine off the event loop."""
        if self._engine is not None:
            return self._engine
        async with self._init_lock:
            if self._engine is not None:
                return self._engine
            loop = asyncio.get_running_loop()
            engine = await loop.run_in_executor(
                None, lambda: TextEmbedding(model_name=self._model_name)
            )
            # If dims weren't in _KNOWN_DIMS, probe from the engine.
            if self._dims is None:
                sample = list(engine.embed(["dim probe"]))
                self._dims = len(sample[0])
            self._engine = engine
            return self._engine

    @property
    def provider_id(self) -> str:
        return "local"

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def dims(self) -> int:
        if self._dims is None:
            raise RuntimeError("Provider not initialised — call embed_query() first")
        return self._dims

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        if not texts:
            return []
        empty = [t for t in texts if not t or not t.strip()]
        if empty:
            raise ValueError("Cannot embed empty text")
        engine = await self._get_engine()
        loop = asyncio.get_running_loop()
        raw = await loop.run_in_executor(None, lambda: list(engine.embed(texts)))
        return [_l2_normalize(vec.tolist()) for vec in raw]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


async def create_provider() -> EmbeddingProvider | None:
    """Create an embedding provider from environment configuration.

    Resolution order:
    1. ``KRAANG_EMBEDDING_PROVIDER`` explicitly set to ``"openai"`` or ``"local"``
    2. Auto-detect: OpenAI if ``OPENAI_API_KEY`` is present, otherwise local

    Returns ``None`` only if ``"openai"`` is requested but no API key is set.
    """
    explicit = os.environ.get("KRAANG_EMBEDDING_PROVIDER", "").strip().lower()
    model_override = os.environ.get("KRAANG_EMBEDDING_MODEL", "").strip() or None

    if explicit == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            logger.warning("KRAANG_EMBEDDING_PROVIDER=openai but OPENAI_API_KEY not set")
            return None
        if model_override:
            logger.warning(
                "KRAANG_EMBEDDING_MODEL=%s ignored for OpenAI provider", model_override
            )
        return OpenAIEmbeddingProvider(api_key)

    if explicit == "local":
        provider = LocalEmbeddingProvider(model=model_override or _LOCAL_DEFAULT_MODEL)
        await provider._get_engine()  # Eagerly load so .dims is available
        return provider

    if explicit:
        logger.warning(
            "Unknown KRAANG_EMBEDDING_PROVIDER=%r — falling back to auto-detect", explicit
        )

    # Auto-detect: prefer OpenAI when key is available, otherwise local
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if api_key:
        return OpenAIEmbeddingProvider(api_key)

    logger.info("No OPENAI_API_KEY — using local embeddings (%s)", _LOCAL_DEFAULT_MODEL)
    provider = LocalEmbeddingProvider(model=model_override or _LOCAL_DEFAULT_MODEL)
    await provider._get_engine()  # Eagerly load so .dims is available
    return provider

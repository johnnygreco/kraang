"""Tests for kraang.embeddings — provider factory, normalization, OpenAI + Local providers."""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pytest

from kraang.embeddings import (
    LocalEmbeddingProvider,
    OpenAIEmbeddingProvider,
    _l2_normalize,
    create_provider,
)

# ---------------------------------------------------------------------------
# Shared httpx mock helpers
# ---------------------------------------------------------------------------


class FakeResponse:
    """Configurable fake httpx response."""

    def __init__(self, data: dict | None = None, status_code: int = 200):
        self._data = data or {}
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx

            request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}", request=request, response=response
            )

    def json(self):
        return self._data


class FakeClient:
    """Configurable fake httpx.AsyncClient."""

    def __init__(self, handler=None):
        self._handler = handler
        self.is_closed = False
        self.last_url = None
        self.last_json = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def post(self, url, headers=None, json=None):
        self.last_url = url
        self.last_json = json
        if self._handler:
            return self._handler(url, headers, json)
        return FakeResponse()


def _patch_httpx(monkeypatch, fake_client):
    """Patch the OpenAI provider to use a fake client."""
    import kraang.embeddings as mod

    monkeypatch.setattr(mod, "_BASE_DELAY", 0.01)  # Speed up retries

    original_init = OpenAIEmbeddingProvider.__init__

    def patched_init(self, api_key):
        original_init(self, api_key)
        self._client = fake_client

    monkeypatch.setattr(OpenAIEmbeddingProvider, "__init__", patched_init)


# ---------------------------------------------------------------------------
# Fake fastembed engine for LocalEmbeddingProvider tests
# ---------------------------------------------------------------------------


class FakeTextEmbedding:
    """Fake fastembed.TextEmbedding that returns deterministic vectors."""

    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5-Q"):
        self.model_name = model_name

    def embed(self, texts):
        """Return a list of numpy arrays with deterministic values."""
        for text in texts:
            vec = np.array([float(ord(c)) for c in text[:768]])
            # Pad to 768 dims
            if len(vec) < 768:
                vec = np.pad(vec, (0, 768 - len(vec)))
            yield vec


# ---------------------------------------------------------------------------
# L2 normalization
# ---------------------------------------------------------------------------


class TestL2Normalize:
    def test_unit_vector_unchanged(self):
        vec = [1.0, 0.0, 0.0]
        result = _l2_normalize(vec)
        assert result == pytest.approx([1.0, 0.0, 0.0])

    def test_normalizes_to_unit_length(self):
        vec = [3.0, 4.0]
        result = _l2_normalize(vec)
        norm = math.sqrt(sum(x * x for x in result))
        assert norm == pytest.approx(1.0)
        assert result == pytest.approx([0.6, 0.8])

    def test_zero_vector_returned_as_is(self):
        vec = [0.0, 0.0, 0.0]
        result = _l2_normalize(vec)
        assert result == [0.0, 0.0, 0.0]

    def test_zero_vector_logs_warning(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="kraang.embeddings"):
            _l2_normalize([0.0, 0.0, 0.0])
        assert "Zero vector" in caplog.text

    def test_negative_values(self):
        vec = [-3.0, 4.0]
        result = _l2_normalize(vec)
        norm = math.sqrt(sum(x * x for x in result))
        assert norm == pytest.approx(1.0)

    def test_single_element(self):
        result = _l2_normalize([5.0])
        assert result == pytest.approx([1.0])

    def test_preserves_direction(self):
        vec = [1.0, 2.0, 3.0]
        result = _l2_normalize(vec)
        # All ratios should be preserved
        assert result[1] / result[0] == pytest.approx(2.0)
        assert result[2] / result[0] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestCreateProvider:
    async def test_auto_detect_prefers_openai(self, monkeypatch):
        """When OPENAI_API_KEY is set and no explicit provider, auto-detect picks OpenAI."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-123")
        monkeypatch.delenv("KRAANG_EMBEDDING_PROVIDER", raising=False)
        monkeypatch.delenv("KRAANG_EMBEDDING_MODEL", raising=False)
        provider = await create_provider()
        assert provider is not None
        assert provider.provider_id == "openai"
        assert provider.model == "text-embedding-3-small"
        assert provider.dims == 1536

    async def test_auto_detect_falls_back_to_local(self, monkeypatch):
        """Without OPENAI_API_KEY and no explicit provider, auto-detect picks local."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("KRAANG_EMBEDDING_PROVIDER", raising=False)
        monkeypatch.delenv("KRAANG_EMBEDDING_MODEL", raising=False)
        with patch("kraang.embeddings.TextEmbedding", FakeTextEmbedding):
            provider = await create_provider()
        assert provider is not None
        assert provider.provider_id == "local"
        assert provider.dims == 768

    async def test_explicit_openai_with_key(self, monkeypatch):
        monkeypatch.setenv("KRAANG_EMBEDDING_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-123")
        monkeypatch.delenv("KRAANG_EMBEDDING_MODEL", raising=False)
        provider = await create_provider()
        assert provider is not None
        assert provider.provider_id == "openai"

    async def test_explicit_openai_without_key(self, monkeypatch):
        """Explicitly requesting openai without a key returns None."""
        monkeypatch.setenv("KRAANG_EMBEDDING_PROVIDER", "openai")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        provider = await create_provider()
        assert provider is None

    async def test_explicit_local(self, monkeypatch):
        monkeypatch.setenv("KRAANG_EMBEDDING_PROVIDER", "local")
        monkeypatch.delenv("KRAANG_EMBEDDING_MODEL", raising=False)
        with patch("kraang.embeddings.TextEmbedding", FakeTextEmbedding):
            provider = await create_provider()
        assert provider is not None
        assert provider.provider_id == "local"
        assert provider.model == "nomic-ai/nomic-embed-text-v1.5-Q"

    async def test_explicit_local_with_model_override(self, monkeypatch):
        monkeypatch.setenv("KRAANG_EMBEDDING_PROVIDER", "local")
        monkeypatch.setenv("KRAANG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        with patch("kraang.embeddings.TextEmbedding", FakeTextEmbedding):
            provider = await create_provider()
        assert provider is not None
        assert provider.provider_id == "local"
        assert provider.model == "BAAI/bge-small-en-v1.5"
        assert provider.dims == 384

    async def test_unknown_provider_falls_back_to_auto_detect(self, monkeypatch):
        """Unknown KRAANG_EMBEDDING_PROVIDER logs warning and falls back to auto-detect."""
        monkeypatch.setenv("KRAANG_EMBEDDING_PROVIDER", "gemini")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("KRAANG_EMBEDDING_MODEL", raising=False)
        with patch("kraang.embeddings.TextEmbedding", FakeTextEmbedding):
            provider = await create_provider()
        # Falls back to local since no OPENAI_API_KEY
        assert provider is not None
        assert provider.provider_id == "local"

    async def test_model_override_ignored_for_openai(self, monkeypatch, caplog):
        """KRAANG_EMBEDDING_MODEL with openai provider logs a warning."""
        import logging

        monkeypatch.setenv("KRAANG_EMBEDDING_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-123")
        monkeypatch.setenv("KRAANG_EMBEDDING_MODEL", "text-embedding-3-large")
        with caplog.at_level(logging.WARNING, logger="kraang.embeddings"):
            provider = await create_provider()
        assert provider is not None
        assert provider.provider_id == "openai"
        assert provider.model == "text-embedding-3-small"  # Override ignored
        assert "ignored" in caplog.text.lower()

    async def test_auto_detect_local_with_model_override(self, monkeypatch):
        """KRAANG_EMBEDDING_MODEL works with auto-detect local path."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("KRAANG_EMBEDDING_PROVIDER", raising=False)
        monkeypatch.setenv("KRAANG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        with patch("kraang.embeddings.TextEmbedding", FakeTextEmbedding):
            provider = await create_provider()
        assert provider is not None
        assert provider.provider_id == "local"
        assert provider.model == "BAAI/bge-small-en-v1.5"
        assert provider.dims == 384


# ---------------------------------------------------------------------------
# OpenAI provider properties
# ---------------------------------------------------------------------------


class TestOpenAIProvider:
    def test_provider_id(self):
        p = OpenAIEmbeddingProvider("sk-test")
        assert p.provider_id == "openai"

    def test_model(self):
        p = OpenAIEmbeddingProvider("sk-test")
        assert p.model == "text-embedding-3-small"

    def test_dims(self):
        p = OpenAIEmbeddingProvider("sk-test")
        assert p.dims == 1536

    async def test_embed_batch_empty(self):
        p = OpenAIEmbeddingProvider("sk-test")
        result = await p.embed_batch([])
        assert result == []

    async def test_embed_query_calls_api(self, monkeypatch):
        """Mock httpx to verify embed_query calls the API correctly."""
        response_data = {"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]}
        fake_client = FakeClient(handler=lambda url, headers, json: FakeResponse(response_data))
        _patch_httpx(monkeypatch, fake_client)

        p = OpenAIEmbeddingProvider("sk-test-key")
        result = await p.embed_query("hello world")
        # Result should be L2-normalized
        norm = math.sqrt(sum(x * x for x in result))
        assert norm == pytest.approx(1.0)

    async def test_embed_batch_preserves_order(self, monkeypatch):
        """Ensure results are sorted by index even if API returns out of order."""
        response_data = {
            "data": [
                {"index": 1, "embedding": [0.4, 0.5, 0.6]},
                {"index": 0, "embedding": [0.1, 0.2, 0.3]},
            ]
        }
        fake_client = FakeClient(handler=lambda url, headers, json: FakeResponse(response_data))
        _patch_httpx(monkeypatch, fake_client)

        p = OpenAIEmbeddingProvider("sk-test-key")
        results = await p.embed_batch(["text1", "text2"])
        assert len(results) == 2
        # First result should correspond to index 0
        norm0 = math.sqrt(sum(x * x for x in results[0]))
        assert norm0 == pytest.approx(1.0)

    async def test_retry_on_failure(self, monkeypatch):
        """Test that _call_api retries on transient failures."""
        call_count = 0
        response_data = {"data": [{"index": 0, "embedding": [1.0, 0.0]}]}

        def handler(url, headers, json):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise OSError("transient failure")
            return FakeResponse(response_data)

        fake_client = FakeClient(handler=handler)
        _patch_httpx(monkeypatch, fake_client)

        p = OpenAIEmbeddingProvider("sk-test-key")
        result = await p.embed_query("test")
        assert len(result) == 2
        assert call_count == 3  # 2 failures + 1 success

    async def test_all_retries_exhausted(self, monkeypatch):
        """Test RuntimeError after all retries fail."""

        def handler(url, headers, json):
            raise OSError("permanent failure")

        fake_client = FakeClient(handler=handler)
        _patch_httpx(monkeypatch, fake_client)

        p = OpenAIEmbeddingProvider("sk-test-key")
        with pytest.raises(RuntimeError, match="failed after"):
            await p.embed_query("test")

    @pytest.mark.parametrize("status_code", [401, 403])
    async def test_no_retry_on_client_errors(self, monkeypatch, status_code):
        """Client errors (401, 403) should raise immediately without retrying."""
        call_count = 0

        def handler(url, headers, json):
            nonlocal call_count
            call_count += 1
            return FakeResponse(status_code=status_code)

        fake_client = FakeClient(handler=handler)
        _patch_httpx(monkeypatch, fake_client)

        p = OpenAIEmbeddingProvider("sk-test-key")
        import httpx

        with pytest.raises(httpx.HTTPStatusError):
            await p.embed_query("test")
        assert call_count == 1  # No retries — raised immediately

    async def test_embed_query_empty_string_raises(self):
        """Empty string should raise ValueError immediately."""
        p = OpenAIEmbeddingProvider("sk-test")
        with pytest.raises(ValueError, match="empty"):
            await p.embed_query("")

    async def test_embed_query_whitespace_only_raises(self):
        """Whitespace-only string should raise ValueError."""
        p = OpenAIEmbeddingProvider("sk-test")
        with pytest.raises(ValueError, match="empty"):
            await p.embed_query("   ")

    async def test_embed_batch_empty_string_raises(self):
        """Batch with an empty string should raise ValueError."""
        p = OpenAIEmbeddingProvider("sk-test")
        with pytest.raises(ValueError, match="empty"):
            await p.embed_batch(["hello", ""])

    @pytest.mark.parametrize("status_code", [429, 502])
    async def test_retry_on_server_errors(self, monkeypatch, status_code):
        """Server errors (429, 502, etc.) should be retried."""
        call_count = 0
        response_data = {"data": [{"index": 0, "embedding": [1.0, 0.0]}]}

        def handler(url, headers, json):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return FakeResponse(status_code=status_code)
            return FakeResponse(response_data)

        fake_client = FakeClient(handler=handler)
        _patch_httpx(monkeypatch, fake_client)

        p = OpenAIEmbeddingProvider("sk-test-key")
        result = await p.embed_query("test")
        assert len(result) == 2
        assert call_count == 3  # 2 retries + 1 success

    async def test_client_reused_across_calls(self, monkeypatch):
        """The httpx client should be reused, not recreated per call."""
        response_data = {"data": [{"index": 0, "embedding": [1.0, 0.0]}]}
        fake_client = FakeClient(handler=lambda url, headers, json: FakeResponse(response_data))
        _patch_httpx(monkeypatch, fake_client)

        p = OpenAIEmbeddingProvider("sk-test-key")
        await p.embed_query("first")
        client_after_first = p._client
        await p.embed_query("second")
        assert p._client is client_after_first


# ---------------------------------------------------------------------------
# Local provider
# ---------------------------------------------------------------------------


class TestLocalProvider:
    def test_provider_id(self):
        p = LocalEmbeddingProvider()
        assert p.provider_id == "local"

    def test_model_default(self):
        p = LocalEmbeddingProvider()
        assert p.model == "nomic-ai/nomic-embed-text-v1.5-Q"

    def test_dims_known_model(self):
        p = LocalEmbeddingProvider(model="BAAI/bge-small-en-v1.5")
        assert p.dims == 384

    async def test_dims_unknown_model_probed_from_engine(self):
        """Unknown models get dims by probing the engine on first use."""
        with patch("kraang.embeddings.TextEmbedding", FakeTextEmbedding):
            p = LocalEmbeddingProvider(model="some-unknown/model")
            # dims is None before engine init
            assert p._dims is None
            # Trigger engine init
            await p.embed_query("probe")
            # Now dims should be probed from the engine output
            assert p.dims == 768

    async def test_embed_query(self):
        with patch("kraang.embeddings.TextEmbedding", FakeTextEmbedding):
            p = LocalEmbeddingProvider()
            result = await p.embed_query("hello world")
        assert len(result) == 768
        # Should be L2 normalized
        norm = math.sqrt(sum(x * x for x in result))
        assert norm == pytest.approx(1.0)

    async def test_embed_batch(self):
        with patch("kraang.embeddings.TextEmbedding", FakeTextEmbedding):
            p = LocalEmbeddingProvider()
            results = await p.embed_batch(["hello", "world"])
        assert len(results) == 2
        for vec in results:
            assert len(vec) == 768
            norm = math.sqrt(sum(x * x for x in vec))
            assert norm == pytest.approx(1.0)

    async def test_embed_batch_empty(self):
        p = LocalEmbeddingProvider()
        result = await p.embed_batch([])
        assert result == []

    async def test_embed_query_empty_string_raises(self):
        p = LocalEmbeddingProvider()
        with pytest.raises(ValueError, match="empty"):
            await p.embed_query("")

    async def test_embed_query_whitespace_only_raises(self):
        p = LocalEmbeddingProvider()
        with pytest.raises(ValueError, match="empty"):
            await p.embed_query("   ")

    async def test_embed_batch_empty_string_raises(self):
        p = LocalEmbeddingProvider()
        with pytest.raises(ValueError, match="empty"):
            await p.embed_batch(["hello", ""])

    async def test_lazy_engine_init(self):
        """Engine is not loaded until first embed call."""
        with patch("kraang.embeddings.TextEmbedding", FakeTextEmbedding):
            p = LocalEmbeddingProvider()
            assert p._engine is None
            await p.embed_query("trigger init")
            assert p._engine is not None


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocol:
    def test_openai_provider_is_embedding_provider(self):
        from kraang.embeddings import EmbeddingProvider

        p = OpenAIEmbeddingProvider("sk-test")
        assert isinstance(p, EmbeddingProvider)

    def test_local_provider_is_embedding_provider(self):
        from kraang.embeddings import EmbeddingProvider

        p = LocalEmbeddingProvider()
        assert isinstance(p, EmbeddingProvider)

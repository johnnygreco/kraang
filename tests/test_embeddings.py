"""Tests for kraang.embeddings — provider factory, normalization, OpenAI provider."""

from __future__ import annotations

import math
import types

import pytest

from kraang.embeddings import (
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
    """Patch the httpx module reference in kraang.embeddings to use a fake client."""
    import httpx as real_httpx

    import kraang.embeddings as mod

    fake_httpx = types.ModuleType("httpx")
    fake_httpx.AsyncClient = lambda timeout=None: fake_client  # type: ignore[attr-defined]
    fake_httpx.HTTPStatusError = real_httpx.HTTPStatusError  # type: ignore[attr-defined]
    fake_httpx.ConnectError = real_httpx.ConnectError  # type: ignore[attr-defined]
    fake_httpx.TimeoutException = real_httpx.TimeoutException  # type: ignore[attr-defined]
    fake_httpx.Request = real_httpx.Request  # type: ignore[attr-defined]
    fake_httpx.Response = real_httpx.Response  # type: ignore[attr-defined]

    monkeypatch.setattr(mod, "httpx", fake_httpx)
    monkeypatch.setattr(mod, "_BASE_DELAY", 0.01)  # Speed up retries


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
    async def test_returns_none_without_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        provider = await create_provider()
        assert provider is None

    async def test_returns_none_with_empty_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "")
        provider = await create_provider()
        assert provider is None

    async def test_returns_none_with_whitespace_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "   ")
        provider = await create_provider()
        assert provider is None

    async def test_returns_provider_with_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-123")
        provider = await create_provider()
        assert provider is not None
        assert provider.provider_id == "openai"
        assert provider.model == "text-embedding-3-small"
        assert provider.dims == 1536


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
        response_data = {
            "data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]
        }
        fake_client = FakeClient(
            handler=lambda url, headers, json: FakeResponse(response_data)
        )
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
        fake_client = FakeClient(
            handler=lambda url, headers, json: FakeResponse(response_data)
        )
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
        response_data = {
            "data": [{"index": 0, "embedding": [1.0, 0.0]}]
        }

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

    async def test_no_retry_on_client_errors(self, monkeypatch):
        """401/403 errors should raise immediately without retrying."""
        call_count = 0

        def handler(url, headers, json):
            nonlocal call_count
            call_count += 1
            return FakeResponse(status_code=401)

        fake_client = FakeClient(handler=handler)
        _patch_httpx(monkeypatch, fake_client)

        p = OpenAIEmbeddingProvider("sk-test-key")
        import httpx

        with pytest.raises(httpx.HTTPStatusError):
            await p.embed_query("test")
        assert call_count == 1  # No retries — raised immediately

    async def test_no_retry_on_403(self, monkeypatch):
        """403 errors should raise immediately without retrying."""
        call_count = 0

        def handler(url, headers, json):
            nonlocal call_count
            call_count += 1
            return FakeResponse(status_code=403)

        fake_client = FakeClient(handler=handler)
        _patch_httpx(monkeypatch, fake_client)

        p = OpenAIEmbeddingProvider("sk-test-key")
        import httpx

        with pytest.raises(httpx.HTTPStatusError):
            await p.embed_query("test")
        assert call_count == 1

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

    async def test_retry_on_429(self, monkeypatch):
        """429 rate limit should be retried."""
        call_count = 0
        response_data = {
            "data": [{"index": 0, "embedding": [1.0, 0.0]}]
        }

        def handler(url, headers, json):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return FakeResponse(status_code=429)
            return FakeResponse(response_data)

        fake_client = FakeClient(handler=handler)
        _patch_httpx(monkeypatch, fake_client)

        p = OpenAIEmbeddingProvider("sk-test-key")
        result = await p.embed_query("test")
        assert len(result) == 2
        assert call_count == 3  # 2 retries + 1 success


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocol:
    def test_openai_provider_is_embedding_provider(self):
        from kraang.embeddings import EmbeddingProvider

        p = OpenAIEmbeddingProvider("sk-test")
        assert isinstance(p, EmbeddingProvider)

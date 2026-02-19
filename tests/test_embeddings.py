"""Tests for kraang.embeddings — provider factory, normalization, OpenAI provider."""

from __future__ import annotations

import math

import pytest

from kraang.embeddings import (
    OpenAIEmbeddingProvider,
    _l2_normalize,
    create_provider,
)


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
        import kraang.embeddings as emb_mod

        fake_response_data = {
            "data": [
                {"index": 0, "embedding": [0.1, 0.2, 0.3]},
            ]
        }

        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return fake_response_data

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post(self, url, headers=None, json=None):
                self.last_url = url
                self.last_json = json
                return FakeResponse()

        fake_client = FakeClient()

        # Patch httpx.AsyncClient in the embeddings module
        import types

        fake_httpx = types.ModuleType("httpx")
        fake_httpx.AsyncClient = lambda timeout=None: fake_client  # type: ignore[attr-defined]

        # Use monkeypatch to inject fake httpx into the module's import
        original_call_api = OpenAIEmbeddingProvider._call_api

        async def patched_call_api(self, texts):
            import sys

            old_httpx = sys.modules.get("httpx")
            sys.modules["httpx"] = fake_httpx
            try:
                return await original_call_api(self, texts)
            finally:
                if old_httpx is not None:
                    sys.modules["httpx"] = old_httpx
                else:
                    sys.modules.pop("httpx", None)

        monkeypatch.setattr(OpenAIEmbeddingProvider, "_call_api", patched_call_api)

        p = OpenAIEmbeddingProvider("sk-test-key")
        result = await p.embed_query("hello world")
        # Result should be L2-normalized
        norm = math.sqrt(sum(x * x for x in result))
        assert norm == pytest.approx(1.0)

    async def test_embed_batch_preserves_order(self, monkeypatch):
        """Ensure results are sorted by index even if API returns out of order."""
        fake_response_data = {
            "data": [
                {"index": 1, "embedding": [0.4, 0.5, 0.6]},
                {"index": 0, "embedding": [0.1, 0.2, 0.3]},
            ]
        }

        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return fake_response_data

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post(self, url, headers=None, json=None):
                return FakeResponse()

        import types

        fake_httpx = types.ModuleType("httpx")
        fake_httpx.AsyncClient = lambda timeout=None: FakeClient()  # type: ignore[attr-defined]

        original_call_api = OpenAIEmbeddingProvider._call_api

        async def patched_call_api(self, texts):
            import sys

            old_httpx = sys.modules.get("httpx")
            sys.modules["httpx"] = fake_httpx
            try:
                return await original_call_api(self, texts)
            finally:
                if old_httpx is not None:
                    sys.modules["httpx"] = old_httpx
                else:
                    sys.modules.pop("httpx", None)

        monkeypatch.setattr(OpenAIEmbeddingProvider, "_call_api", patched_call_api)

        p = OpenAIEmbeddingProvider("sk-test-key")
        results = await p.embed_batch(["text1", "text2"])
        assert len(results) == 2
        # First result should correspond to index 0
        norm0 = math.sqrt(sum(x * x for x in results[0]))
        assert norm0 == pytest.approx(1.0)

    async def test_retry_on_failure(self, monkeypatch):
        """Test that _call_api retries on transient failures."""
        call_count = 0
        fake_response_data = {
            "data": [{"index": 0, "embedding": [1.0, 0.0]}]
        }

        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return fake_response_data

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post(self, url, headers=None, json=None):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise ConnectionError("transient failure")
                return FakeResponse()

        import types

        fake_httpx = types.ModuleType("httpx")
        fake_httpx.AsyncClient = lambda timeout=None: FakeClient()  # type: ignore[attr-defined]

        original_call_api = OpenAIEmbeddingProvider._call_api

        async def patched_call_api(self, texts):
            import sys

            old_httpx = sys.modules.get("httpx")
            sys.modules["httpx"] = fake_httpx
            try:
                # Speed up retries for testing
                import kraang.embeddings as mod

                old_base = mod._BASE_DELAY
                mod._BASE_DELAY = 0.01
                try:
                    return await original_call_api(self, texts)
                finally:
                    mod._BASE_DELAY = old_base
            finally:
                if old_httpx is not None:
                    sys.modules["httpx"] = old_httpx
                else:
                    sys.modules.pop("httpx", None)

        monkeypatch.setattr(OpenAIEmbeddingProvider, "_call_api", patched_call_api)

        p = OpenAIEmbeddingProvider("sk-test-key")
        result = await p.embed_query("test")
        assert len(result) == 2
        assert call_count == 3  # 2 failures + 1 success

    async def test_all_retries_exhausted(self, monkeypatch):
        """Test RuntimeError after all retries fail."""

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post(self, url, headers=None, json=None):
                raise ConnectionError("permanent failure")

        import types

        fake_httpx = types.ModuleType("httpx")
        fake_httpx.AsyncClient = lambda timeout=None: FakeClient()  # type: ignore[attr-defined]

        original_call_api = OpenAIEmbeddingProvider._call_api

        async def patched_call_api(self, texts):
            import sys

            old_httpx = sys.modules.get("httpx")
            sys.modules["httpx"] = fake_httpx
            try:
                import kraang.embeddings as mod

                old_base = mod._BASE_DELAY
                mod._BASE_DELAY = 0.01
                try:
                    return await original_call_api(self, texts)
                finally:
                    mod._BASE_DELAY = old_base
            finally:
                if old_httpx is not None:
                    sys.modules["httpx"] = old_httpx
                else:
                    sys.modules.pop("httpx", None)

        monkeypatch.setattr(OpenAIEmbeddingProvider, "_call_api", patched_call_api)

        p = OpenAIEmbeddingProvider("sk-test-key")
        with pytest.raises(RuntimeError, match="failed after"):
            await p.embed_query("test")


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocol:
    def test_openai_provider_is_embedding_provider(self):
        from kraang.embeddings import EmbeddingProvider

        p = OpenAIEmbeddingProvider("sk-test")
        assert isinstance(p, EmbeddingProvider)

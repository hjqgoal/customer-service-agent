import io
import json
from urllib.error import HTTPError

import pytest
from pydantic import SecretStr

from customer_service_agent.config import Settings
from customer_service_agent.infrastructure import dashscope_embeddings as module
from customer_service_agent.infrastructure.dashscope_embeddings import (
    DashScopeEmbeddingError,
    DashScopeTextEmbeddings,
)
from customer_service_agent.model_factory import build_embedding_model


def test_native_embedding_batches_and_restores_response_order(monkeypatch):
    requests = []

    def respond(request, timeout):
        payload = json.loads(request.data)
        requests.append(payload)
        assert "/services/embeddings/multimodal-embedding/" in request.full_url
        entries = [
            {"index": i, "embedding": [float(item["text"])] * 1024}
            for i, item in enumerate(payload["input"]["contents"])
        ]
        return io.BytesIO(json.dumps({"output": {"embeddings": entries[::-1]}}).encode())

    monkeypatch.setattr(module, "urlopen", respond)
    model = DashScopeTextEmbeddings(SecretStr("test-key"))
    vectors = model.embed_documents([str(i) for i in range(1, 11)])
    assert [v[0] for v in vectors] == list(range(1, 11))
    assert [len(r["input"]["contents"]) for r in requests] == [8, 2]
    assert requests[0]["parameters"] == {"dimension": 1024, "enable_fusion": False}


def test_native_embedding_rejects_missing_indices(monkeypatch):
    data = {"output": {"embeddings": [{"index": 1, "embedding": [1.0] * 1024}]}}
    monkeypatch.setattr(module, "urlopen", lambda *a, **kw: io.BytesIO(json.dumps(data).encode()))
    with pytest.raises(DashScopeEmbeddingError, match="格式无效"):
        DashScopeTextEmbeddings(SecretStr("test-key")).embed_query("hello")


def test_native_embedding_auth_failure_is_not_retried_or_leaked(monkeypatch):
    calls = []

    def reject(*args, **kwargs):
        calls.append(1)
        raise HTTPError("https://example.invalid", 401, "secret-response", {}, None)

    monkeypatch.setattr(module, "urlopen", reject)
    with pytest.raises(DashScopeEmbeddingError) as caught:
        DashScopeTextEmbeddings(SecretStr("test-key")).embed_query("hello")
    assert str(caught.value) == "DashScope embedding HTTP 401"
    assert len(calls) == 1


def test_factory_uses_native_route_only_for_matching_official_provider():
    settings = Settings(
        _env_file=None,
        embedding_api_key="test-key",
        embedding_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        embedding_model_name="qwen3-vl-embedding",
    )
    assert isinstance(build_embedding_model(settings), DashScopeTextEmbeddings)
    custom = settings.model_copy(update={"embedding_base_url": "https://example.invalid/v1"})
    assert not isinstance(build_embedding_model(custom), DashScopeTextEmbeddings)

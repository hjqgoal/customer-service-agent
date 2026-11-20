import json

import httpx
import pytest

from customer_service_agent.models import RAGHit
from customer_service_agent.rag.reranker import (
    DashScopeReranker,
    RerankError,
    reciprocal_rank_fusion,
)
from customer_service_agent.rag.service import RAGService
from customer_service_agent.rag.sqlite_vector_store import VectorSearchHit
from customer_service_agent.tools.rag_tool import build_rag_search_tool
from tests.fakes import FakeEmbeddings


def configured(s, **updates):
    # model_copy不验证更新，因此明确构造SecretStr。
    from pydantic import SecretStr

    return s.model_copy(
        update={
            "rag_rerank_enabled": True,
            "rag_rerank_api_url": "https://rank.example.test/rerank",
            "rag_rerank_api_key": SecretStr("test-key"),
            "rag_rerank_model_name": "test-reranker",
            "rag_rerank_top_n": 2,
            **updates,
        }
    )


def hits():
    return [
        RAGHit(content="运费说明", score=0.8, metadata={"source": "shipping.md"}),
        RAGHit(content="退款规则：七天内申请", score=0.4, metadata={"source": "refund.md"}),
    ]


def test_real_http_contract_maps_indices_without_rewriting_evidence(test_settings):
    s = configured(test_settings)

    def handle(request):
        value = json.loads(request.content)
        assert value["input"]["documents"] == [h.content for h in hits()]
        assert value["parameters"]["top_n"] == 2
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(
            200,
            json={
                "output": {
                    "results": [
                        {"index": 0, "relevance_score": 0.1, "document": "恶意改写"},
                        {"index": 1, "relevance_score": 0.95},
                    ]
                }
            },
        )

    result = DashScopeReranker(s, transport=httpx.MockTransport(handle)).rerank("退款", hits())
    assert result[0].content == hits()[1].content
    assert result[0].score == 0.4
    assert result[0].metadata["rerank_score"] == 0.95
    assert result[0].metadata["source"] == "refund.md"


def test_jina_compatible_contract_uses_flat_payload_and_results(test_settings):
    s = configured(test_settings, rag_rerank_api_format="jina")

    def handle(request):
        value = json.loads(request.content)
        assert value == {
            "model": "test-reranker",
            "query": "退款",
            "documents": [h.content for h in hits()],
            "top_n": 2,
        }
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(
            200,
            json={
                "model": "test-reranker",
                "results": [
                    {"index": 1, "relevance_score": 0.9, "document": {"text": "恶意改写"}},
                    {"index": 0, "relevance_score": 0.2},
                ],
            },
        )

    result = DashScopeReranker(s, transport=httpx.MockTransport(handle)).rerank("退款", hits())
    assert [h.content for h in result] == [hits()[1].content, hits()[0].content]
    assert result[0].metadata["rerank_score"] == 0.9


def test_jina_format_rejects_dashscope_shaped_response(test_settings):
    s = configured(test_settings, rag_rerank_api_format="jina")
    response = {"output": {"results": [{"index": 0, "relevance_score": 0.5}] * 2}}
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=response))
    with pytest.raises(RerankError, match="rerank_invalid_response"):
        DashScopeReranker(s, transport=transport).rerank("退款", hits())


@pytest.mark.parametrize(
    "results",
    [
        [{"index": 0, "relevance_score": 0.8}] * 2,
        [{"index": 9, "relevance_score": 0.8}, {"index": 0, "relevance_score": 0.3}],
        [{"index": True, "relevance_score": 0.8}, {"index": 0, "relevance_score": 0.3}],
        [{"index": 1, "relevance_score": float("nan")}, {"index": 0, "relevance_score": 0.3}],
        [{"index": 1, "relevance_score": 1.2}, {"index": 0, "relevance_score": 0.3}],
        [],
        {},
    ],
)
def test_invalid_results_rejected(test_settings, results):
    # content允许测试NaN；HTTP响应验证必须拒绝非有限数字。
    transport = httpx.MockTransport(
        lambda r: httpx.Response(200, content=json.dumps({"output": {"results": results}}).encode())
    )
    with pytest.raises(RerankError, match="rerank_invalid_response"):
        DashScopeReranker(configured(test_settings), transport=transport).rerank("退款", hits())


@pytest.mark.parametrize(
    "mode,code", [("timeout", "rerank_timeout"), ("http", "rerank_http_error")]
)
def test_failures_are_sanitized(test_settings, mode, code):
    def fail(request):
        if mode == "timeout":
            raise httpx.ReadTimeout("secret-query-and-key")
        return httpx.Response(500, text="secret-query-and-key")

    with pytest.raises(RerankError) as error:
        DashScopeReranker(configured(test_settings), transport=httpx.MockTransport(fail)).rerank(
            "退款", hits()
        )
    assert str(error.value) == code


def test_document_limit_and_score_threshold(test_settings):
    s = configured(test_settings, rag_rerank_max_document_chars=100, rag_rerank_min_score=0.6)

    def response(request):
        assert len(json.loads(request.content)["input"]["documents"][0]) == 100
        return httpx.Response(
            200,
            json={
                "output": {
                    "results": [
                        {"index": 0, "relevance_score": 0.2},
                        {"index": 1, "relevance_score": 0.7},
                    ]
                }
            },
        )

    evidence = hits()
    evidence[0].content = "中" * 500
    result = DashScopeReranker(s, transport=httpx.MockTransport(response)).rerank("退款", evidence)
    assert len(result) == 1


def test_rrf_uses_ranks_not_incomparable_scores():
    evidence = hits()
    evidence[0].score = 999
    evidence[0].metadata["retrieval_ranks"] = {"keyword": 1}
    evidence[1].metadata["retrieval_ranks"] = {"keyword": 2, "semantic": 1}
    result = reciprocal_rank_fusion(evidence)
    assert result[0].content == evidence[1].content
    assert "rrf_score" not in evidence[0].metadata


def test_reranked_evidence_actually_enters_agent_tool(test_settings, monkeypatch):
    s = configured(test_settings, rag_rerank_top_n=1)
    transport = httpx.MockTransport(
        lambda r: httpx.Response(
            200, json={"output": {"results": [{"index": 1, "relevance_score": 0.9}]}}
        )
    )
    with RAGService(
        s, FakeEmbeddings(), reranker=DashScopeReranker(s, transport=transport)
    ) as service:
        candidates = [
            VectorSearchHit(content=h.content, score=h.score, metadata=h.metadata) for h in hits()
        ]
        monkeypatch.setattr(service.store, "keyword_search", lambda *a, **k: candidates)
        monkeypatch.setattr(service.store, "similarity_search", lambda *a, **k: candidates)
        evidence = json.loads(build_rag_search_tool(service).invoke({"query": "退款"}))
    assert evidence["ranking_method"] == "model" and evidence["rerank_status"] == "success"
    assert evidence["hits"][0]["content"] == hits()[1].content
    assert "keyword_hits" not in evidence  # 原始正文不重复占用LLM上下文。


def test_missing_rerank_config_falls_back_without_web_call(test_settings, monkeypatch):
    s = test_settings.model_copy(update={"rag_rerank_enabled": True})
    with RAGService(s, FakeEmbeddings()) as service:
        candidates = [
            VectorSearchHit(content=h.content, score=h.score, metadata=h.metadata) for h in hits()
        ]
        monkeypatch.setattr(service.store, "keyword_search", lambda *a, **k: candidates)
        monkeypatch.setattr(service.store, "similarity_search", lambda *a, **k: candidates)
        result = service.retrieve("退款")
    assert result.hits and result.rerank_status == "not_configured"
    assert result.ranking_method == "rrf" and not result.web_attempted
    assert result.warnings == ["rerank_not_configured"]


@pytest.mark.parametrize(
    "url",
    [
        "http://rank.test/rerank",
        "https://key@rank.test/rerank",
        "https://rank.test/rerank?key=secret",
    ],
)
def test_unsafe_endpoint_configuration_is_rejected(test_settings, url):
    s = configured(test_settings, rag_rerank_api_url=url)
    with pytest.raises(RerankError, match="rerank_invalid_url"):
        DashScopeReranker(s).rerank("退款", hits())


def test_rerank_http_failure_keeps_rrf_evidence(test_settings, monkeypatch):
    s = configured(test_settings)
    ranker = DashScopeReranker(s, transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    with RAGService(s, FakeEmbeddings(), reranker=ranker) as service:
        candidates = [
            VectorSearchHit(content=h.content, score=h.score, metadata=h.metadata) for h in hits()
        ]
        monkeypatch.setattr(service.store, "keyword_search", lambda *a, **k: candidates)
        monkeypatch.setattr(service.store, "similarity_search", lambda *a, **k: candidates)
        report = service.retrieve("退款")
    assert report.hits and report.ranking_method == "rrf"
    assert report.rerank_status == "failed" and report.warnings == ["rerank_http_error"]


def test_api_exposes_final_order_and_ranking_diagnostics(test_settings, monkeypatch):
    from fastapi.testclient import TestClient

    from customer_service_agent.api import app, get_rag_service
    from customer_service_agent.config import get_settings

    s = configured(test_settings, rag_rerank_top_n=1)
    transport = httpx.MockTransport(
        lambda r: httpx.Response(
            200, json={"output": {"results": [{"index": 1, "relevance_score": 0.9}]}}
        )
    )
    with RAGService(
        s, FakeEmbeddings(), reranker=DashScopeReranker(s, transport=transport)
    ) as service:
        candidates = [
            VectorSearchHit(content=h.content, score=h.score, metadata=h.metadata) for h in hits()
        ]
        monkeypatch.setattr(service.store, "keyword_search", lambda *a, **k: candidates)
        monkeypatch.setattr(service.store, "similarity_search", lambda *a, **k: candidates)
        app.dependency_overrides[get_settings] = lambda: s
        app.dependency_overrides[get_rag_service] = lambda: service
        try:
            with TestClient(app) as client:
                response = client.post("/rag/search", json={"query": "退款"})
            value = response.json()
            assert response.status_code == 200
            assert value["rerank_status"] == "success"
            assert value["ranking_method"] == "model"
            assert value["hits"][0]["content"] == hits()[1].content
            assert len(value["keyword_hits"]) == 2
        finally:
            app.dependency_overrides.clear()

"""检索合同、并发、旧库迁移、MCP降级和工具到模型的集成验证。"""

import asyncio
import json
import sqlite3
import sys
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from customer_service_agent.agent import build_customer_service_agent
from customer_service_agent.api import app, get_rag_service
from customer_service_agent.config import get_settings
from customer_service_agent.models import RAGHit
from customer_service_agent.rag.service import RAGService
from customer_service_agent.rag.sqlite_vector_store import SQLiteVectorStore, VectorSearchHit
from customer_service_agent.rag.web_fallback import MCPBrowserSearch, WebSearchError, search_url
from customer_service_agent.tools.rag_tool import build_rag_search_tool
from tests.fakes import FakeEmbeddings, FakeToolCallingChatModel


class StubWeb:
    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def search(self, query):
        self.calls.append(query)
        if self.error:
            raise self.error
        return [
            RAGHit(
                content="公开网页测试证据",
                score=0,
                metadata={"source": "https://example.test/public", "retrieval_methods": ["web"]},
            )
        ]


def raw(content, score=0.8):
    return VectorSearchHit(content=content, score=score, metadata={"source": "faq.md"})


def test_both_branches_really_execute_in_parallel(test_settings, monkeypatch):
    service = RAGService(test_settings, FakeEmbeddings(), web_search=StubWeb())
    barrier = threading.Barrier(2, timeout=3)

    def retrieve(query, **kwargs):
        barrier.wait()  # 若两个分支串行运行，这里会失败，而不是仅依赖计时断言。
        return [raw("退款片段")]

    monkeypatch.setattr(service.store, "keyword_search", retrieve)
    monkeypatch.setattr(service.store, "similarity_search", retrieve)
    result = service.retrieve("退款")
    assert len(result.keyword_hits) == len(result.semantic_hits) == 1
    assert not result.warnings and not result.web_attempted
    assert len(result.hits) == 1
    assert result.hits[0].metadata["retrieval_methods"] == ["keyword", "semantic"]


def test_each_branch_gets_three_and_llm_receives_both(test_settings, monkeypatch):
    service = RAGService(test_settings, FakeEmbeddings(), web_search=StubWeb())
    observed = []

    def candidates(query, *, top_k, **kwargs):
        observed.append(top_k)
        return [raw(f"知识片段{i}") for i in range(top_k)]

    monkeypatch.setattr(service.store, "keyword_search", candidates)
    monkeypatch.setattr(service.store, "similarity_search", candidates)
    value = json.loads(build_rag_search_tool(service).invoke({"query": "退款", "top_k": 10}))
    assert observed == [3, 3]
    assert len(value["keyword_hits"]) == len(value["semantic_hits"]) == 3
    assert value["keyword_hits"][0]["metadata"]["score_type"] == "bm25"
    assert value["semantic_hits"][0]["metadata"]["score_type"] == "cosine"


def test_sqlite_chinese_bm25_and_atomic_replacement(test_settings):
    store = SQLiteVectorStore(test_settings.rag_db_file, FakeEmbeddings())
    store.replace_documents(
        [
            Document(
                page_content="退款条件：商品完好，七天内申请退款。",
                metadata={"source": "refund.md"},
            ),
            Document(
                page_content="物流查询：发货后可查看快递轨迹。", metadata={"source": "logistics.md"}
            ),
        ]
    )
    hits = store.keyword_search("退款怎么申请", top_k=3)
    assert hits and hits[0].metadata["source"] == "refund.md"
    assert not store.keyword_search("终身保修")
    assert store.keyword_search('退款 OR " * )')
    store.replace_documents(
        [Document(page_content="电子发票开具说明", metadata={"source": "invoice.md"})]
    )
    assert not store.keyword_search("退款")
    store.clear()
    assert not store.keyword_search("发票")


def test_old_index_gets_keyword_index_without_embedding_calls(test_settings):
    with sqlite3.connect(test_settings.rag_db_file) as connection:
        connection.execute(
            "CREATE TABLE rag_chunks (id INTEGER PRIMARY KEY, content TEXT NOT NULL, "
            "metadata_json TEXT NOT NULL, embedding_json TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO rag_chunks VALUES (1, ?, ?, ?)",
            ("退款需要提交材料", '{"source":"legacy.md"}', "[1,0]"),
        )
    store = SQLiteVectorStore(test_settings.rag_db_file, None)
    hit = store.keyword_search("退款")[0]
    assert hit.metadata["source"] == "legacy.md"
    with sqlite3.connect(test_settings.rag_db_file) as connection:
        assert connection.execute("SELECT embedding_json FROM rag_chunks").fetchone()[0] == "[1,0]"


def test_failed_semantic_branch_preserves_keyword_and_uses_web(test_settings, monkeypatch):
    web = StubWeb()
    service = RAGService(test_settings, FakeEmbeddings(), web_search=web)
    monkeypatch.setattr(service.store, "keyword_search", lambda *a, **kw: [raw("本地退款规则")])

    def fail(*args, **kwargs):
        raise RuntimeError("do-not-expose-secret-or-provider-url")

    monkeypatch.setattr(service.store, "similarity_search", fail)
    result = service.retrieve("公开退货流程")
    assert result.keyword_hits and result.web_hits and not result.semantic_hits
    assert web.calls == ["公开退货流程"]
    assert result.warnings == ["semantic_unavailable:RuntimeError"]
    assert "do-not-expose" not in result.model_dump_json()


def test_empty_local_index_goes_to_web(test_settings):
    web = StubWeb()
    result = RAGService(test_settings, None, web_search=web).retrieve("公开保修说明")
    assert result.web_attempted and result.web_hits
    assert web.calls == ["公开保修说明"]


def test_web_failure_preserves_local_results(test_settings, monkeypatch):
    service = RAGService(test_settings, None, web_search=StubWeb(WebSearchError("web_timeout")))
    monkeypatch.setattr(service.store, "keyword_search", lambda *a, **kw: [raw("退款条件")])
    monkeypatch.setattr(service.store, "similarity_search", lambda *a, **kw: [])
    result = service.retrieve("退款", force_web=True)
    assert result.hits and result.keyword_hits and not result.web_hits
    assert "web_timeout" in result.warnings


def test_missing_web_configuration_is_explicit(test_settings):
    result = RAGService(test_settings, None).retrieve("公开保修说明")
    assert not result.hits
    assert result.warnings == ["web_mcp_not_configured"]


def test_disabled_web_and_query_validation(test_settings):
    web = StubWeb()
    settings = test_settings.model_copy(update={"rag_web_fallback_enabled": False})
    service = RAGService(settings, None, web_search=web)
    assert service.retrieve("保修").warnings == ["web_disabled"]
    assert not web.calls
    for query, k in [(" ", 3), ("退" * 2001, 3), ("退款", 0), ("退款", 21)]:
        with pytest.raises(ValueError):
            service.retrieve(query, top_k=k)


@pytest.mark.parametrize(
    "query",
    [
        "查EVAL-ORDER-1001",
        "订单123456789在哪",
        "电话13812345678",
        "a@example.com退款",
        "我的订单退款",
    ],
)
def test_private_queries_never_reach_browser(test_settings, query, monkeypatch):
    client = MCPBrowserSearch(
        test_settings.model_copy(update={"rag_web_mcp_command": "test-command"})
    )
    called = []

    async def snapshot(url):
        called.append(url)
        return "unexpected"

    monkeypatch.setattr(client, "_snapshot", snapshot)
    with pytest.raises(WebSearchError, match="public_terms"):
        client.search(query)
    assert not called


@pytest.mark.parametrize(
    "url", ["file:///tmp/a", "http://www.bing.com", "https://127.0.0.1", "https://localhost/search"]
)
def test_browser_endpoint_is_fixed_public_search(url):
    with pytest.raises(WebSearchError, match="invalid"):
        search_url(url, "公开政策")


def test_query_encoding_and_no_shell_interpolation():
    url = search_url("https://www.bing.com/search?setlang=zh&q=old", "退款 & 保修")
    assert "setlang=zh" in url and "q=old" not in url
    assert "%26" in url


def test_force_web_keeps_local_both_branches(test_settings, monkeypatch):
    web = StubWeb()
    service = RAGService(test_settings, FakeEmbeddings(), web_search=web)
    for name in ("keyword_search", "similarity_search"):
        monkeypatch.setattr(service.store, name, lambda *a, **kw: [raw("退款规则")])
    assert not service.retrieve("退款").web_attempted
    forced = service.retrieve("退款", force_web=True)
    assert forced.keyword_hits and forced.semantic_hits and forced.web_hits


def test_web_snapshot_errors_and_length_limit(test_settings, monkeypatch):
    client = MCPBrowserSearch(
        test_settings.model_copy(update={"rag_web_mcp_command": "test", "rag_web_max_chars": 1000})
    )

    async def snapshot(url):
        return "公开内容" * 1000

    monkeypatch.setattr(client, "_snapshot", snapshot)
    hit = client.search("公开政策")[0]
    assert len(hit.content) == 1000
    assert hit.metadata["score_type"] == "not_scored"
    assert not hit.metadata["verified_shop_policy"]

    async def verification(url):
        return "Verify you are human"

    monkeypatch.setattr(client, "_snapshot", verification)
    with pytest.raises(WebSearchError, match="verification_required"):
        client.search("公开政策")


def test_native_async_rag_tool(test_settings):
    service = RAGService(test_settings, None, web_search=StubWeb())
    value = asyncio.run(build_rag_search_tool(service).ainvoke({"query": "公开售后流程"}))
    assert json.loads(value)["web_attempted"]


def test_real_mcp_stdio_round_trip(test_settings):
    pytest.importorskip("mcp")
    settings = test_settings.model_copy(
        update={
            "rag_web_mcp_command": sys.executable,
            "rag_web_mcp_args": [str(Path(__file__).with_name("mcp_browser_fixture.py"))],
            "rag_web_timeout_seconds": 10,
        }
    )
    result = RAGService(settings, None).retrieve("SQLite FTS5 BM25 documentation")
    assert result.web_hits and not result.warnings
    assert "SQLite FTS5 BM25 documentation" in result.web_hits[0].content
    assert "https://www.sqlite.org/fts5.html" in result.web_hits[0].content


class EvidenceReadingModel(FakeToolCallingChatModel):
    def _generate(self, messages, **kwargs):
        if isinstance(messages[-1], ToolMessage):
            evidence = json.loads(messages[-1].content)
            assert evidence["keyword_hits"] and evidence["semantic_hits"]
            message = AIMessage(content="模型已收到两路证据，可申请需审核。")
        else:
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_knowledge_base",
                        "args": {"query": "退款"},
                        "id": "hybrid-search",
                        "type": "tool_call",
                    }
                ],
            )
        return ChatResult(generations=[ChatGeneration(message=message)])


def test_agent_model_receives_keyword_and_semantic_evidence(test_settings):
    service = RAGService(test_settings, FakeEmbeddings(), web_search=StubWeb())
    service.store.replace_documents(
        [Document(page_content="退款规则：退款需审核。", metadata={"source": "refund.md"})]
    )
    with build_customer_service_agent(
        test_settings, llm=EvidenceReadingModel(), tools=[build_rag_search_tool(service)]
    ) as agent:
        result = agent.ask("退款规则", include_trace=True)
    assert result.answer == "模型已收到两路证据，可申请需审核。"
    assert result.intermediate_steps[0].tool == "search_knowledge_base"


def test_api_returns_grouped_candidates_and_web_diagnostics(test_settings):
    service = RAGService(test_settings, FakeEmbeddings(), web_search=StubWeb())
    service.store.replace_documents(
        [Document(page_content="退款需审核", metadata={"source": "faq.md"})]
    )
    app.dependency_overrides[get_settings] = lambda: test_settings
    app.dependency_overrides[get_rag_service] = lambda: service
    try:
        with TestClient(app) as client:
            response = client.post("/rag/search", json={"query": "退款", "force_web": True})
        value = response.json()
        assert response.status_code == 200
        assert value["keyword_hits"] and value["semantic_hits"] and value["web_hits"]
        assert value["web_attempted"] and not value["warnings"]
    finally:
        app.dependency_overrides.clear()


def test_missing_embedding_config_preserves_lexical_search(test_settings):
    RAGService(test_settings, FakeEmbeddings()).store.replace_documents(
        [Document(page_content="退款需要审核", metadata={"source": "faq.md"})]
    )
    settings = test_settings.model_copy(update={"embedding_api_key": None, "openai_api_key": None})
    result = RAGService.from_settings(settings).retrieve("退款")
    assert result.keyword_hits and not result.semantic_hits
    assert "semantic_unavailable:RAGIndexError" in result.warnings


@pytest.mark.parametrize(
    "mode,expected", [("error", "web_snapshot_failed"), ("empty", "web_no_evidence")]
)
def test_mcp_protocol_failure_has_explicit_diagnostic(test_settings, mode, expected):
    pytest.importorskip("mcp")
    settings = test_settings.model_copy(
        update={
            "rag_web_mcp_command": sys.executable,
            "rag_web_mcp_args": [str(Path(__file__).with_name("mcp_browser_fixture.py")), mode],
            "rag_web_timeout_seconds": 10,
        }
    )
    result = RAGService(settings, None).retrieve("公开服务说明")
    assert not result.hits and expected in result.warnings


def test_semantic_timeout_returns_keyword_and_web_without_waiting(test_settings, monkeypatch):
    settings = test_settings.model_copy(update={"rag_local_timeout_seconds": 0.1})
    web = StubWeb()
    service = RAGService(settings, None, web_search=web)
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def delayed(*args, **kwargs):
        started.set()
        release.wait(3)
        finished.set()
        return [raw("迟到的语义证据")]

    monkeypatch.setattr(service.store, "keyword_search", lambda *a, **kw: [raw("本地退款规则")])
    monkeypatch.setattr(service.store, "similarity_search", delayed)
    try:
        result = service.retrieve("公开退款流程")
        assert started.is_set() and not finished.is_set()
        assert result.keyword_hits and result.web_hits and not result.semantic_hits
        assert "semantic_unavailable:TimeoutError" in result.warnings
        assert service._executor._max_workers == 4
    finally:
        release.set()
        service.close()
        assert finished.wait(3)

import pytest

from customer_service_agent.rag.service import RAGService
from tests.fakes import FailingEmbeddings, FakeEmbeddings


def _write_faq(test_settings) -> None:
    (test_settings.knowledge_base_path / "faq.md").write_text(
        "# FAQ\n\n"
        "退款规则：普通商品签收后 7 天内可以申请无理由退款。\n\n"
        "物流规则：发货后可查物流。",
        encoding="utf-8",
    )


def test_rag_ingest_and_search(test_settings):
    _write_faq(test_settings)
    service = RAGService(test_settings, FakeEmbeddings())

    ingest_result = service.ingest_directory()
    hits = service.search("退款怎么处理", top_k=2)

    assert ingest_result.loaded_documents == 1
    assert ingest_result.indexed_chunks >= 1
    assert hits
    assert "退款" in hits[0].content
    assert hits[0].metadata["source"] == "faq.md"


def test_rag_append_deduplicates_chunks(test_settings):
    _write_faq(test_settings)
    service = RAGService(test_settings, FakeEmbeddings())
    first = service.ingest_directory(replace=True)
    second = service.ingest_directory(replace=False)

    assert second.indexed_chunks == 0
    assert second.total_chunks == first.total_chunks


def test_failed_reindex_preserves_previous_index(test_settings):
    _write_faq(test_settings)
    working_service = RAGService(test_settings, FakeEmbeddings())
    original = working_service.ingest_directory(replace=True)
    failing_service = RAGService(test_settings, FailingEmbeddings())

    with pytest.raises(RuntimeError, match="embedding service unavailable"):
        failing_service.ingest_directory(replace=True)

    assert working_service.store.count() == original.total_chunks

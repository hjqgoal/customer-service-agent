"""RAG application service."""

from pathlib import Path

from langchain_core.embeddings import Embeddings

from customer_service_agent.config import Settings
from customer_service_agent.models import RAGHit, RAGIngestResponse
from customer_service_agent.rag.loader import load_documents, split_documents
from customer_service_agent.rag.sqlite_vector_store import SQLiteVectorStore


class RAGService:
    def __init__(self, settings: Settings, embeddings: Embeddings) -> None:
        self.settings = settings
        self.store = SQLiteVectorStore(
            settings.rag_db_file,
            embeddings,
            busy_timeout_seconds=settings.sqlite_busy_timeout_seconds,
        )

    def ingest_directory(
        self,
        directory: str | Path | None = None,
        *,
        replace: bool = True,
    ) -> RAGIngestResponse:
        target_dir = Path(directory) if directory else self.settings.knowledge_base_path
        documents = load_documents(target_dir)
        if not documents:
            raise ValueError(f"知识库目录中没有可加载的文档：{target_dir}")

        chunks = split_documents(
            documents,
            chunk_size=self.settings.rag_chunk_size,
            chunk_overlap=self.settings.rag_chunk_overlap,
        )
        indexed = (
            self.store.replace_documents(chunks) if replace else self.store.add_documents(chunks)
        )
        return RAGIngestResponse(
            loaded_documents=len(documents),
            indexed_chunks=indexed,
            total_chunks=self.store.count(),
        )

    def search(self, query: str, *, top_k: int | None = None) -> list[RAGHit]:
        requested_top_k = top_k or self.settings.rag_top_k
        if requested_top_k > 20:
            raise ValueError("top_k 不能超过 20。")
        hits = self.store.similarity_search(
            query,
            top_k=requested_top_k,
            min_score=self.settings.rag_min_score,
        )
        return [RAGHit(content=hit.content, score=hit.score, metadata=hit.metadata) for hit in hits]

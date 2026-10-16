"""Educational SQLite vector store with transactional index replacement."""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from customer_service_agent.infrastructure.sqlite import sqlite_connection
from customer_service_agent.rag.tokenizer import match_expression, tokenize


class RAGIndexError(RuntimeError):
    """Raised when the persisted index is incompatible with the current embedding model."""


@dataclass(frozen=True)
class VectorSearchHit:
    content: str
    score: float
    metadata: dict[str, Any]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


class SQLiteVectorStore:
    """Small local vector store for learning and development.

    Search is O(n) because vectors are stored as JSON and scanned in Python. Use
    pgvector, Qdrant, Milvus, or another vector database for a large production corpus.
    """

    def __init__(
        self,
        db_path: str | Path,
        embeddings: Embeddings | None,
        *,
        busy_timeout_seconds: float = 30.0,
    ) -> None:
        self.db_path = Path(db_path)
        self.embeddings = embeddings
        self.busy_timeout_seconds = busy_timeout_seconds
        self._ensure_schema()

    def _connection(self):
        return sqlite_connection(
            self.db_path,
            busy_timeout_seconds=self.busy_timeout_seconds,
        )

    def _ensure_schema(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    content_hash TEXT,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(rag_chunks)").fetchall()
            }
            if "content_hash" not in columns:
                connection.execute("ALTER TABLE rag_chunks ADD COLUMN content_hash TEXT")
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_rag_chunks_content_hash
                ON rag_chunks(content_hash)
                WHERE content_hash IS NOT NULL
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_index_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            # 与向量共用chunk ID；首次打开旧库只补关键词索引，不重算向量。
            connection.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS rag_keyword_chunks USING fts5(tokens)"
            )
            connection.execute("""
                CREATE TRIGGER IF NOT EXISTS rag_keyword_delete AFTER DELETE ON rag_chunks
                BEGIN DELETE FROM rag_keyword_chunks WHERE rowid = old.id; END
            """)
            connection.execute("""
                CREATE TRIGGER IF NOT EXISTS rag_keyword_update AFTER UPDATE ON rag_chunks
                BEGIN DELETE FROM rag_keyword_chunks WHERE rowid = old.id; END
            """)
            self._sync_keyword_index(connection)

    @staticmethod
    def _sync_keyword_index(connection) -> None:
        rows = connection.execute("""
            SELECT c.id, c.content FROM rag_chunks AS c
            LEFT JOIN rag_keyword_chunks AS k ON k.rowid = c.id WHERE k.rowid IS NULL
        """).fetchall()
        connection.executemany(
            "INSERT INTO rag_keyword_chunks(rowid, tokens) VALUES (?, ?)",
            [(chunk_id, " ".join(tokenize(content))) for chunk_id, content in rows],
        )

    @staticmethod
    def _content_hash(document: Document) -> str:
        payload = json.dumps(
            {"content": document.page_content, "metadata": document.metadata},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _prepare_rows(self, documents: list[Document]) -> list[tuple[str, str, str, str]]:
        if not documents:
            return []
        if self.embeddings is None:
            raise RAGIndexError("Embedding未配置，无法构建向量索引；请先填写EMBEDDING_*。")
        vectors = self.embeddings.embed_documents([document.page_content for document in documents])
        if len(vectors) != len(documents):
            raise RAGIndexError("Embedding 服务返回的向量数量与文档数量不一致。")
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1 or 0 in dimensions:
            raise RAGIndexError("Embedding 服务返回了空向量或不一致的向量维度。")

        return [
            (
                document.page_content,
                json.dumps(document.metadata, ensure_ascii=False, sort_keys=True),
                json.dumps(vector, separators=(",", ":")),
                self._content_hash(document),
            )
            for document, vector in zip(documents, vectors, strict=True)
        ]

    @staticmethod
    def _write_index_metadata(connection: Any, rows: list[tuple[str, str, str, str]]) -> None:
        if not rows:
            return
        vector_dimension = len(json.loads(rows[0][2]))
        connection.execute(
            """
            INSERT INTO rag_index_metadata(key, value) VALUES ('vector_dimension', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (str(vector_dimension),),
        )

    def clear(self) -> None:
        with self._connection() as connection:
            connection.execute("DELETE FROM rag_chunks")
            connection.execute("DELETE FROM rag_index_metadata")

    def count(self) -> int:
        with self._connection() as connection:
            row = connection.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()
        return int(row[0])

    def add_documents(self, documents: list[Document]) -> int:
        rows = self._prepare_rows(documents)
        if not rows:
            return 0
        with self._connection() as connection:
            before = int(connection.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0])
            connection.executemany(
                """
                INSERT OR IGNORE INTO rag_chunks(
                    content, metadata_json, embedding_json, content_hash
                ) VALUES (?, ?, ?, ?)
                """,
                rows,
            )
            self._write_index_metadata(connection, rows)
            self._sync_keyword_index(connection)
            after = int(connection.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0])
            return after - before

    def replace_documents(self, documents: list[Document]) -> int:
        """Embed first, then atomically replace the persisted index."""

        rows = self._prepare_rows(documents)
        if not rows:
            raise ValueError("没有可写入的知识库文档，拒绝清空现有索引。")
        with self._connection() as connection:
            connection.execute("DELETE FROM rag_chunks")
            connection.execute("DELETE FROM rag_index_metadata")
            connection.executemany(
                """
                INSERT INTO rag_chunks(content, metadata_json, embedding_json, content_hash)
                VALUES (?, ?, ?, ?)
                """,
                rows,
            )
            self._write_index_metadata(connection, rows)
            self._sync_keyword_index(connection)
        return len(rows)

    def keyword_search(self, query: str, *, top_k: int = 3) -> list[VectorSearchHit]:
        if not query.strip() or not 1 <= top_k <= 20:
            raise ValueError("查询不能为空，top_k必须为1到20。")
        expression = match_expression(query)
        if not expression:
            return []
        with self._connection() as connection:
            self._sync_keyword_index(connection)
            rows = connection.execute(
                """
                SELECT c.id, c.content, c.metadata_json, bm25(rag_keyword_chunks)
                FROM rag_keyword_chunks JOIN rag_chunks AS c
                ON c.id = rag_keyword_chunks.rowid
                WHERE rag_keyword_chunks MATCH ?
                ORDER BY bm25(rag_keyword_chunks), c.id LIMIT ?
            """,
                (expression, top_k),
            ).fetchall()
        # SQLite BM25越小越好且通常为负；取负号便于展示，不能与余弦分数比较。
        return [
            VectorSearchHit(
                content=content,
                score=-rank,
                metadata={**json.loads(metadata), "chunk_id": chunk_id},
            )
            for chunk_id, content, metadata, rank in rows
        ]

    def similarity_search(
        self,
        query: str,
        *,
        top_k: int = 4,
        min_score: float = 0.0,
    ) -> list[VectorSearchHit]:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("RAG 查询不能为空。")
        if top_k < 1:
            raise ValueError("top_k 必须大于 0。")

        with self._connection() as connection:
            rows = connection.execute(
                "SELECT id, content, metadata_json, embedding_json FROM rag_chunks"
            ).fetchall()
            dimension_row = connection.execute(
                "SELECT value FROM rag_index_metadata WHERE key = 'vector_dimension'"
            ).fetchone()
        if not rows:
            return []
        if self.embeddings is None:
            raise RAGIndexError("Embedding未配置，语义检索不可用。")
        query_vector = self.embeddings.embed_query(normalized_query)
        expected_dimension = int(dimension_row[0]) if dimension_row else len(json.loads(rows[0][3]))
        if len(query_vector) != expected_dimension:
            raise RAGIndexError(
                "当前 embedding 模型的向量维度与已有索引不一致，请重新执行 rag-ingest。"
            )

        scored = [
            VectorSearchHit(
                content=content,
                score=cosine_similarity(query_vector, json.loads(embedding_json)),
                metadata={**json.loads(metadata_json), "chunk_id": chunk_id},
            )
            for chunk_id, content, metadata_json, embedding_json in rows
        ]
        return [
            hit
            for hit in sorted(scored, key=lambda item: item.score, reverse=True)
            if hit.score >= min_score
        ][:top_k]

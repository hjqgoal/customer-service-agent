"""BM25与向量并行，各取Top3；缺少本地证据时通过浏览器MCP补充。

这里收集证据；LangChain Agent收到工具结果后判断相关性、冲突和引用。
网页资料不能代表本店政策或私人订单事实。
"""

import asyncio
import hashlib
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from typing import Protocol

from langchain_core.embeddings import Embeddings

from customer_service_agent.config import Settings
from customer_service_agent.model_factory import ModelConfigError, build_embedding_model
from customer_service_agent.models import RAGHit, RAGIngestResponse, RAGSearchResponse
from customer_service_agent.rag.loader import load_documents, split_documents
from customer_service_agent.rag.sqlite_vector_store import SQLiteVectorStore
from customer_service_agent.rag.web_fallback import MCPBrowserSearch, WebSearchError


class WebSearch(Protocol):
    def search(self, query: str) -> list[RAGHit]: ...


def merge_hits(groups: list[list[RAGHit]]) -> list[RAGHit]:
    """合并重复chunk，保留两条检索渠道各自的排名和分数。"""
    merged: dict[str, RAGHit] = {}
    for hits in groups:
        for hit in hits:
            payload = str(hit.metadata.get("source", "")) + "\0" + hit.content
            key = hashlib.sha256(payload.encode()).hexdigest()
            if key not in merged:
                merged[key] = hit.model_copy(deep=True)
            else:
                metadata = merged[key].metadata
                for name in ("retrieval_scores", "retrieval_ranks"):
                    metadata[name].update(hit.metadata[name])
                metadata["retrieval_methods"] = list(metadata["retrieval_scores"])
    return list(merged.values())


class RAGService:
    def __init__(
        self,
        settings: Settings,
        embeddings: Embeddings | None,
        *,
        web_search: WebSearch | None = None,
    ) -> None:
        self.settings = settings
        self.store = SQLiteVectorStore(
            settings.rag_db_file,
            embeddings,
            busy_timeout_seconds=settings.sqlite_busy_timeout_seconds,
        )
        self.web_search = web_search or MCPBrowserSearch(settings)
        self._executor = ThreadPoolExecutor(
            max_workers=settings.rag_retrieval_workers, thread_name_prefix="rag-retrieval"
        )
        self._closed = False

    @classmethod
    def from_settings(cls, settings: Settings) -> "RAGService":
        # 配置缺失不阻止关键词和网页路径运行；入库仍必须有真正的embedding。
        try:
            embeddings = build_embedding_model(settings)
        except ModelConfigError:
            embeddings = None
        return cls(settings, embeddings)

    def ingest_directory(
        self, directory: str | Path | None = None, *, replace: bool = True
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
            loaded_documents=len(documents), indexed_chunks=indexed, total_chunks=self.store.count()
        )

    def retrieve(
        self, query: str, *, top_k: int | None = None, force_web: bool = False
    ) -> RAGSearchResponse:
        if self._closed:
            raise RuntimeError("RAG服务已关闭。")
        query = query.strip()
        if not query or len(query) > 2000:
            raise ValueError("RAG查询必须为1到2000个字符。")
        requested = self.settings.rag_top_k if top_k is None else top_k
        if not 1 <= requested <= 20:
            raise ValueError("top_k必须为1到20。")
        branch_k = min(requested, self.settings.rag_branch_top_k)
        report = RAGSearchResponse(query=query, hits=[])
        # 两个任务先一起提交，再取结果；关键词不必等embedding完成后才开始。
        jobs = {
            "keyword": self._executor.submit(self.store.keyword_search, query, top_k=branch_k),
            "semantic": self._executor.submit(
                self.store.similarity_search,
                query,
                top_k=branch_k,
                min_score=max(self.settings.rag_min_score, self.settings.rag_semantic_min_score),
            ),
        }
        completed, _ = wait(jobs.values(), timeout=self.settings.rag_local_timeout_seconds)
        for method, future in jobs.items():
            if future not in completed:
                future.cancel()  # 已运行的线程不能强杀；仍受模型客户端超时约束。
                report.warnings.append(f"{method}_unavailable:TimeoutError")
                continue
            try:
                raw_hits = future.result()
                hits = [
                    RAGHit(
                        content=h.content,
                        score=h.score,
                        metadata={
                            **h.metadata,
                            "retrieval_methods": [method],
                            "retrieval_scores": {method: h.score},
                            "retrieval_ranks": {method: rank},
                            "score_type": "bm25" if method == "keyword" else "cosine",
                        },
                    )
                    for rank, h in enumerate(raw_hits, 1)
                ]
                setattr(report, f"{method}_hits", hits)
            except Exception as exc:
                # 不把供应商返回的原始异常、URL或密钥发给模型。
                report.warnings.append(f"{method}_unavailable:{type(exc).__name__}")
        local_hits = merge_hits([report.keyword_hits, report.semantic_hits])
        if force_web or not local_hits or report.warnings:
            if self.settings.rag_web_fallback_enabled:
                report.web_attempted = True
                try:
                    report.web_hits = self.web_search.search(query)
                    if not report.web_hits:
                        report.warnings.append("web_no_evidence")
                except WebSearchError as exc:
                    report.warnings.append(exc.code)
                except Exception as exc:
                    report.warnings.append(f"web_unavailable:{type(exc).__name__}")
            else:
                report.warnings.append("web_disabled")
        report.hits = [*local_hits, *report.web_hits]
        return report

    def search(self, query: str, *, top_k: int | None = None) -> list[RAGHit]:
        """保留原来的列表接口；新调用方用retrieve获取分组和降级诊断。"""
        return self.retrieve(query, top_k=top_k).hits

    async def aretrieve(
        self, query: str, *, top_k: int | None = None, force_web: bool = False
    ) -> RAGSearchResponse:
        return await asyncio.to_thread(self.retrieve, query, top_k=top_k, force_web=force_web)

    def close(self) -> None:
        self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)

    def __enter__(self) -> "RAGService":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

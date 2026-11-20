"""独立的文本重排序阶段，支持DashScope原生协议与Jina/Cohere兼容协议。

RRF只融合名次，不是语义重排序模型。真实rerank把query和候选正文一起
送给排序模型，再用返回的index定位原始证据，绝不接受模型改写的正文。
"""

import json
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, StrictInt

from customer_service_agent.config import Settings
from customer_service_agent.models import RAGHit


class RerankError(RuntimeError):
    """只携带可公开的错误代码，避免泄露供应商响应中的密钥或正文。"""


class RankedIndex(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    index: StrictInt = Field(ge=0)
    relevance_score: float = Field(ge=0, le=1, strict=True)


def reciprocal_rank_fusion(hits: list[RAGHit], *, k: int = 60) -> list[RAGHit]:
    """用两路排名融合，不直接相加BM25与余弦分数。"""
    ranked = []
    for position, original in enumerate(hits, 1):
        hit = original.model_copy(deep=True)
        ranks = hit.metadata.get("retrieval_ranks") or {"web": position}
        hit.metadata["rrf_score"] = sum(1 / (k + int(rank)) for rank in ranks.values())
        ranked.append(hit)
    return sorted(ranked, key=lambda h: h.metadata["rrf_score"], reverse=True)


class DashScopeReranker:
    def __init__(self, settings: Settings, *, transport: httpx.BaseTransport | None = None):
        self.settings = settings
        self.transport = transport

    def rerank(self, query: str, hits: list[RAGHit]) -> list[RAGHit]:
        s = self.settings
        if not s.rag_rerank_api_url or not s.rag_rerank_api_key or not s.rag_rerank_model_name:
            raise RerankError("rerank_not_configured")
        url = urlsplit(s.rag_rerank_api_url)
        if (
            url.scheme != "https"
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise RerankError("rerank_invalid_url")
        top_n = min(s.rag_rerank_top_n, len(hits))
        documents = [h.content[: s.rag_rerank_max_document_chars] for h in hits]
        if s.rag_rerank_api_format == "jina":
            # 只传各家网关都支持的字段；返回正文也不会被使用，下方只读取 index。
            payload = {
                "model": s.rag_rerank_model_name,
                "query": query,
                "documents": documents,
                "top_n": top_n,
            }
        else:
            payload = {
                "model": s.rag_rerank_model_name,
                "input": {"query": query, "documents": documents},
                "parameters": {"top_n": top_n, "return_documents": False},
            }
        # 每次请求独立客户端：线程安全、无隐式重试、无重定向泄露Authorization。
        try:
            with httpx.Client(
                timeout=s.rag_rerank_timeout_seconds,
                transport=self.transport,
                follow_redirects=False,
            ) as client:
                with client.stream(
                    "POST",
                    s.rag_rerank_api_url,
                    json=payload,
                    headers={"Authorization": "Bearer " + s.rag_rerank_api_key.get_secret_value()},
                ) as response:
                    response.raise_for_status()
                    body = bytearray()
                    for chunk in response.iter_bytes():
                        body.extend(chunk)
                        if len(body) > 100_000:
                            raise RerankError("rerank_response_too_large")
                    # 不记录或转发原始响应；只校验索引和分数。
                    parsed = json.loads(body)
                    if s.rag_rerank_api_format == "jina":
                        values = parsed["results"]
                    else:
                        values = parsed["output"]["results"]
            if not isinstance(values, list) or len(values) != top_n:
                raise ValueError("incorrect result count")
            ranks = [RankedIndex.model_validate(value) for value in values]
            indices = [item.index for item in ranks]
            if len(set(indices)) != len(indices) or any(index >= len(hits) for index in indices):
                raise ValueError("invalid result indices")
        except RerankError:
            raise
        except httpx.TimeoutException:
            raise RerankError("rerank_timeout") from None
        except httpx.HTTPStatusError:
            raise RerankError("rerank_http_error") from None
        except httpx.RequestError:
            raise RerankError("rerank_unavailable") from None
        except Exception:
            raise RerankError("rerank_invalid_response") from None

        result = []
        for item in sorted(ranks, key=lambda r: r.relevance_score, reverse=True):
            if item.relevance_score < s.rag_rerank_min_score:
                continue
            hit = hits[item.index].model_copy(deep=True)
            # 原始score仍是召回分数；排序分数使用独立字段，不能混淆含义。
            hit.metadata.update(
                rerank_score=item.relevance_score,
                rerank_rank=len(result) + 1,
                rerank_model=s.rag_rerank_model_name,
            )
            result.append(hit)
        return result

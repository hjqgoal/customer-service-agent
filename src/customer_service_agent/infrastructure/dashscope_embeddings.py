"""DashScope Qwen3-VL 纯文本向量适配；该模型不使用 OpenAI embeddings 路由。"""

import json
import math
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from langchain_core.embeddings import Embeddings
from pydantic import SecretStr


class DashScopeEmbeddingError(RuntimeError):
    """只包含安全诊断，不包含供应商响应正文或凭据。"""


class DashScopeTextEmbeddings(Embeddings):
    endpoint = (
        "https://dashscope.aliyuncs.com/api/v1/services/embeddings/"
        "multimodal-embedding/multimodal-embedding"
    )

    def __init__(
        self,
        api_key: SecretStr,
        *,
        model: str = "qwen3-vl-embedding",
        timeout: float = 60,
        max_retries: int = 2,
    ) -> None:
        self._api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.chunk_size = 8
        self.dimension = 1024

    def _batch(self, texts: list[str]) -> list[list[float]]:
        payload = {
            "model": self.model,
            "input": {"contents": [{"text": text} for text in texts]},
            "parameters": {"dimension": self.dimension, "enable_fusion": False},
        }
        request = Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key.get_secret_value()}",
            },
            method="POST",
        )
        for attempt in range(self.max_retries + 1):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    data = json.load(response)
                break
            except HTTPError as exc:
                code = exc.code
                exc.close()
                if code not in {429, 500, 502, 503, 504} or attempt == self.max_retries:
                    raise DashScopeEmbeddingError(f"DashScope embedding HTTP {code}") from None
            except (URLError, TimeoutError):
                if attempt == self.max_retries:
                    raise DashScopeEmbeddingError("DashScope embedding 网络连接失败") from None
            time.sleep(min(2**attempt, 4))
        try:
            entries = data["output"]["embeddings"]
            indices = [entry["index"] for entry in entries]
            if sorted(indices) != list(range(len(texts))):
                raise ValueError("missing or duplicate vector indices")
            vectors = [entry["embedding"] for entry in sorted(entries, key=lambda x: x["index"])]
            if any(
                len(vector) != self.dimension
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for value in vector
                )
                or not any(vector)
                for vector in vectors
            ):
                raise ValueError("invalid vector")
        except (KeyError, TypeError, ValueError):
            raise DashScopeEmbeddingError("DashScope embedding 返回向量格式无效") from None
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if any(not text.strip() for text in texts):
            raise ValueError("向量输入不能为空")
        vectors = []
        for start in range(0, len(texts), self.chunk_size):
            vectors.extend(self._batch(texts[start : start + self.chunk_size]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

"""Pydantic request and response contracts."""

from typing import Any, Literal

from pydantic import BaseModel, Field

SESSION_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.:\-]{0,127}$"
USER_ID_PATTERN = SESSION_ID_PATTERN


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000, description="用户输入")
    session_id: str = Field(
        default="default",
        pattern=SESSION_ID_PATTERN,
        description="会话 ID，用于隔离短期记忆",
    )
    user_id: str | None = Field(
        default=None,
        pattern=USER_ID_PATTERN,
        description="稳定用户 ID，用于跨会话结构化记忆；省略时使用 session_id",
    )
    request_id: str | None = Field(
        default=None,
        pattern=SESSION_ID_PATTERN,
        description="同一次请求重试时复用此 ID；新一轮问题必须换 ID",
    )
    confirm_action: bool = Field(default=False, description="明确确认创建模拟售后申请或工单")


class AgentStep(BaseModel):
    tool: str
    tool_input: str
    observation: str


class TaskReport(BaseModel):
    """可公开的任务进度；原始异常和内部工具参数不进入 API。"""

    task_id: str
    agent: Literal["order", "logistics", "after_sales"]
    status: Literal["success", "failed", "blocked"]
    attempts: int = 0
    error_code: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    user_id: str
    answer: str
    intermediate_steps: list[AgentStep] = Field(default_factory=list)
    mode: Literal["single", "multi"] = "single"
    request_id: str | None = None
    status: Literal["completed", "partial", "failed"] = "completed"
    tasks: list[TaskReport] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok"]
    app_name: str
    environment: str


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    checks: dict[str, bool]


class ErrorResponse(BaseModel):
    code: str
    message: str
    request_id: str | None = None


class RAGIngestRequest(BaseModel):
    replace: bool = Field(
        default=True,
        description="true 为原子替换索引；false 为按内容哈希增量添加",
    )


class RAGIngestResponse(BaseModel):
    loaded_documents: int
    indexed_chunks: int
    total_chunks: int


class RAGSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(
        default=3, ge=1, le=20, description="每路候选数量，上限由RAG_BRANCH_TOP_K控制"
    )
    force_web: bool = Field(default=False, description="本地候选不足以回答时，显式尝试网页降级")


class RAGHit(BaseModel):
    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGSearchResponse(BaseModel):
    query: str
    hits: list[RAGHit]
    # 保留分组和原始分数，避免把BM25与余弦相似度当成同一标尺。
    keyword_hits: list[RAGHit] = Field(default_factory=list)
    semantic_hits: list[RAGHit] = Field(default_factory=list)
    web_hits: list[RAGHit] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    web_attempted: bool = False
    rerank_status: Literal["disabled", "not_configured", "success", "failed", "no_candidates"] = (
        "disabled"
    )
    ranking_method: Literal["original", "rrf", "model"] = "original"

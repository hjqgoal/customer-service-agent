"""FastAPI transport layer."""

import hmac
import logging
import threading
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from customer_service_agent.agent import CustomerServiceAgent, build_customer_service_agent
from customer_service_agent.config import Settings, get_settings
from customer_service_agent.logging_config import configure_logging
from customer_service_agent.memory import CustomerProfile
from customer_service_agent.model_factory import ModelConfigError
from customer_service_agent.models import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthResponse,
    RAGIngestRequest,
    RAGIngestResponse,
    RAGSearchRequest,
    RAGSearchResponse,
    ReadinessResponse,
)
from customer_service_agent.multi_agent.request_store import (
    RequestConflictError,
    SessionOwnershipError,
)
from customer_service_agent.multi_agent.service import MultiAgentCustomerService
from customer_service_agent.rag.service import RAGService
from customer_service_agent.rag.sqlite_vector_store import RAGIndexError
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)
AgentService = CustomerServiceAgent | MultiAgentCustomerService
_service_lock = threading.Lock()
_agent: AgentService | None = None
_rag_service: RAGService | None = None


def get_rag_service() -> RAGService:
    global _rag_service
    with _service_lock:
        if _rag_service is None:
            settings = get_settings()
            _rag_service = RAGService.from_settings(settings)
        return _rag_service


def get_agent() -> AgentService:
    global _agent
    with _service_lock:
        if _agent is None:
            _agent = build_customer_service_agent(get_settings())
        return _agent


def close_runtime_services() -> None:
    global _agent, _rag_service
    with _service_lock:
        if _agent is not None:
            _agent.close()
        _agent = None
        if _rag_service is not None:
            _rag_service.close()
        _rag_service = None


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    configure_logging(get_settings())
    yield
    close_runtime_services()


app = FastAPI(
    title="LangChain Customer Service Agent",
    description="Supervisor-led customer service with specialist agents, RAG, and memory.",
    version="0.2.0",
    lifespan=lifespan,
)


def _error_response(request: Request, status_code: int, code: str, message: str) -> JSONResponse:
    body = ErrorResponse(
        code=code,
        message=message,
        request_id=getattr(request.state, "request_id", None),
    )
    return JSONResponse(status_code=status_code, content=body.model_dump())


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.request_id = request_id
    started_at = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request method=%s path=%s status=%s duration_ms=%.2f request_id=%s",
        request.method,
        request.url.path,
        response.status_code,
        (time.perf_counter() - started_at) * 1000,
        request_id,
    )
    return response


@app.exception_handler(ModelConfigError)
async def handle_model_config_error(request: Request, exc: ModelConfigError) -> JSONResponse:
    return _error_response(request, 503, "model_configuration_error", str(exc))


@app.exception_handler(StarletteHTTPException)
async def handle_http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    return _error_response(request, exc.status_code, "http_error", str(exc.detail))


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    logger.info("validation_error request_id=%s errors=%s", request.state.request_id, exc.errors())
    return _error_response(request, 422, "validation_error", "请求参数校验失败。")


@app.exception_handler(RAGIndexError)
async def handle_rag_index_error(request: Request, exc: RAGIndexError) -> JSONResponse:
    return _error_response(request, 409, "rag_index_error", str(exc))


@app.exception_handler(FileNotFoundError)
async def handle_not_found(request: Request, exc: FileNotFoundError) -> JSONResponse:
    return _error_response(request, 404, "not_found", str(exc))


@app.exception_handler(ValueError)
async def handle_value_error(request: Request, exc: ValueError) -> JSONResponse:
    return _error_response(request, 400, "invalid_request", str(exc))


@app.exception_handler(RequestConflictError)
async def handle_request_conflict(request: Request, exc: RequestConflictError) -> JSONResponse:
    return _error_response(request, 409, "request_conflict", str(exc))


@app.exception_handler(SessionOwnershipError)
async def handle_session_owner(request: Request, exc: SessionOwnershipError) -> JSONResponse:
    return _error_response(request, 403, "session_owner_mismatch", str(exc))


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "unhandled_error request_id=%s",
        request.state.request_id,
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    return _error_response(request, 500, "internal_error", "服务暂时不可用，请稍后重试。")


def require_admin(
    settings: Annotated[Settings, Depends(get_settings)],
    x_admin_key: Annotated[str | None, Header(alias="X-Admin-Key")] = None,
) -> None:
    configured = settings.admin_api_key
    if configured is not None and not configured.get_secret_value().strip():
        configured = None
    if configured is None:
        if settings.app_env == "prod":
            raise HTTPException(status_code=503, detail="ADMIN_API_KEY 未配置")
        return
    expected = configured.get_secret_value()
    if x_admin_key is None or not hmac.compare_digest(x_admin_key, expected):
        raise HTTPException(status_code=401, detail="无效的管理密钥")


@app.get("/health", response_model=HealthResponse)
def health(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    return HealthResponse(status="ok", app_name=settings.app_name, environment=settings.app_env)


@app.get("/ready", response_model=ReadinessResponse)
def readiness(
    response: Response,
    settings: Annotated[Settings, Depends(get_settings)],
) -> ReadinessResponse:
    chat_key_configured = bool(
        settings.resolved_chat_api_key and settings.resolved_chat_api_key.get_secret_value().strip()
    )
    embedding_key_configured = bool(
        settings.resolved_embedding_api_key
        and settings.resolved_embedding_api_key.get_secret_value().strip()
    )
    checks = {
        "data_directory": settings.memory_db_file.parent.is_dir(),
        "chat_model_configured": bool(settings.resolved_chat_model_name) and chat_key_configured,
        "embedding_model_configured": (
            bool(settings.resolved_embedding_model_name) and embedding_key_configured
        ),
    }
    ready = all(checks.values())
    if not ready:
        response.status_code = 503
    return ReadinessResponse(status="ready" if ready else "not_ready", checks=checks)


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    http_request: Request,
    agent: Annotated[AgentService, Depends(get_agent)],
) -> ChatResponse:
    if isinstance(agent, CustomerServiceAgent):
        response = await run_in_threadpool(
            agent.ask,
            request.message,
            session_id=request.session_id,
            user_id=request.user_id,
            include_trace=False,
        )
        return response.model_copy(
            update={
                "request_id": request.request_id
                or http_request.state.request_id
            }
        )  

    return await agent.aask(
        request.message,
        session_id=request.session_id,
        user_id=request.user_id,
        include_trace=False,
        # 未显式提供业务请求 ID 时，使用响应头可见的 ID，异常后也能据此重试。
        request_id=request.request_id or http_request.state.request_id,
        confirm_action=request.confirm_action,
    )


@app.post("/memory/{session_id}/clear", dependencies=[Depends(require_admin)])
def clear_memory(
    session_id: str,
    agent: Annotated[AgentService, Depends(get_agent)],
) -> dict[str, str]:
    agent.clear_memory(session_id)
    return {"status": "ok", "session_id": session_id}


@app.get(
    "/memory/profiles/{user_id}",
    response_model=CustomerProfile,
    dependencies=[Depends(require_admin)],
)
def get_customer_profile(
    user_id: str,
    agent: Annotated[AgentService, Depends(get_agent)],
) -> CustomerProfile:
    profile = agent.get_customer_profile(user_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="客户画像不存在")
    return profile


@app.delete("/memory/profiles/{user_id}", dependencies=[Depends(require_admin)])
def delete_customer_profile(
    user_id: str,
    agent: Annotated[AgentService, Depends(get_agent)],
) -> dict[str, str | bool]:
    return {"status": "ok", "user_id": user_id, "deleted": agent.delete_customer_profile(user_id)}


@app.post(
    "/rag/ingest",
    response_model=RAGIngestResponse,
    dependencies=[Depends(require_admin)],
)
def rag_ingest(
    request: RAGIngestRequest,
    rag_service: Annotated[RAGService, Depends(get_rag_service)],
) -> RAGIngestResponse:
    return rag_service.ingest_directory(replace=request.replace)


@app.post("/rag/search", response_model=RAGSearchResponse)
def rag_search(
    request: RAGSearchRequest,
    rag_service: Annotated[RAGService, Depends(get_rag_service)],
) -> RAGSearchResponse:
    return rag_service.retrieve(request.query, top_k=request.top_k, force_web=request.force_web)


def run() -> None:
    settings = get_settings()
    configure_logging(settings)
    uvicorn.run(
        "customer_service_agent.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.app_env == "dev",
    )


if __name__ == "__main__":
    run()

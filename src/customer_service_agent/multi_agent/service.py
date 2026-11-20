"""应用服务：同步/异步入口、会话互斥、异步 checkpoint 和请求重试回执。"""

import asyncio
import uuid

import aiosqlite
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Overwrite

from customer_service_agent.agent import _final_answer
from customer_service_agent.config import Settings
from customer_service_agent.infrastructure.async_runner import AsyncRunner
from customer_service_agent.memory import (
    AgentContext,
    ConversationCompactor,
    CustomerProfileRepository,
)
from customer_service_agent.models import ChatRequest, ChatResponse, TaskReport
from customer_service_agent.multi_agent.request_store import RequestStore
from customer_service_agent.multi_agent.specialists import build_specialists
from customer_service_agent.multi_agent.state import TaskResult
from customer_service_agent.multi_agent.workflow import CollaborationWorkflow
from customer_service_agent.rag.service import RAGService


class MultiAgentCustomerService:
    def __init__(
        self,
        settings: Settings,
        model,
        tools_by_agent: dict,
        profile_repository: CustomerProfileRepository,
        *,
        rag_service: RAGService | None = None,
    ) -> None:
        self.settings = settings
        self.llm = model
        self.profile_repository = profile_repository
        # 由工厂创建的 RAG 服务归本服务所有，close 时一并释放；外部注入的工具自行管理。
        self.rag_service = rag_service
        self.requests = RequestStore(
            settings.multi_agent_db_file, settings.sqlite_busy_timeout_seconds
        )
        self.compactor = ConversationCompactor(
            model,
            recent_turns=settings.memory_recent_turns,
            use_model_summary=settings.memory_summary_enabled,
            summary_max_chars=settings.memory_summary_max_chars,
            summary_input_max_chars=settings.memory_summary_input_max_chars,
        )
        self.runner = AsyncRunner()
        self._closed = False
        try:
            self.runner.submit(self._initialize(tools_by_agent)).result()
        except BaseException:
            self.runner.close()
            if self.rag_service is not None:
                self.rag_service.close()
            raise

    async def _initialize(self, tools_by_agent: dict) -> None:
        self._connection = await aiosqlite.connect(
            self.settings.multi_agent_db_file,
            timeout=self.settings.sqlite_busy_timeout_seconds,
        )
        try:
            await self._connection.execute("PRAGMA journal_mode=WAL")
            self.saver = AsyncSqliteSaver(self._connection)
            await self.saver.setup()
            self._session_locks = [asyncio.Lock() for _ in range(64)]
            self._active_requests: set[asyncio.Task] = set()
            self.workflow = CollaborationWorkflow(
                self.settings,
                self.llm,
                build_specialists(self.settings, self.llm, tools_by_agent),
                self.profile_repository,
            )
            self.graph = self.workflow.compile(self.saver)
        except BaseException:
            await self._connection.close()
            raise

    @staticmethod
    def _public_response(response: ChatResponse, include_trace: bool) -> ChatResponse:
        return response if include_trace else response.model_copy(update={"intermediate_steps": []})

    async def _blocking(self, function, *args):
        """取消请求时先等同步数据库操作收尾，再释放会话锁。"""
        pending = asyncio.create_task(asyncio.to_thread(function, *args))
        try:
            return await asyncio.shield(pending)
        except asyncio.CancelledError:
            await pending
            raise

    async def _tracked_ask(self, request: ChatRequest, include_trace: bool) -> ChatResponse:
        if self._closed:
            raise RuntimeError("Agent 已关闭")
        current = asyncio.current_task()
        self._active_requests.add(current)
        try:
            return await self._ask(request, include_trace)
        finally:
            self._active_requests.discard(current)

    async def _ask(self, request: ChatRequest, include_trace: bool) -> ChatResponse:
        session_id = request.session_id
        user_id = request.user_id or session_id
        request_id = request.request_id or uuid.uuid4().hex
        config = {"configurable": {"thread_id": session_id}, "recursion_limit": 32}
        async with self._session_locks[hash(session_id) % len(self._session_locks)]:
            cached = await self._blocking(
                self.requests.claim,
                session_id,
                request_id,
                user_id,
                request.message,
                request.confirm_action,
            )
            if cached:
                return self._public_response(cached, include_trace)
            snapshot = await self.graph.aget_state(config)
            continuing = snapshot.values.get("request_id") == request_id
            if continuing and not snapshot.next:
                # 进程可能在图完成之后、写回执之前退出，此时直接补写回执。
                result = snapshot.values
            else:
                if not continuing:
                    # 原摘要模块是同步的，放在线程中运行；异步 saver 支持跨线程同步桥接。
                    # 摘要同样可能调用模型，因此也计入全局并发上限。
                    async with self.workflow.semaphore:
                        await self._blocking(
                            self.compactor.compact_before_new_turn, self.graph, config
                        )
                inputs = (
                    None
                    if continuing
                    else {
                        "messages": [HumanMessage(content=request.message)],
                        "request_id": request_id,
                        "plan": [],
                        "results": Overwrite({}),
                        "ready": [],
                        "planning_error": None,
                        "status": "running",
                    }
                )
                result = await self.graph.ainvoke(
                    inputs,
                    config=config,
                    context=AgentContext(
                        user_id=user_id,
                        session_id=session_id,
                        request_id=request_id,
                        confirm_action=request.confirm_action,
                    ),
                )
            task_results = [
                TaskResult.model_validate(result["results"][t["task_id"]]) for t in result["plan"]
            ]
            response = ChatResponse(
                session_id=session_id,
                user_id=user_id,
                request_id=request_id,
                mode="multi",
                answer=_final_answer(result["messages"]),
                status=result["status"],
                tasks=[TaskReport.model_validate(t.model_dump()) for t in task_results],
                intermediate_steps=[step for task in task_results for step in task.steps],
            )
            await self._blocking(self.requests.complete, response)
            return self._public_response(response, include_trace)

    def ask(
        self,
        message: str,
        *,
        session_id: str = "default",
        user_id: str | None = None,
        include_trace: bool = False,
        request_id: str | None = None,
        confirm_action: bool = False,
    ) -> ChatResponse:
        request = ChatRequest(
            message=message,
            session_id=session_id,
            user_id=user_id,
            request_id=request_id,
            confirm_action=confirm_action,
        )
        return self.runner.submit(self._tracked_ask(request, include_trace)).result()

    async def aask(self, message: str, **kwargs) -> ChatResponse:
        include_trace = kwargs.pop("include_trace", False)
        request = ChatRequest(message=message, **kwargs)
        return await asyncio.wrap_future(
            self.runner.submit(self._tracked_ask(request, include_trace))
        )

    async def _clear(self, session_id: str) -> None:
        async with self._session_locks[hash(session_id) % len(self._session_locks)]:
            await self.saver.adelete_thread(session_id)
            await asyncio.to_thread(self.requests.clear, session_id)

    def clear_memory(self, session_id: str = "default") -> None:
        self.runner.submit(self._clear(session_id)).result()

    def get_customer_profile(self, user_id: str):
        return self.profile_repository.get(user_id)

    def delete_customer_profile(self, user_id: str) -> bool:
        return self.profile_repository.delete(user_id)

    async def _shutdown(self) -> None:
        tasks = list(self._active_requests)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        # 同步工具/摘要可能在线程中收尾，必须让它们先完成再关 SQLite。
        await asyncio.get_running_loop().shutdown_default_executor()
        await self._connection.close()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                self.runner.submit(self._shutdown()).result()
            finally:
                self.runner.close()
                if self.rag_service is not None:
                    self.rag_service.close()

    def __enter__(self):
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

"""协作主图：规划 -> 调度 -> 并行专业 Agent -> 调度 -> 汇总。

Supervisor 决定“做什么”，图决定“何时执行”。每次调度只派发依赖已成功的
任务；并行分支使用 reducer 合并结果，全部完成或失败后才进行统一答复。
"""

import asyncio
import json
import logging
from dataclasses import replace

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Send
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

from customer_service_agent.agent import _final_answer, _format_intermediate_steps, _message_text
from customer_service_agent.config import Settings
from customer_service_agent.memory import AgentContext, CustomerProfileRepository
from customer_service_agent.multi_agent.prompts import PLANNER_PROMPT, SYNTHESIS_PROMPT
from customer_service_agent.multi_agent.state import (
    AgentTask,
    TaskPlan,
    TaskResult,
    WorkerInput,
    WorkflowState,
)

logger = logging.getLogger(__name__)
TRANSIENT_ERRORS = (
    TimeoutError,
    ConnectionError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)


class CollaborationWorkflow:
    def __init__(
        self,
        settings: Settings,
        model: BaseChatModel,
        specialists: dict,
        profiles: CustomerProfileRepository,
    ) -> None:
        self.settings = settings
        self.model = model
        self.specialists = specialists
        self.profiles = profiles
        # 所有会话共用限流器，避免 10 个用户请求各启动 3 个模型调用。
        self.semaphore = asyncio.Semaphore(settings.multi_agent_max_concurrency)

    async def _profile(self, user_id: str) -> dict:
        profile = await asyncio.to_thread(self.profiles.get, user_id)
        return profile.prompt_payload() if profile else {}

    async def _plan(self, state: WorkflowState, runtime: Runtime[AgentContext]) -> dict:
        context = runtime.context
        profile = await self._profile(context.user_id)
        messages = [
            SystemMessage(content=PLANNER_PROMPT),
            *state["messages"],
            HumanMessage(
                content=json.dumps(
                    {"customer_profile": profile, "confirm_action": context.confirm_action},
                    ensure_ascii=False,
                )
            ),
        ]
        error_code = "invalid_plan"
        # 最多修正一次结构，不把自由文本计划直接当作可执行代码。
        for _ in range(2):
            try:
                async with self.semaphore:
                    async with asyncio.timeout(self.settings.supervisor_timeout_seconds):
                        response = await self.model.ainvoke(messages)
                text = _message_text(response).strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                plan = TaskPlan.model_validate_json(text)
                return {"plan": [t.model_dump() for t in plan.tasks], "planning_error": None}
            except (ValueError, IndexError):
                messages.append(
                    HumanMessage(content="计划格式或依赖无效，请重新输出符合规范的 JSON。")
                )
                error_code = "invalid_plan"
            except Exception as exc:
                logger.warning(
                    "supervisor_plan_failed request_id=%s error_type=%s",
                    context.request_id,
                    type(exc).__name__,
                )
                error_code = "supervisor_unavailable"
                break
        return {"plan": [], "planning_error": error_code}

    def _schedule(self, state: WorkflowState) -> dict:
        tasks = [AgentTask.model_validate(t) for t in state["plan"]]
        results = dict(state["results"])
        blocked = {}
        # 传播依赖失败：A 失败时，依赖 A 的 B 及依赖 B 的 C 都不会被执行。
        changed = True
        while changed:
            changed = False
            for task in tasks:
                if task.task_id in results:
                    continue
                if any(
                    dep in results and results[dep]["status"] != "success"
                    for dep in task.depends_on
                ):
                    result = TaskResult(
                        task_id=task.task_id,
                        agent=task.agent,
                        status="blocked",
                        error_code="dependency_failed",
                    ).model_dump()
                    blocked[task.task_id] = results[task.task_id] = result
                    changed = True
        ready = [
            task.task_id
            for task in tasks
            if task.task_id not in results
            and all(
                dep in results and results[dep]["status"] == "success" for dep in task.depends_on
            )
        ]
        return {"ready": ready, "results": blocked}

    def _dispatch(self, state: WorkflowState):
        if not state["ready"]:
            return "summarize"
        by_id = {task["task_id"]: task for task in state["plan"]}
        return [
            Send(
                by_id[task_id]["agent"],
                {
                    "task": by_id[task_id],
                    "messages": state["messages"],
                    "dependencies": {
                        dep: state["results"][dep] for dep in by_id[task_id]["depends_on"]
                    },
                },
            )
            for task_id in state["ready"]
        ]

    async def _worker(self, state: WorkerInput, runtime: Runtime[AgentContext]) -> dict:
        task = AgentTask.model_validate(state["task"])
        context = replace(runtime.context, task_id=task.task_id)
        if task.kind == "action" and not context.confirm_action:
            result = TaskResult(
                task_id=task.task_id,
                agent=task.agent,
                status="blocked",
                error_code="confirmation_required",
            )
            return {"results": {task.task_id: result.model_dump()}}
        profile = await self._profile(context.user_id)
        task_data = {
            "task": task.model_dump(),
            "dependency_results": state["dependencies"],
            "customer_profile": profile,
        }
        messages = [
            *state["messages"],
            HumanMessage(
                content=(
                    "以下 JSON 是本次分配的任务和参考数据：\n"
                    + json.dumps(task_data, ensure_ascii=False)
                )
            ),
        ]
        attempts = 0
        error_code = "agent_error"
        for attempt in range(1, self.settings.multi_agent_max_attempts + 1):
            attempts = attempt
            try:
                async with self.semaphore:
                    async with asyncio.timeout(self.settings.multi_agent_task_timeout_seconds):
                        result = await self.specialists[(task.agent, task.kind)].ainvoke(
                            {"messages": messages},
                            config={"recursion_limit": self.settings.agent_recursion_limit},
                            context=context,
                        )
                steps = _format_intermediate_steps(result["messages"])
                output = TaskResult(
                    task_id=task.task_id,
                    agent=task.agent,
                    status="success",
                    attempts=attempt,
                    answer=_final_answer(result["messages"]),
                    steps=steps,
                    evidence=[step.observation[:6000] for step in steps],
                )
                return {"results": {task.task_id: output.model_dump()}}
            except Exception as exc:
                error_code = "timeout" if isinstance(exc, TimeoutError) else "agent_error"
                logger.warning(
                    "worker_failed request_id=%s task=%s attempt=%s error_type=%s",
                    context.request_id,
                    task.task_id,
                    attempt,
                    type(exc).__name__,
                )
                if (
                    not isinstance(exc, TRANSIENT_ERRORS)
                    or attempt == self.settings.multi_agent_max_attempts
                ):
                    break
                await asyncio.sleep(0.1 * attempt)
        output = TaskResult(
            task_id=task.task_id,
            agent=task.agent,
            status="failed",
            attempts=attempts,
            error_code=error_code,
        )
        return {"results": {task.task_id: output.model_dump()}}

    async def _summarize(self, state: WorkflowState) -> dict:
        if state["planning_error"]:
            return {
                "messages": [AIMessage(content="暂时无法安排客服任务，请稍后重试或联系人工客服。")],
                "status": "failed",
            }
        # 按计划顺序汇总，避免并发完成顺序导致每次回复顺序不同。
        results = [
            TaskResult.model_validate(state["results"][task["task_id"]]) for task in state["plan"]
        ]
        failed = [r for r in results if r.status != "success"]
        successful = [r for r in results if r.status == "success"]
        status = "completed" if not failed else ("partial" if successful else "failed")
        payload = [
            {
                "task_id": r.task_id,
                "agent": r.agent,
                "status": r.status,
                "answer": r.answer,
                "evidence": r.evidence,
                "error_code": r.error_code,
            }
            for r in results
        ]
        try:
            async with self.semaphore:
                async with asyncio.timeout(self.settings.supervisor_timeout_seconds):
                    response = await self.model.ainvoke(
                        [
                            SystemMessage(content=SYNTHESIS_PROMPT),
                            *state["messages"],
                            HumanMessage(
                                content="任务执行结果：\n" + json.dumps(payload, ensure_ascii=False)
                            ),
                        ]
                    )
            answer = _message_text(response).strip()
            if not answer:
                raise ValueError("empty supervisor answer")
        except Exception as exc:
            logger.warning("supervisor_summary_failed error_type=%s", type(exc).__name__)
            answer = "\n\n".join(r.answer for r in successful) or "暂时无法完成处理，请稍后重试。"
            answer = "自动汇总暂时不可用，以下为已获得的结果：\n" + answer
            status = "partial" if successful else "failed"
        if failed:
            labels = {"order": "订单", "logistics": "物流", "after_sales": "售后"}
            pending = "、".join(dict.fromkeys(labels[r.agent] for r in failed))
            answer += f"\n\n尚未完成：{pending}相关任务。未取得的结果不能作为业务判断依据。"
        return {"messages": [AIMessage(content=answer)], "status": status}

    def compile(self, checkpointer):
        builder = StateGraph(WorkflowState, context_schema=AgentContext)
        builder.add_node("plan", self._plan)
        builder.add_node("schedule", self._schedule)
        for name in ("order", "logistics", "after_sales"):
            builder.add_node(name, self._worker, input_schema=WorkerInput)
            builder.add_edge(name, "schedule")
        builder.add_node("summarize", self._summarize)
        builder.add_edge(START, "plan")
        builder.add_edge("plan", "schedule")
        builder.add_conditional_edges(
            "schedule", self._dispatch, ["order", "logistics", "after_sales", "summarize"]
        )
        builder.add_edge("summarize", END)
        return builder.compile(checkpointer=checkpointer)

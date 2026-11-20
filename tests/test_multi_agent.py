import asyncio
import logging
import sqlite3

import httpx
import openai
import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from customer_service_agent.agent import build_customer_service_agent
from customer_service_agent.memory.compaction import SUMMARY_PREFIX
from customer_service_agent.multi_agent.request_store import (
    RequestConflictError,
    SessionOwnershipError,
)
from customer_service_agent.multi_agent.state import TaskPlan
from customer_service_agent.multi_agent.workflow import error_details
from tests.multi_fakes import CollaborationModel, task


@pytest.fixture
def multi_settings(test_settings):
    return test_settings.model_copy(update={"agent_mode": "multi", "memory_summary_enabled": False})


def build(settings, model, specialist_tools=None):
    return build_customer_service_agent(
        settings, llm=model, specialist_tools=specialist_tools or {}
    )


def test_parallel_tasks_and_dependency_join(multi_settings):
    model = CollaborationModel(
        plan={
            "tasks": [
                task("order"),
                task("policy", "after_sales"),
                task("tracking", "logistics", ["order"]),
                task("assessment", "after_sales", ["order", "policy", "tracking"]),
            ]
        },
        delays={"order": 0.05, "policy": 0.05},
    )
    with build(multi_settings, model) as agent:
        response = agent.ask("查询订单、物流并评估退款", request_id="parallel")
        state = agent.graph.get_state({"configurable": {"thread_id": "default"}})
    assert response.status == "completed"
    assert len(response.tasks) == 4
    assert model.max_active >= 2
    assert model.events.index(("end", "order")) < model.events.index(("start", "tracking"))
    assert model.events.index(("end", "tracking")) < model.events.index(("start", "assessment"))
    assert model.events.index(("end", "policy")) < model.events.index(("start", "assessment"))
    assert len(state.values["messages"]) == 2  # 子 Agent 的工具消息不污染父级会话。


def test_only_selected_agent_runs(multi_settings):
    model = CollaborationModel(plan={"tasks": [task("order")]})
    with build(multi_settings, model) as agent:
        response = agent.ask("查订单")
    assert [t.agent for t in response.tasks] == ["order"]
    assert model.attempts == {"order": 1}


def test_timeout_preserves_success_and_blocks_dependents(multi_settings):
    settings = multi_settings.model_copy(
        update={"multi_agent_task_timeout_seconds": 0.08, "multi_agent_max_attempts": 1}
    )
    model = CollaborationModel(
        plan={
            "tasks": [
                task("order"),
                task("tracking", "logistics"),
                task("assessment", "after_sales", ["tracking"]),
            ]
        },
        delays={"tracking": 1},
    )
    with build(settings, model) as agent:
        response = agent.ask("查询进度")
    results = {t.task_id: t for t in response.tasks}
    assert response.status == "partial"
    assert results["order"].status == "success"
    assert results["tracking"].error_code == "timeout"
    assert results["assessment"].status == "blocked"
    assert "assessment" not in model.attempts
    assert "尚未完成" in response.answer


def test_retry_only_transient_errors(multi_settings):
    model = CollaborationModel(
        plan={"tasks": [task("order"), task("policy", "after_sales")]},
        failures={"order": "transient", "policy": "fatal"},
    )
    with build(multi_settings, model) as agent:
        response = agent.ask("订单及政策")
    results = {t.task_id: t for t in response.tasks}
    assert results["order"].status == "success"
    assert results["order"].attempts == 2
    assert results["policy"].attempts == 1


@pytest.mark.parametrize(
    "tasks",
    [
        [task("a", depends_on=["missing"])],
        [task("a", depends_on=["b"]), task("b", depends_on=["a"])],
        [task("a"), task("a")],
        [task("write", "after_sales", kind="action")],
    ],
)
def test_plan_rejects_unknown_dependencies_cycles_and_unsafe_actions(tasks):
    with pytest.raises(ValidationError):
        TaskPlan(tasks=tasks)


def test_invalid_plan_has_bounded_repair_and_clear_failure(multi_settings):
    model = CollaborationModel(plan={"tasks": [task("a", depends_on=["a"])]})
    with build(multi_settings, model) as agent:
        response = agent.ask("查订单")
    assert response.status == "failed"
    assert model.plan_calls == 2
    assert not model.attempts


def test_provider_failure_logs_status_and_error_code(multi_settings, caplog):
    class ArrearsModel(CollaborationModel):
        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            if "只生成 JSON 任务计划" in str(messages[0].content):
                request = httpx.Request("POST", "https://model.example.test/chat/completions")
                raise openai.BadRequestError(
                    "Access denied",
                    response=httpx.Response(400, request=request),
                    body={"code": "Arrearage", "message": "账户欠费明细"},
                )
            return await super()._agenerate(messages, stop, run_manager, **kwargs)

    caplog.set_level(logging.WARNING, logger="customer_service_agent.multi_agent.workflow")
    with build(multi_settings, ArrearsModel()) as agent:
        response = agent.ask("查订单", request_id="arrears")
    assert response.status == "failed"
    assert (
        "request_id=arrears error_type=BadRequestError status=400 provider_code=Arrearage"
        in caplog.text
    )
    assert "账户欠费明细" not in caplog.text  # 不记录供应商响应正文
    assert error_details(TimeoutError()) == "status=- provider_code=-"


def test_synthesis_failure_returns_worker_results(multi_settings):
    model = CollaborationModel(plan={"tasks": [task("order")]}, summary_fails=True)
    with build(multi_settings, model) as agent:
        response = agent.ask("查订单")
    assert "order 已完成" in response.answer
    assert response.status == "partial"


def test_request_replay_after_restart_does_not_call_models(multi_settings):
    model = CollaborationModel(plan={"tasks": [task("order")]})
    with build(multi_settings, model) as agent:
        first = agent.ask("查订单", session_id="s", user_id="u", request_id="r")
    next_model = CollaborationModel()
    with build(multi_settings, next_model) as agent:
        repeated = agent.ask("查订单", session_id="s", user_id="u", request_id="r")
        with pytest.raises(RequestConflictError):
            agent.ask("另一条问题", session_id="s", user_id="u", request_id="r")
        with pytest.raises(SessionOwnershipError):
            agent.ask("查订单", session_id="s", user_id="other", request_id="r")
    assert first == repeated
    assert next_model.plan_calls == 0


def test_same_session_concurrent_duplicate_is_executed_once(multi_settings):
    model = CollaborationModel(plan={"tasks": [task("order")]}, delays={"order": 0.04})
    with build(multi_settings, model) as agent:

        async def calls():
            return await asyncio.gather(
                *[
                    agent.aask("查订单", session_id="same", request_id="same-request")
                    for _ in range(3)
                ]
            )

        responses = asyncio.run(calls())
        state = agent.graph.get_state({"configurable": {"thread_id": "same"}})
    assert responses[0] == responses[1] == responses[2]
    assert model.plan_calls == 1
    assert sum(isinstance(m, HumanMessage) for m in state.values["messages"]) == 1


def test_global_concurrency_limit_and_session_isolation(multi_settings):
    settings = multi_settings.model_copy(update={"multi_agent_max_concurrency": 2})
    model = CollaborationModel(
        plan={"tasks": [task("order"), task("policy", "after_sales")]},
        delays={"order": 0.04, "policy": 0.04},
    )
    with build(settings, model) as agent:

        async def calls():
            return await asyncio.gather(
                agent.aask("甲的问题", session_id="a"), agent.aask("乙的问题", session_id="b")
            )

        asyncio.run(calls())
        a = agent.graph.get_state({"configurable": {"thread_id": "a"}}).values["messages"]
        b = agent.graph.get_state({"configurable": {"thread_id": "b"}}).values["messages"]
    assert model.max_active <= 2
    assert a[0].content == "甲的问题"
    assert b[0].content == "乙的问题"


def test_multi_memory_compacts_only_parent_dialogue(multi_settings):
    model = CollaborationModel(plan={"tasks": [task("order")]})
    with build(multi_settings, model) as agent:
        for message in ("第一轮", "第二轮", "第三轮", "第四轮"):
            agent.ask(message)
        state = agent.graph.get_state({"configurable": {"thread_id": "default"}})
        messages = state.values["messages"]
        assert len(state.values["results"]) == 1  # 每轮清空任务结果，不累积旧任务。
        agent.clear_memory()
        assert not agent.graph.get_state({"configurable": {"thread_id": "default"}}).values
    assert [m.content for m in messages if isinstance(m, HumanMessage)] == ["第三轮", "第四轮"]
    assert any(
        isinstance(m, SystemMessage) and str(m.content).startswith(SUMMARY_PREFIX) for m in messages
    )


def test_cancelled_request_resumes_after_restart_without_replanning(multi_settings):
    plan = {"tasks": [task("order"), task("policy", "after_sales", ["order"])]}
    model = CollaborationModel(plan=plan, delays={"policy": 5})
    with build(multi_settings, model) as agent:

        async def interrupt():
            pending = asyncio.create_task(agent.aask("订单和政策", request_id="resume"))
            async with asyncio.timeout(5):
                while ("start", "policy") not in model.events:
                    await asyncio.sleep(0.01)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending

        asyncio.run(interrupt())
    fresh = CollaborationModel(plan=plan)
    with build(multi_settings, fresh) as agent:
        response = agent.ask("订单和政策", request_id="resume")
    assert response.status == "completed"
    assert fresh.plan_calls == 0
    assert "order" not in fresh.attempts
    assert fresh.attempts["policy"] == 1


def test_unconfirmed_write_is_blocked(multi_settings):
    model = CollaborationModel(
        plan={"tasks": [task("order"), task("submit", "after_sales", ["order"], "action")]}
    )
    with build(multi_settings, model) as agent:
        response = agent.ask("提交申请")
    assert response.tasks[-1].error_code == "confirmation_required"
    assert "submit" not in model.attempts


def test_default_mode_is_multi():
    from customer_service_agent.config import Settings

    assert Settings(_env_file=None).agent_mode == "multi"


def test_concurrent_different_turns_in_one_session_are_serialized(multi_settings):
    settings = multi_settings.model_copy(update={"memory_recent_turns": 6})
    model = CollaborationModel(plan={"tasks": [task("order")]}, delays={"order": 0.03})
    with build(settings, model) as agent:

        async def calls():
            return await asyncio.gather(
                *[
                    agent.aask(f"第 {i} 轮", session_id="one", request_id=f"turn-{i}")
                    for i in range(3)
                ]
            )

        responses = asyncio.run(calls())
        state = agent.graph.get_state({"configurable": {"thread_id": "one"}})
    assert all(r.status == "completed" for r in responses)
    assert model.max_active == 1
    assert sum(isinstance(m, HumanMessage) for m in state.values["messages"]) == 3


def test_completed_receipt_survives_memory_clear(multi_settings):
    model = CollaborationModel(plan={"tasks": [task("order")]})
    with build(multi_settings, model) as agent:
        first = agent.ask("查订单", request_id="clear-replay")
        agent.clear_memory()
        second = agent.ask("查订单", request_id="clear-replay")
    assert first == second
    assert model.plan_calls == 1
    with sqlite3.connect(multi_settings.multi_agent_db_file) as connection:
        assert connection.execute("SELECT count(*) FROM workflow_requests").fetchone()[0] == 1

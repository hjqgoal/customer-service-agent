"""按协议返回任务计划和工具调用的假模型，不访问网络。"""

import asyncio
import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from tests.fakes import FakeToolCallingChatModel


def task(task_id, agent="order", depends_on=None, kind="query"):
    return {
        "task_id": task_id,
        "agent": agent,
        "instruction": f"执行 {task_id}",
        "depends_on": depends_on or [],
        "kind": kind,
    }


class CollaborationModel(FakeToolCallingChatModel):
    plan: dict = Field(default_factory=lambda: {"tasks": []})
    delays: dict[str, float] = Field(default_factory=dict)
    failures: dict[str, str] = Field(default_factory=dict)
    scripted_calls: dict[str, dict] = Field(default_factory=dict)
    events: list[tuple[str, str]] = Field(default_factory=list)
    attempts: dict[str, int] = Field(default_factory=dict)
    plan_calls: int = 0
    active: int = 0
    max_active: int = 0
    summary_fails: bool = False

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        system = str(messages[0].content)
        if "只生成 JSON 任务计划" in system:
            self.plan_calls += 1
            message = AIMessage(content=json.dumps(self.plan, ensure_ascii=False))
        elif "负责把专业 Agent 的结果" in system:
            if self.summary_fails:
                raise ConnectionError("summary unavailable")
            message = AIMessage(content="已汇总演示查询结果。")
        else:
            data_message = next(
                m
                for m in reversed(messages)
                if isinstance(m, HumanMessage)
                and str(m.content).startswith("以下 JSON 是本次分配的任务")
            )
            data = json.loads(str(data_message.content).split("\n", 1)[1])
            task_id = data["task"]["task_id"]
            if isinstance(messages[-1], ToolMessage):
                message = AIMessage(content=f"{task_id} 完成：{messages[-1].content}")
            else:
                self.attempts[task_id] = self.attempts.get(task_id, 0) + 1
                self.events.append(("start", task_id))
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                try:
                    await asyncio.sleep(self.delays.get(task_id, 0))
                    failure = self.failures.get(task_id)
                    if failure == "fatal":
                        raise ValueError("invalid backend data")
                    if failure == "transient" and self.attempts[task_id] == 1:
                        raise ConnectionError("temporary disconnect")
                    call = self.scripted_calls.get(task_id)
                    if call:
                        message = AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    **call,
                                    "id": f"{task_id}-call",
                                    "type": "tool_call",
                                }
                            ],
                        )
                    else:
                        message = AIMessage(content=f"{task_id} 已完成演示查询")
                finally:
                    self.active -= 1
                    self.events.append(("end", task_id))
        return ChatResult(generations=[ChatGeneration(message=message)])

"""任务契约和共享状态。LLM 提出计划，Pydantic 和调度器负责约束执行。"""

from graphlib import CycleError, TopologicalSorter
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field, model_validator

from customer_service_agent.models import AgentStep, TaskReport

AgentName = Literal["order", "logistics", "after_sales"]


class AgentTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,39}$")
    agent: AgentName
    instruction: str = Field(min_length=1, max_length=2000)
    depends_on: list[str] = Field(default_factory=list, max_length=5)
    kind: Literal["query", "action"] = "query"


class TaskPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tasks: list[AgentTask] = Field(default_factory=list, max_length=6)

    @model_validator(mode="after")
    def validate_dependencies(self) -> "TaskPlan":
        by_id = {task.task_id: task for task in self.tasks}
        if len(by_id) != len(self.tasks):
            raise ValueError("task_id 不能重复")
        for task in self.tasks:
            if any(dep not in by_id for dep in task.depends_on):
                raise ValueError("依赖必须指向计划中的任务")
            if task.kind == "action":
                if task.agent != "after_sales":
                    raise ValueError("只有售后 Agent 可以执行写操作")
                if not any(by_id[dep].agent == "order" for dep in task.depends_on):
                    raise ValueError("售后写操作必须依赖订单核实任务")
        try:
            tuple(TopologicalSorter({t.task_id: t.depends_on for t in self.tasks}).static_order())
        except CycleError as exc:
            raise ValueError("任务之间不能存在循环依赖") from exc
        return self


class TaskResult(TaskReport):
    """内部结果可包含证据与工具轨迹，汇总时提供给 Supervisor。"""

    answer: str = ""
    evidence: list[str] = Field(default_factory=list)
    steps: list[AgentStep] = Field(default_factory=list)


def merge_results(current: dict, updates: dict) -> dict:
    """并发分支各写一个 task_id；合并时拒绝相互覆盖。"""

    merged = dict(current)
    for task_id, result in updates.items():
        if task_id in merged and merged[task_id] != result:
            raise ValueError(f"同一个任务收到冲突结果: {task_id}")
        merged[task_id] = result
    return merged


class WorkflowState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    request_id: str
    plan: list[dict]
    results: Annotated[dict[str, dict], merge_results]
    ready: list[str]
    planning_error: str | None
    status: str


class WorkerInput(TypedDict):
    task: dict
    messages: list[BaseMessage]
    dependencies: dict[str, dict]

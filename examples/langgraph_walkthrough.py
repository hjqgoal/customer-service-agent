"""把项目的隐式 Agent 图展开，观察真实 LangGraph 执行过程。

运行：python -m examples.langgraph_walkthrough
模型是本地规则模拟器；复用项目真实工具实现，订单数据来自内置演示库。
StateGraph、reducer 和 SQLite checkpoint 使用当前安装的 LangGraph。
不读取 .env，不调用模型 API，不写入项目原有数据库。
"""

import argparse
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt

from customer_service_agent.agent import build_customer_service_agent
from customer_service_agent.config import Settings
from customer_service_agent.memory import AgentContext, SQLiteAgentCheckpointer
from customer_service_agent.tools.order_tool import build_order_lookup_tool


class DemoModel(BaseChatModel):
    """只模拟“提出工具调用/读取工具结果”，不具有真实语言理解能力。"""

    @property
    def _llm_type(self) -> str:
        return "offline-langgraph-lesson"

    def bind_tools(self, tools, **kwargs):
        # 真实模型会把工具 schema 发给模型服务；此处只模拟已知的 lookup_order。
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        last = messages[-1]
        if isinstance(last, ToolMessage):
            answer = AIMessage(content=f"根据工具返回：{last.content}")
        else:
            order = re.search(r"ORDER-\d+", str(last.content).upper())
            if order:
                answer = AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "lookup_order",
                            "args": {"order_id": order.group()},
                            "id": "lesson-order-call",
                            "type": "tool_call",
                        }
                    ],
                )
            else:
                answer = AIMessage(content="你好，请提供订单号，例如 ORDER-1001。")
        return ChatResult(generations=[ChatGeneration(message=answer)])


# 1. State：节点共享的数据结构。TypedDict 声明字段，不是一个需要 new 的业务对象。
class LearningState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    model_calls: NotRequired[int]
    approved: NotRequired[bool]


def build_explicit_graph(model, tools, saver, *, with_review: bool = False):
    """展开 create_agent 的核心循环；教学版不复制其全部中间件、错误处理与工具语义。"""
    bound_model = model.bind_tools(tools)

    # 2. Node：读取 state，返回本节点的更新，而不是直接改传入的 state。
    def call_model(state: LearningState, runtime: Runtime[AgentContext]) -> dict:
        system = SystemMessage(
            content=f"你是客服。当前演示用户：{runtime.context.user_id}。订单事实先查工具。"
        )
        response = bound_model.invoke([system, *state["messages"]])
        return {
            "messages": [response],  # add_messages 把新消息合并进旧消息。
            "model_calls": state.get("model_calls", 0) + 1,  # 普通字段使用新值覆盖。
        }

    # 3. 条件边：根据模型刚写入的 AIMessage，选择下一个节点。
    def after_model(state: LearningState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "review" if with_review else "tools"
        return END

    def review(state: LearningState) -> dict:
        calls = state["messages"][-1].tool_calls
        # 这是新增的教学关卡，原单 Agent 没有此审批节点。
        # 恢复时 review 会从头重跑，所以 interrupt 之前不要放不可重复的写操作。
        approved = interrupt({"question": "是否允许执行这些演示工具？", "tool_calls": calls})
        if type(approved) is not bool:
            raise ValueError("教学审批只接受布尔值 True 或 False")
        if approved:
            return {"approved": True}
        # 拒绝时也补齐工具协议，给每个 tool_call 一条对应的 ToolMessage。
        return {
            "approved": False,
            "messages": [
                ToolMessage(
                    content="人工拒绝了本次工具调用，工具未执行。",
                    tool_call_id=call["id"],
                    name=call["name"],
                )
                for call in calls
            ],
        }

    def after_review(state: LearningState) -> Literal["tools", "model"]:
        return "tools" if state["approved"] else "model"

    # 4. Graph：先注册节点，再声明普通边和条件边。
    builder = StateGraph(LearningState, context_schema=AgentContext)
    builder.add_node("model", call_model)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "model")
    destinations = {"tools": "tools", END: END}
    if with_review:
        builder.add_node("review", review)
        builder.add_conditional_edges("review", after_review)
        destinations = {"review": "review", END: END}
    builder.add_conditional_edges("model", after_model, destinations)
    builder.add_edge("tools", "model")

    # 5. compile 生成可执行图；checkpointer 配合 thread_id 保存每个执行步骤的状态。
    return builder.compile(checkpointer=saver)


def readable(value: Any) -> Any:
    """隐藏随机消息 ID，把教学重点留给消息类型、工具调用和状态字段。"""
    if hasattr(value, "content") and hasattr(value, "type"):
        result = {"type": value.type, "content": value.content}
        if isinstance(value, AIMessage) and value.tool_calls:
            result["tool_calls"] = value.tool_calls
        if isinstance(value, ToolMessage):
            result["tool_call_id"] = value.tool_call_id
        return result
    if isinstance(value, dict):
        return {key: readable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [readable(item) for item in value]
    if hasattr(value, "value"):  # LangGraph Interrupt
        return readable(value.value)
    return value


def print_value(label: str, value: Any) -> None:
    print(label)
    print(json.dumps(readable(value), ensure_ascii=False, indent=2))


def trace(graph, inputs, config, context) -> None:
    # 同一次执行输出两种事件；不要先 invoke 再 stream 同一个请求，否则会执行两遍。
    for mode, payload in graph.stream(
        inputs, config=config, context=context, stream_mode=["updates", "values"]
    ):
        print_value(f"\n[{mode}]", payload)
    snapshot = graph.get_state(config)
    print_value("\n[snapshot.values：已保存的数据]", snapshot.values)
    print_value("[snapshot.next：等待执行的节点]", snapshot.next)


def lesson_settings(root: Path) -> Settings:
    return Settings(
        _env_file=None,
        app_env="test",
        agent_mode="single",
        memory_summary_enabled=False,
        memory_db_path=root / "original.sqlite3",
        multi_agent_db_path=root / "multi.sqlite3",
        after_sales_db_path=root / "after-sales.sqlite3",
        profile_db_path=root / "profiles.sqlite3",
        rag_db_path=root / "rag.sqlite3",
        ticket_db_path=root / "tickets.sqlite3",
        knowledge_base_dir=root / "knowledge",
    )


def run_original(root: Path) -> None:
    print("\n=== A. 项目原有 create_agent：图和 State 确实存在 ===")
    with build_customer_service_agent(
        lesson_settings(root), llm=DemoModel(), tools=[build_order_lookup_tool()]
    ) as agent:
        print(agent.graph.get_graph().draw_mermaid())
        config = {"configurable": {"thread_id": "original-lesson"}, "recursion_limit": 12}
        # 直接观察底层图，故意不经过 ask 的会话锁、历史压缩和 HTTP 响应包装。
        trace(
            agent.graph,
            {"messages": [("user", "查询 ORDER-1001")]},
            config,
            AgentContext(user_id="learner"),
        )


def run_explicit(root: Path, *, decision: bool, message: str) -> None:
    print("\n=== B. 显式 StateGraph：可见的状态、节点、分支及暂停恢复 ===")
    config = {"configurable": {"thread_id": "explicit-lesson"}, "recursion_limit": 12}
    context = AgentContext(user_id="learner")
    database = root / "explicit.sqlite3"
    tools = [build_order_lookup_tool()]
    checkpoint = SQLiteAgentCheckpointer(database)
    try:
        graph = build_explicit_graph(DemoModel(), tools, checkpoint.saver, with_review=True)
        print(graph.get_graph().draw_mermaid())
        trace(graph, {"messages": [("user", message)]}, config, context)
        pending = bool(graph.get_state(config).next)
    finally:
        checkpoint.close()

    # 关闭连接、重新构图，验证状态来自 SQLite，不是 Python 局部变量碰巧还在。
    checkpoint = SQLiteAgentCheckpointer(database)
    try:
        graph = build_explicit_graph(DemoModel(), tools, checkpoint.saver, with_review=True)
        print_value("\n[重新打开 SQLite 后的 next]", graph.get_state(config).next)
        if pending:
            print_value("[脚本模拟外部审批；可用 --decision reject 改成拒绝]", decision)
            trace(graph, Command(resume=decision), config, context)
        print_value(
            "[另一个 thread_id 的状态]",
            graph.get_state({"configurable": {"thread_id": "another-lesson"}}).values,
        )
        print_value(
            "[checkpoint 历史：最新在前]",
            [
                {
                    "step": snapshot.metadata.get("step"),
                    "next": snapshot.next,
                    "messages": len(snapshot.values.get("messages", [])),
                }
                for snapshot in graph.get_state_history(config)
            ],
        )
    finally:
        checkpoint.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["original", "explicit", "all"], default="all")
    parser.add_argument("--decision", choices=["approve", "reject"], default="approve")
    parser.add_argument("--message", default="查询 ORDER-1001")
    args = parser.parse_args()
    with TemporaryDirectory(prefix="langgraph-lesson-") as directory:
        root = Path(directory)
        if args.mode in {"original", "all"}:
            run_original(root)
        if args.mode in {"explicit", "all"}:
            run_explicit(root, decision=args.decision == "approve", message=args.message)


if __name__ == "__main__":
    main()

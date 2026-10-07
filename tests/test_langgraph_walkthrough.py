"""验证教学展示的状态合并、工具分支和持久化审批恢复确实成立。"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.types import Command

from customer_service_agent.memory import AgentContext, SQLiteAgentCheckpointer
from examples.langgraph_walkthrough import DemoModel, build_explicit_graph


@pytest.mark.parametrize("decision", [True, False])
def test_review_resumes_from_sqlite_and_controls_actual_tool_execution(tmp_path, decision):
    calls = []

    @tool
    def lookup_order(order_id: str) -> str:
        """查询测试订单。"""
        calls.append(order_id)
        return "已发货"

    database = tmp_path / "lesson.sqlite3"
    config = {"configurable": {"thread_id": "lesson"}}
    context = AgentContext(user_id="student")
    checkpoint = SQLiteAgentCheckpointer(database)
    try:
        graph = build_explicit_graph(
            DemoModel(), [lookup_order], checkpoint.saver, with_review=True
        )
        graph.invoke({"messages": [("user", "查询 ORDER-1001")]}, config, context=context)
        snapshot = graph.get_state(config)
        assert snapshot.next == ("review",)
        assert snapshot.values["model_calls"] == 1
        assert calls == []
        assert "user_id" not in snapshot.values  # context 不会自动变成 state。
    finally:
        checkpoint.close()

    checkpoint = SQLiteAgentCheckpointer(database)
    try:
        graph = build_explicit_graph(
            DemoModel(), [lookup_order], checkpoint.saver, with_review=True
        )
        result = graph.invoke(Command(resume=decision), config, context=context)
        assert [message.type for message in result["messages"]] == ["human", "ai", "tool", "ai"]
        assert result["model_calls"] == 2
        assert result["approved"] is decision
        assert calls == (["ORDER-1001"] if decision else [])
        assert graph.get_state(config).next == ()
        assert not graph.get_state({"configurable": {"thread_id": "other"}}).values
    finally:
        checkpoint.close()


def test_without_review_tool_loop_and_second_turn_keep_history(tmp_path):
    @tool
    def lookup_order(order_id: str) -> str:
        """查询测试订单。"""
        return "已发货"

    checkpoint = SQLiteAgentCheckpointer(tmp_path / "direct.sqlite3")
    config = {"configurable": {"thread_id": "lesson"}}
    context = AgentContext(user_id="student")
    try:
        graph = build_explicit_graph(DemoModel(), [lookup_order], checkpoint.saver)
        updates = list(
            graph.stream(
                {"messages": [("user", "查询 ORDER-1001")]},
                config,
                context=context,
                stream_mode="updates",
            )
        )
        assert [next(iter(update)) for update in updates] == ["model", "tools", "model"]
        second = graph.invoke({"messages": [("user", "你好")]}, config, context=context)
        assert [m.type for m in second["messages"]] == ["human", "ai", "tool", "ai", "human", "ai"]
        assert second["model_calls"] == 3
        assert isinstance(second["messages"][2], ToolMessage)
    finally:
        checkpoint.close()


def test_add_messages_replaces_existing_id_instead_of_blind_append():
    before = [HumanMessage(content="原问题", id="user-1"), AIMessage(content="旧答案", id="ai-1")]
    after = add_messages(before, [AIMessage(content="修正答案", id="ai-1")])
    assert len(after) == 2
    assert after[-1].content == "修正答案"

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from customer_service_agent.agent import build_customer_service_agent
from customer_service_agent.memory import CustomerProfileRepository
from customer_service_agent.memory.compaction import SUMMARY_PREFIX
from customer_service_agent.tools.profile_tool import build_customer_profile_tools
from tests.fakes import FakeProfileSavingChatModel, FakeToolCallingChatModel


def test_agent_uses_langgraph_sqlite_memory_without_real_api(test_settings):
    with build_customer_service_agent(
        test_settings,
        llm=FakeToolCallingChatModel(),
        tools=[],
    ) as agent:
        response = agent.ask("你好", session_id="student")
        state = agent.graph.get_state({"configurable": {"thread_id": "student"}})
        messages = state.values["messages"]

        assert response.answer == "这是一个测试用客服回复。"
        assert response.user_id == "student"
        assert isinstance(messages[0], HumanMessage)
        assert isinstance(messages[-1], AIMessage)
        assert messages[0].content == "你好"


def test_agent_can_clear_thread_memory(test_settings):
    with build_customer_service_agent(
        test_settings,
        llm=FakeToolCallingChatModel(),
        tools=[],
    ) as agent:
        agent.ask("需要退款", session_id="clear-me")
        agent.clear_memory("clear-me")
        state = agent.graph.get_state({"configurable": {"thread_id": "clear-me"}})

        assert state.values == {}


def test_agent_keeps_recent_turns_and_summarizes_older_history(test_settings):
    deterministic_settings = test_settings.model_copy(update={"memory_summary_enabled": False})
    with build_customer_service_agent(
        deterministic_settings,
        llm=FakeToolCallingChatModel(response_text="固定回复"),
        tools=[],
    ) as agent:
        agent.ask("第一轮：咨询退款", session_id="compact")
        agent.ask("第二轮：补充订单号", session_id="compact")
        agent.ask("第三轮：询问进度", session_id="compact")
        agent.ask("第四轮：继续追问", session_id="compact")
        state = agent.graph.get_state({"configurable": {"thread_id": "compact"}})
        messages = list(state.values["messages"])

    human_messages = [message for message in messages if isinstance(message, HumanMessage)]
    summaries = [
        message
        for message in messages
        if isinstance(message, SystemMessage) and str(message.content).startswith(SUMMARY_PREFIX)
    ]
    assert [message.content for message in human_messages] == [
        "第三轮：询问进度",
        "第四轮：继续追问",
    ]
    assert len(summaries) == 1
    assert "不可信数据" in summaries[0].content
    assert "第一轮：咨询退款" in summaries[0].content
    assert "第二轮：补充订单号" in summaries[0].content
    summary_content = str(summaries[0].content)
    summary_payload = json.loads(summary_content[summary_content.index("{") :])
    assert set(summary_payload) == {
        "confirmed_facts",
        "current_request",
        "tool_results",
        "promises",
        "pending_actions",
    }
    assert summary_payload["current_request"] == "第二轮：补充订单号"


def test_agent_profile_tool_writes_cross_session_structured_memory(test_settings):
    repository = CustomerProfileRepository(test_settings.profile_db_file)
    tools = build_customer_profile_tools(repository)

    with build_customer_service_agent(
        test_settings,
        llm=FakeProfileSavingChatModel(),
        tools=tools,
        profile_repository=repository,
    ) as agent:
        response = agent.ask(
            "我叫张三，订单 ORDER-1001，以后优先短信联系",
            session_id="session-a",
            user_id="customer-42",
        )

    profile = repository.get("customer-42")
    assert response.answer == "客户信息已经记住。"
    assert profile is not None
    assert profile.customer_name == "张三"
    assert profile.order_ids == ["ORDER-1001"]
    assert profile.preferences == ["优先短信联系"]
    assert "runtime" not in tools[0].tool_call_schema.model_json_schema()["properties"]

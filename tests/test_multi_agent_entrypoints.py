import sys
import types

from customer_service_agent import cli, mcp_server
from customer_service_agent.agent import build_customer_service_agent
from tests.fakes import FakeEmbeddings
from tests.multi_fakes import CollaborationModel, task


def test_cli_trace_shows_multi_agent_tasks(test_settings, monkeypatch, capsys):
    settings = test_settings.model_copy(update={"agent_mode": "multi"})
    model = CollaborationModel(plan={"tasks": [task("order")]})
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(
        cli,
        "build_customer_service_agent",
        lambda s: build_customer_service_agent(s, llm=model, specialist_tools={}),
    )
    args = cli._build_parser().parse_args(
        [
            "ask",
            "查订单",
            "--session",
            "cli",
            "--request-id",
            "cli-r1",
            "--trace",
        ]
    )
    cli._run(args)
    output = capsys.readouterr().out
    assert "cli-r1" in output
    assert "order [order] success" in output
    assert "已汇总" in output


def test_mcp_wrapper_uses_multi_agent_and_returns_answer(test_settings, monkeypatch):
    # 验证项目包装层；不依赖可选 MCP 包，也不声称模拟了协议握手。
    settings = test_settings.model_copy(update={"agent_mode": "multi"})
    agent = build_customer_service_agent(settings, llm=CollaborationModel(), specialist_tools={})
    tools = {}
    responses = []

    class FakeMCP:
        def __init__(self, name):
            self.name = name

        def tool(self):
            def register(function):
                tools[function.__name__] = function
                return function

            return register

        def run(self):
            responses.append(tools["ask_customer_agent"]("你好", request_id="mcp-r1"))

    module = types.ModuleType("mcp.server.fastmcp")
    module.FastMCP = FakeMCP
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", module)
    monkeypatch.setattr(mcp_server, "get_settings", lambda: settings)
    monkeypatch.setattr(mcp_server, "build_customer_service_agent", lambda _: agent)
    monkeypatch.setattr(mcp_server, "build_embedding_model", lambda _: FakeEmbeddings())
    mcp_server.main()
    assert responses == ["已汇总演示查询结果。"]
    assert "search_customer_knowledge_base" in tools
    assert agent._closed

"""可选 MCP Server。

MCP 让其他客户端可以把你的 agent 当作工具调用。这个文件是“学习型骨架”：
安装 mcp 额外依赖后，可以用 `customer-agent-mcp` 启动。
"""

from customer_service_agent.agent import build_customer_service_agent
from customer_service_agent.config import get_settings
from customer_service_agent.rag.service import RAGService
from customer_service_agent.tools.rag_tool import format_evidence


def main() -> None:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise SystemExit("缺少 MCP 依赖。请运行：pip install -e '.[mcp]'") from exc

    settings = get_settings()
    agent = build_customer_service_agent(settings)
    rag_service = RAGService.from_settings(settings)
    server = FastMCP("customer-service-agent")

    @server.tool()
    def ask_customer_agent(
        message: str,
        session_id: str = "mcp",
        user_id: str | None = None,
        request_id: str | None = None,
    ) -> str:
        """向智能客服 agent 提问。"""

        # MCP 默认提供查询能力；写操作确认由 CLI/FastAPI 的调用方明确提交。
        return agent.ask(
            message, session_id=session_id, user_id=user_id, request_id=request_id
        ).answer

    @server.tool()
    def search_customer_knowledge_base(query: str, top_k: int = 3, force_web: bool = False) -> str:
        """混合检索客服知识库，并在需要时通过浏览器MCP查询公开网页。"""
        return format_evidence(rag_service.retrieve(query, top_k=top_k, force_web=force_web))

    try:
        server.run()
    finally:
        agent.close()
        rag_service.close()


if __name__ == "__main__":
    main()

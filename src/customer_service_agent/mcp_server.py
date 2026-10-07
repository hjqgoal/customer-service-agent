"""可选 MCP Server。

MCP 让其他客户端可以把你的 agent 当作工具调用。这个文件是“学习型骨架”：
安装 mcp 额外依赖后，可以用 `customer-agent-mcp` 启动。
"""

from customer_service_agent.agent import build_customer_service_agent
from customer_service_agent.config import get_settings
from customer_service_agent.model_factory import build_embedding_model
from customer_service_agent.rag.service import RAGService


def main() -> None:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise SystemExit("缺少 MCP 依赖。请运行：pip install -e '.[mcp]'") from exc

    settings = get_settings()
    agent = build_customer_service_agent(settings)
    rag_service = RAGService(settings, build_embedding_model(settings))
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
    def search_customer_knowledge_base(query: str, top_k: int = 4) -> str:
        """搜索客服知识库。"""

        hits = rag_service.search(query, top_k=top_k)
        if not hits:
            return "没有检索到相关内容。"
        return "\n\n".join(
            f"[{index}] score={hit.score:.3f}\n{hit.content}"
            for index, hit in enumerate(hits, start=1)
        )

    try:
        server.run()
    finally:
        agent.close()


if __name__ == "__main__":
    main()

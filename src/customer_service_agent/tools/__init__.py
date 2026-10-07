"""LangChain tools used by the customer-service agent."""

from langchain_core.tools import BaseTool

from customer_service_agent.config import Settings
from customer_service_agent.memory import CustomerProfileRepository
from customer_service_agent.rag.service import RAGService
from customer_service_agent.tools.order_tool import build_order_lookup_tool
from customer_service_agent.tools.policy_tool import build_policy_lookup_tool
from customer_service_agent.tools.profile_tool import build_customer_profile_tools
from customer_service_agent.tools.rag_tool import build_rag_search_tool
from customer_service_agent.tools.skill_tool import build_skill_loader_tool
from customer_service_agent.tools.ticket_tool import build_ticket_tool


def build_tools(
    settings: Settings,
    rag_service: RAGService | None = None,
    profile_repository: CustomerProfileRepository | None = None,
) -> list[BaseTool]:
    """统一创建 agent 可用工具。"""

    tools: list[BaseTool] = [
        build_order_lookup_tool(),
        build_policy_lookup_tool(),
        build_ticket_tool(settings),
        build_skill_loader_tool(),
    ]

    if rag_service is not None:
        tools.append(build_rag_search_tool(rag_service))
    if profile_repository is not None:
        tools.extend(build_customer_profile_tools(profile_repository))

    return tools


__all__ = ["build_tools"]

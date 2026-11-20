"""为每个专业 Agent 分配最小工具集合；query 任务拿不到售后写工具。"""

import json
from dataclasses import asdict

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool

from customer_service_agent.config import Settings
from customer_service_agent.memory import AgentContext, CustomerProfileRepository
from customer_service_agent.multi_agent.prompts import (
    SPECIALIST_BASE_PROMPT,
    SPECIALIST_PROMPTS,
    SPECIALIST_RULES,
    SPECIALIST_SKILLS,
)
from customer_service_agent.multi_agent.state import AgentName
from customer_service_agent.rag.service import RAGService
from customer_service_agent.skills import SkillRegistry
from customer_service_agent.tools.after_sales_tool import build_after_sales_tools
from customer_service_agent.tools.logistics_tool import build_logistics_tool
from customer_service_agent.tools.order_tool import DEFAULT_ORDER_REPOSITORY
from customer_service_agent.tools.policy_tool import build_policy_lookup_tool
from customer_service_agent.tools.profile_tool import build_customer_profile_tools
from customer_service_agent.tools.rag_tool import build_rag_search_tool

WRITE_TOOLS = {"submit_after_sales_request", "create_support_ticket"}


def build_specialist_prompt(
    settings: Settings, name: AgentName, registry: SkillRegistry | None = None
) -> str:
    """基础规则 + 角色职责 + 任务规则 + 本角色完整 SOP。"""

    skills = registry or SkillRegistry()
    sops = "\n\n---\n\n".join(
        skills.get(skill_id).to_prompt_text() for skill_id in SPECIALIST_SKILLS[name]
    )
    base = SPECIALIST_BASE_PROMPT.format(
        company_name=settings.company_name,
        support_email=settings.support_email,
    )
    return f"{base}\n\n{SPECIALIST_PROMPTS[name]}\n{SPECIALIST_RULES}\n\n本角色 SOP：\n\n{sops}"


def build_specialist_tools(
    settings: Settings,
    rag: RAGService,
    profiles: CustomerProfileRepository,
) -> dict[AgentName, list[BaseTool]]:
    @tool
    def lookup_order(order_id: str) -> str:
        """查询演示订单，返回结构化订单状态、支付标记和运单号。"""
        number = order_id.strip().upper()
        record = DEFAULT_ORDER_REPOSITORY.get(number)
        return json.dumps(
            {
                "source": "demo",
                "order_id": number,
                "found": record is not None,
                "order": asdict(record) if record else None,
            },
            ensure_ascii=False,
        )

    # SOP 已写进各自的系统提示词，专业 Agent 不再需要 load_skill_instructions。
    # 订单/物流事实只能来自各自的查询工具：知识库里也有编号相近的演示订单和运单，
    # 让这两个 Agent 检索知识库会把另一笔订单的商品和轨迹混进来。知识库只留给售后查政策。
    return {
        "order": [lookup_order, *build_customer_profile_tools(profiles)],
        "logistics": [build_logistics_tool()],
        "after_sales": [
            build_policy_lookup_tool(),
            build_rag_search_tool(rag),
            *build_after_sales_tools(settings),
        ],
    }


def build_specialists(
    settings: Settings, model: BaseChatModel, tools_by_agent: dict[AgentName, list[BaseTool]]
) -> dict:
    agents = {}
    registry = SkillRegistry()
    for name in SPECIALIST_PROMPTS:
        system_prompt = build_specialist_prompt(settings, name, registry)
        kinds = ("query", "action") if name == "after_sales" else ("query",)
        for kind in kinds:
            tools = [
                t
                for t in tools_by_agent.get(name, [])
                if kind == "action" or t.name not in WRITE_TOOLS
            ]
            agents[(name, kind)] = create_agent(
                model=model,
                tools=tools,
                context_schema=AgentContext,
                system_prompt=system_prompt,
                # 父图保存任务结果；子 Agent 每次只接收本任务所需的上下文。
                checkpointer=False,
                name=f"{name}_{kind}",
            )
    return agents

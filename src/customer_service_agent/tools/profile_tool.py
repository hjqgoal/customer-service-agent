"""LangChain tool for writing structured long-term customer memory."""

from typing import Annotated

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import BaseTool
from pydantic import Field

from customer_service_agent.memory.profile import (
    AgentContext,
    CustomerProfileRepository,
    CustomerProfileUpdate,
)


def build_customer_profile_tools(repository: CustomerProfileRepository) -> list[BaseTool]:
    """Build tools as closures so persistence stays dependency-injected and testable."""

    @tool
    def save_customer_profile(
        runtime: ToolRuntime[AgentContext],
        customer_name: Annotated[str | None, Field(description="用户明确提供的姓名或称呼")] = None,
        phone: Annotated[str | None, Field(description="用户明确提供的联系电话")] = None,
        email: Annotated[str | None, Field(description="用户明确提供的电子邮箱")] = None,
        order_ids: Annotated[
            list[str] | None, Field(description="本次对话确认相关的订单号列表")
        ] = None,
        issue_summary: Annotated[
            str | None, Field(description="当前仍需处理的客服问题简要描述")
        ] = None,
        preferences: Annotated[
            list[str] | None, Field(description="用户明确表达的稳定服务偏好")
        ] = None,
    ) -> str:
        """保存用户明确提供的重要客户信息；只传需要新增或修正的字段。"""

        update = CustomerProfileUpdate(
            customer_name=customer_name,
            phone=phone,
            email=email,
            order_ids=order_ids,
            issue_summary=issue_summary,
            preferences=preferences,
        )
        supplied_fields = update.model_dump(exclude_none=True)
        if not supplied_fields:
            return "没有提供可保存的客户信息。"

        repository.upsert(runtime.context.user_id, update)
        field_names = "、".join(supplied_fields)
        return f"客户信息已保存，更新字段：{field_names}。"

    return [save_customer_profile]


__all__ = ["build_customer_profile_tools"]

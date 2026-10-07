"""Order lookup tool with an injectable repository boundary."""

from dataclasses import dataclass
from typing import Protocol

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class OrderLookupInput(BaseModel):
    order_id: str = Field(
        ...,
        min_length=1,
        max_length=80,
        description="用户提供的订单号，例如 ORDER-1001",
    )


@dataclass(frozen=True)
class OrderRecord:
    status: str
    logistics: str
    eta: str
    tracking_number: str | None = None
    paid: bool = True


class OrderRepository(Protocol):
    def get(self, order_id: str) -> OrderRecord | None: ...


class DemoOrderRepository:
    """Local demo adapter. Replace it with an HTTP/database adapter in production."""

    _orders = {
        "ORDER-1001": OrderRecord(
            "已发货", "顺丰 SF123456789", "预计 2 个工作日内送达", "SF123456789"
        ),
        "ORDER-1002": OrderRecord("待付款", "暂无物流", "付款后 24 小时内安排发货", paid=False),
        "ORDER-1003": OrderRecord(
            "退款处理中",
            "用户已退回，仓库待验收",
            "验收后 1 到 3 个工作日原路退款",
            "SFRETURN1003",
        ),
    }

    def get(self, order_id: str) -> OrderRecord | None:
        return self._orders.get(order_id)


DEFAULT_ORDER_REPOSITORY = DemoOrderRepository()


def lookup_order(order_id: str, repository: OrderRepository = DEFAULT_ORDER_REPOSITORY) -> str:
    normalized_id = order_id.strip().upper()
    order = repository.get(normalized_id)
    if not order:
        return (
            f"没有查到订单 {normalized_id}。请确认订单号是否正确；如果用户确认无误，"
            "建议创建工单转人工核查。"
        )
    return (
        f"订单 {normalized_id} 当前状态：{order.status}；"
        f"物流信息：{order.logistics}；预计进展：{order.eta}。"
    )


def build_order_lookup_tool(
    repository: OrderRepository = DEFAULT_ORDER_REPOSITORY,
) -> StructuredTool:
    def lookup(order_id: str) -> str:
        return lookup_order(order_id, repository)

    return StructuredTool.from_function(
        func=lookup,
        name="lookup_order",
        description="查询订单状态、物流单号和预计进展。用户提供订单号时优先使用。",
        args_schema=OrderLookupInput,
    )

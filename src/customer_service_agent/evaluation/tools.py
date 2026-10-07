"""可直接传给现有 Agent 的只读测试工具，不会提交售后申请或真实退款。

user_id 必须由可信测试端传入，而不是模型工具参数。这是夹具级数据隔离，
不是完整的登录认证；评估真实 API 的鉴权时仍要另外搭建安全测试环境。
"""

import json
from pathlib import Path

from langchain_core.tools import BaseTool, StructuredTool

from customer_service_agent.evaluation.store import FixtureStore
from customer_service_agent.tools.order_tool import OrderRecord


class SQLiteFixtureOrders:
    """实现现有 OrderRepository.get 契约，可注入原项目的订单/售后工具。"""

    def __init__(self, store: FixtureStore, user_id: str):
        self.store, self.user_id = store, user_id

    def get(self, order_id: str) -> OrderRecord | None:
        record = self.store.get("order", order_id)
        if record is None or record["customer_id"] != self.user_id:
            return None
        fields = ("status", "logistics", "eta", "tracking_number", "paid")
        return OrderRecord(**{field: record[field] for field in fields})


def build_fixture_tools(path: Path, *, user_id: str = "EVAL-USER-A") -> list[BaseTool]:
    store = FixtureStore(path)

    def response(**fields) -> str:
        return json.dumps(
            {"source": "synthetic_evaluation", "as_of": "2026-09-23T12:00:00+08:00", **fields},
            ensure_ascii=False,
        )

    def lookup_order(order_id: str) -> str:
        """查询虚构测试订单及金额、签收天数；仅返回当前测试用户的订单。"""
        order = store.get("order", order_id)
        visible = order if order and order["customer_id"] == user_id else None
        return response(found=visible is not None, order=visible)

    def lookup_logistics(tracking_number: str) -> str:
        """查询当前测试用户的虚构物流轨迹，不代表任何真实快递状态。"""
        number = tracking_number.strip().upper()
        owned = any(
            o["customer_id"] == user_id and o["tracking_number"] == number
            for o in store.records("order")
        )
        record = store.get("logistics", number) if owned else None
        return response(found=record is not None, logistics=record)

    def lookup_policy(topic: str) -> str:
        """按规则编号或关键词查询虚构政策；这些规则不是现实商家的政策或法律意见。"""
        topic = topic.strip().upper()
        matches = [
            p
            for p in store.records("policy")
            if topic and (p["id"] == topic or any(word.upper() in topic for word in p["keywords"]))
        ]
        return response(found=bool(matches), policies=matches)

    return [
        StructuredTool.from_function(func=func)
        for func in (lookup_order, lookup_logistics, lookup_policy)
    ]

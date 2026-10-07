"""物流工具的演示适配器。接真实接口时保留工具契约，替换 repository。"""

import json
from typing import Protocol

from langchain.tools import tool
from langchain_core.tools import BaseTool


class LogisticsRepository(Protocol):
    def get(self, tracking_number: str) -> dict | None: ...


class DemoLogisticsRepository:
    _records = {
        "SF123456789": {
            "status": "运输中",
            "events": ["商家已交寄", "已到达中转站"],
            "eta": "演示预计 2 个工作日内送达",
        },
        "SFRETURN1003": {
            "status": "退货已签收",
            "events": ["用户寄回", "仓库已签收，等待验收"],
            "eta": "演示验收后 1 到 3 个工作日处理",
        },
    }

    def get(self, tracking_number: str) -> dict | None:
        return self._records.get(tracking_number)


def build_logistics_tool(repository: LogisticsRepository | None = None) -> BaseTool:
    repository = repository or DemoLogisticsRepository()

    @tool
    def lookup_logistics(tracking_number: str) -> str:
        """使用已核实的运单号查询物流轨迹；当前数据为演示，不代表真实运输状态。"""
        number = tracking_number.strip().upper()
        record = repository.get(number)
        return json.dumps(
            {
                "source": "demo",
                "tracking_number": number,
                "found": record is not None,
                "logistics": record,
            },
            ensure_ascii=False,
        )

    return lookup_logistics

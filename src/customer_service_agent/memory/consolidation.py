"""长期记忆的确定性提炼：只处理已确认事实，不猜测新事实。

聊天模型通过 save_customer_profile 提取事实；本模块负责合并、去重、
限量及覆盖生成摘要。不让模型改写手机号、订单号等关键值。
"""

import json
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class ProfileMemoryPolicy:
    max_orders: int = 20
    max_preferences: int = 10
    summary_max_chars: int = 800
    max_bytes: int = 8192

    @classmethod
    def from_settings(cls, settings):
        return cls(
            max_orders=settings.memory_profile_max_orders,
            max_preferences=settings.memory_profile_max_preferences,
            summary_max_chars=settings.memory_profile_summary_max_chars,
            max_bytes=settings.memory_profile_max_bytes,
        )

    @staticmethod
    def recent_unique(
        values: list[str],
        *,
        limit: int,
        item_chars: int,
        normalize_text: bool = True,
    ) -> list[str]:
        # 重提同一事实会刷新位置；满额时淘汰最早的，而不是丢弃新事实。
        recent: dict[str, str] = {}
        for value in values:
            value = (
                " ".join(unicodedata.normalize("NFKC", value).split())
                if normalize_text
                else value.strip()
            )[:item_chars]
            if value:
                recent.pop(value, None)
                recent[value] = value
        return list(recent.values())[-limit:]

    def consolidate(self, values: dict) -> dict:
        facts = dict(values)
        facts["order_ids"] = self.recent_unique(
            facts.get("order_ids", []),
            limit=self.max_orders,
            item_chars=100,
            normalize_text=False,
        )
        facts["preferences"] = self.recent_unique(
            facts.get("preferences", []), limit=self.max_preferences, item_chars=200
        )
        facts["issue_summary"] = (facts.get("issue_summary") or "")[: self.summary_max_chars]

        def refresh_summary():
            parts = []
            if facts["issue_summary"]:
                issue_limit = (
                    self.summary_max_chars // 2 if facts["preferences"] else self.summary_max_chars
                )
                parts.append("最近待处理问题：" + facts["issue_summary"][:issue_limit])
            if facts["preferences"]:
                parts.append("已确认服务偏好：" + "；".join(facts["preferences"]))
            facts["memory_summary"] = "\n".join(parts)[: self.summary_max_chars]

        def byte_size():
            return len(json.dumps(facts, ensure_ascii=False).encode("utf-8"))

        refresh_summary()
        # 预算针对画像业务载荷，不包括SQLite页、索引和checkpoint。
        while byte_size() > self.max_bytes:
            if facts["preferences"]:
                facts["preferences"].pop(0)
            elif facts["order_ids"]:
                facts["order_ids"].pop(0)
            elif facts["issue_summary"]:
                facts["issue_summary"] = facts["issue_summary"][:-100]
            else:
                raise ValueError("身份字段超出画像预算；请提高 MEMORY_PROFILE_MAX_BYTES。")
            refresh_summary()
        return facts

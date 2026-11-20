"""Structured long-term customer memory backed by SQLite."""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from customer_service_agent.infrastructure.sqlite import sqlite_connection
from customer_service_agent.memory.consolidation import ProfileMemoryPolicy


@dataclass(frozen=True)  # dataclass自动生成__init__，frozen True表示不可更改
class AgentContext:
    """Immutable request context passed to LangChain tools and middleware."""

    user_id: str
    # 以下字段由多 Agent 服务注入；单 Agent 只需要 user_id，因此都给默认值。
    session_id: str = "default"
    request_id: str | None = None  # 业务请求 ID，售后写操作用它生成幂等键
    confirm_action: bool = False  # 调用方是否确认本次写操作
    task_id: str | None = None  # 当前子任务 ID，只在专业 Agent 内有值


class CustomerProfileUpdate(BaseModel):
    """Whitelisted customer facts that the agent is allowed to remember."""

    # 禁止传入为定义的字段，自动去除space
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    customer_name: str | None = Field(default=None, max_length=100)
    phone: str | None = Field(default=None, max_length=50)
    email: str | None = Field(default=None, max_length=254)
    order_ids: list[str] | None = Field(default=None, max_length=20)
    issue_summary: str | None = Field(default=None, max_length=1000)
    preferences: list[str] | None = Field(default=None, max_length=20)


class CustomerProfile(BaseModel):
    """Persisted long-term profile shared by all sessions of one user."""

    user_id: str
    customer_name: str | None = None
    phone: str | None = None
    email: str | None = None
    order_ids: list[str] = Field(default_factory=list)
    issue_summary: str | None = None
    preferences: list[str] = Field(default_factory=list)
    memory_summary: str = ""
    created_at: datetime
    updated_at: datetime

    def prompt_payload(self) -> dict[str, str | list[str]]:  # 转换成适合放进 LLM 提示词的数据
        """Return only business facts useful to the model, without database metadata."""

        values: dict[str, str | list[str] | None] = {
            "customer_name": self.customer_name,
            "phone": self.phone,
            "email": self.email,
            "order_ids": self.order_ids,
            "issue_summary": self.issue_summary,
            "preferences": self.preferences,
        }
        if self.memory_summary:
            # 问题进入摘要；偏好仍保留结构化值，避免摘要截断后漏掉重要偏好。
            values.pop("issue_summary")
            values["memory_summary"] = self.memory_summary
        # 过滤空值
        return {key: value for key, value in values.items() if value not in (None, "", [])}


class CustomerProfileRepository:
    """SQLite repository for cross-session structured customer facts.

    This is intentionally separate from LangGraph checkpoints: deleting a chat
    thread must not silently delete a customer's long-term profile.
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        busy_timeout_seconds: float = 30.0,
        policy: ProfileMemoryPolicy | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.busy_timeout_seconds = busy_timeout_seconds
        self.policy = policy or ProfileMemoryPolicy()
        self._ensure_table()

    def _connection(self):  # 连接SQLite的方法
        return sqlite_connection(
            self.db_path,
            busy_timeout_seconds=self.busy_timeout_seconds,
        )

    def _ensure_table(self) -> None:  # 创建数据库的表
        with self._connection() as connection:  # 实际连接SQLite
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS customer_profiles (
                    user_id TEXT PRIMARY KEY,
                    customer_name TEXT,
                    phone TEXT,
                    email TEXT,
                    order_ids_json TEXT NOT NULL DEFAULT '[]',
                    issue_summary TEXT,
                    preferences_json TEXT NOT NULL DEFAULT '[]',
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            # 兼容已有SQLite文件，不需要删库重建。
            columns = {row[1] for row in connection.execute("PRAGMA table_info(customer_profiles)")}
            if "memory_summary" not in columns:
                connection.execute(
                    "ALTER TABLE customer_profiles "
                    "ADD COLUMN memory_summary TEXT NOT NULL DEFAULT ''"
                )

    @staticmethod  # 静态方法，只用于类的内部，并且不需要创建实例使用，不需要使用类的属性
    def _normalize_user_id(user_id: str) -> str:
        normalized = user_id.strip()
        if not normalized or len(normalized) > 128:
            raise ValueError("user_id 不能为空且不能超过 128 个字符。")
        return normalized

    @staticmethod  # 清洗去重后的列表
    def _clean_list(values: list[str] | None, *, max_item_length: int = 200) -> list[str]:
        if values is None:
            return []
        cleaned: list[str] = []
        for value in values:
            normalized = value.strip()[:max_item_length]
            if normalized and normalized not in cleaned:  # 重复的就不加入cleaned列表
                cleaned.append(normalized)
        return cleaned

    @staticmethod
    def _row_to_profile(row) -> CustomerProfile:
        return CustomerProfile(
            user_id=row[0],
            customer_name=row[1],
            phone=row[2],
            email=row[3],
            order_ids=json.loads(row[4]),  # json.dumps/ json.loads json加载成列表
            issue_summary=row[5],
            preferences=json.loads(row[6]),
            created_at=row[7],
            updated_at=row[8],
            memory_summary=row[9],
        )

    """SQL查询结果：
    (
    "user_001",
    "张三",
    "13800000000",
    "zhangsan@example.com",
    '["ORDER-001"]',
    "咨询退款问题",
    '["喜欢电话联系"]',
    created_at,
    updated_at,
    )
    """

    def get(self, user_id: str) -> CustomerProfile | None:
        normalized_user_id = self._normalize_user_id(user_id)
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT user_id, customer_name, phone, email, order_ids_json,
                       issue_summary, preferences_json, created_at, updated_at, memory_summary
                FROM customer_profiles
                WHERE user_id = ?
                """,
                (normalized_user_id,),
            ).fetchone()
        return self._row_to_profile(row) if row else None

    def upsert(
        self, user_id: str, update: CustomerProfileUpdate
    ) -> CustomerProfile:  # 存在就update 不存在就insert
        """Merge supplied facts; omitted scalar fields remain unchanged."""

        normalized_user_id = self._normalize_user_id(user_id)

        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")  # 立即获取写事务锁
            row = connection.execute(
                """
                SELECT user_id, customer_name, phone, email, order_ids_json,
                       issue_summary, preferences_json, created_at, updated_at, memory_summary
                FROM customer_profiles
                WHERE user_id = ?
                """,
                (normalized_user_id,),
            ).fetchone()
            current = self._row_to_profile(row) if row else None

            current_order_ids = current.order_ids if current else []
            current_preferences = current.preferences if current else []

            def merged(value: str | None, existing: str | None) -> str | None:
                # 新值不是 None 或空字符串：使用新值。
                return value if value not in (None, "") else existing

            facts = self.policy.consolidate(
                {
                    "customer_name": merged(
                        update.customer_name, current.customer_name if current else None
                    ),
                    "phone": merged(update.phone, current.phone if current else None),
                    "email": merged(update.email, current.email if current else None),
                    "order_ids": [*current_order_ids, *(update.order_ids or [])],
                    "preferences": [*current_preferences, *(update.preferences or [])],
                    "issue_summary": merged(
                        update.issue_summary, current.issue_summary if current else None
                    ),
                }
            )
            connection.execute(
                """
                INSERT INTO customer_profiles(
                    user_id, customer_name, phone, email, order_ids_json,
                    issue_summary, preferences_json, memory_summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    customer_name = excluded.customer_name,
                    phone = excluded.phone,
                    email = excluded.email,
                    order_ids_json = excluded.order_ids_json,
                    issue_summary = excluded.issue_summary,
                    preferences_json = excluded.preferences_json,
                    memory_summary = excluded.memory_summary,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    normalized_user_id,
                    facts["customer_name"],
                    facts["phone"],
                    facts["email"],
                    json.dumps(facts["order_ids"], ensure_ascii=False),
                    facts["issue_summary"],
                    json.dumps(facts["preferences"], ensure_ascii=False),
                    facts["memory_summary"],
                ),
            )

        # 前面的 SQL 只负责写入，没有直接返回完整对象。
        # 写入成功后，再查询一次数据库，得到最终保存的数据
        profile = self.get(normalized_user_id)
        if profile is None:  # pragma: no cover - defensive database invariant
            raise RuntimeError("客户画像保存失败。")
        return profile

    def delete(self, user_id: str) -> bool:
        normalized_user_id = self._normalize_user_id(user_id)
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM customer_profiles WHERE user_id = ?",
                (normalized_user_id,),
            )
        return cursor.rowcount > 0

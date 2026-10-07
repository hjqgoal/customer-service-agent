"""Structured long-term customer memory backed by SQLite."""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from customer_service_agent.infrastructure.sqlite import sqlite_connection


@dataclass(frozen=True)     #dataclass自动生成__init__，frozen True表示不可更改
class AgentContext:
    """Immutable request context passed to LangChain tools and middleware."""

    user_id: str


class CustomerProfileUpdate(BaseModel):
    """Whitelisted customer facts that the agent is allowed to remember."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)    #禁止传入为定义的字段，自动去除space

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
    created_at: datetime
    updated_at: datetime

    def prompt_payload(self) -> dict[str, str | list[str]]:     #转换成适合放进 LLM 提示词的数据
        """Return only business facts useful to the model, without database metadata."""

        values: dict[str, str | list[str] | None] = {
            "customer_name": self.customer_name,
            "phone": self.phone,
            "email": self.email,
            "order_ids": self.order_ids,
            "issue_summary": self.issue_summary,
            "preferences": self.preferences,
        }
        return {key: value for key, value in values.items() if value not in (None, "", [])} #过滤空值


class CustomerProfileRepository:
    """SQLite repository for cross-session structured customer facts.

    This is intentionally separate from LangGraph checkpoints: deleting a chat
    thread must not silently delete a customer's long-term profile.
    """

    def __init__(self, db_path: str | Path, *, busy_timeout_seconds: float = 30.0) -> None:
        self.db_path = Path(db_path)
        self.busy_timeout_seconds = busy_timeout_seconds
        self._ensure_table()

    def _connection(self):                                  #连接SQLite的方法
        return sqlite_connection(
            self.db_path,
            busy_timeout_seconds=self.busy_timeout_seconds,
        )

    def _ensure_table(self) -> None:                        #创建数据库的表
        with self._connection() as connection:              #实际连接SQLite
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

    @staticmethod   #静态方法，只用于类的内部，并且不需要创建实例使用，不需要使用类的属性
    def _normalize_user_id(user_id: str) -> str:
        normalized = user_id.strip()
        if not normalized or len(normalized) > 128:
            raise ValueError("user_id 不能为空且不能超过 128 个字符。")
        return normalized

    @staticmethod   #清洗去重后的列表
    def _clean_list(values: list[str] | None, *, max_item_length: int = 200) -> list[str]:
        if values is None:
            return []
        cleaned: list[str] = []
        for value in values:
            normalized = value.strip()[:max_item_length]
            if normalized and normalized not in cleaned:    #重复的就不加入cleaned列表
                cleaned.append(normalized)
        return cleaned

    @staticmethod
    def _row_to_profile(row) -> CustomerProfile:
        return CustomerProfile(
            user_id=row[0],
            customer_name=row[1],
            phone=row[2],
            email=row[3],
            order_ids=json.loads(row[4]),       #json.dumps/ json.loads json加载成列表
            issue_summary=row[5],
            preferences=json.loads(row[6]),
            created_at=row[7],
            updated_at=row[8],
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
                       issue_summary, preferences_json, created_at, updated_at
                FROM customer_profiles
                WHERE user_id = ?
                """,
                (normalized_user_id,),
            ).fetchone()
        return self._row_to_profile(row) if row else None

    def upsert(self, user_id: str, update: CustomerProfileUpdate) -> CustomerProfile:   #存在就update 不存在就insert
        """Merge supplied facts; omitted scalar fields remain unchanged."""

        normalized_user_id = self._normalize_user_id(user_id)

        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")   #立即获取写事务锁   
            row = connection.execute(
                """
                SELECT user_id, customer_name, phone, email, order_ids_json,
                       issue_summary, preferences_json, created_at, updated_at
                FROM customer_profiles
                WHERE user_id = ?
                """,
                (normalized_user_id,),
            ).fetchone()
            current = self._row_to_profile(row) if row else None

            current_order_ids = current.order_ids if current else []
            current_preferences = current.preferences if current else []
            order_ids = self._clean_list(
                [*current_order_ids, *self._clean_list(update.order_ids, max_item_length=100)],
                max_item_length=100,
            )[:20]
            preferences = self._clean_list(
                [*current_preferences, *self._clean_list(update.preferences)],
            )[:20]

            def merged(value: str | None, existing: str | None) -> str | None:
                return value if value not in (None, "") else existing       #新值不是 None 或空字符串：使用新值。

            connection.execute(
                """
                INSERT INTO customer_profiles(
                    user_id, customer_name, phone, email, order_ids_json,
                    issue_summary, preferences_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    customer_name = excluded.customer_name,
                    phone = excluded.phone,
                    email = excluded.email,
                    order_ids_json = excluded.order_ids_json,
                    issue_summary = excluded.issue_summary,
                    preferences_json = excluded.preferences_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    normalized_user_id,
                    merged(update.customer_name, current.customer_name if current else None),
                    merged(update.phone, current.phone if current else None),
                    merged(update.email, current.email if current else None),
                    json.dumps(order_ids, ensure_ascii=False),
                    merged(update.issue_summary, current.issue_summary if current else None),
                    json.dumps(preferences, ensure_ascii=False),
                ),
            )

        profile = self.get(normalized_user_id)  #前面的 SQL 只负责写入，没有直接返回完整对象。写入成功后，再查询一次数据库，得到最终保存的数据
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

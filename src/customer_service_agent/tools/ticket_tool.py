"""Support ticket tool backed by a local SQLite repository."""

import uuid
from pathlib import Path
from typing import Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from customer_service_agent.config import Settings
from customer_service_agent.infrastructure.sqlite import sqlite_connection

TicketPriority = Literal["low", "normal", "high"]


class TicketInput(BaseModel):
    issue_type: str = Field(..., min_length=1, max_length=50, description="问题类型")
    summary: str = Field(..., min_length=1, max_length=1000, description="用户问题摘要")
    contact: str = Field(default="", max_length=200, description="用户联系方式，没有就留空")
    priority: TicketPriority = "normal"


class TicketRepository:
    def __init__(self, db_path: str | Path, *, busy_timeout_seconds: float = 30.0) -> None:
        self.db_path = Path(db_path)
        self.busy_timeout_seconds = busy_timeout_seconds
        self._ensure_table()

    def _connection(self):
        return sqlite_connection(
            self.db_path,
            busy_timeout_seconds=self.busy_timeout_seconds,
        )

    def _ensure_table(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS tickets (
                    id TEXT PRIMARY KEY,
                    issue_type TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    contact TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS ticket_operations "
                "(operation_key TEXT PRIMARY KEY, ticket_id TEXT NOT NULL REFERENCES tickets(id))"
            )

    def create(
        self,
        issue_type: str,
        summary: str,
        contact: str,
        priority: TicketPriority,
        *,
        operation_key: str | None = None,
    ) -> str:
        ticket_id = f"TICKET-{uuid.uuid4().hex[:8].upper()}"
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if operation_key:
                existing = connection.execute(
                    "SELECT ticket_id FROM ticket_operations WHERE operation_key = ?",
                    (operation_key,),
                ).fetchone()
                if existing:
                    return str(existing[0])
            connection.execute(
                """
                INSERT INTO tickets(id, issue_type, summary, contact, priority, status)
                VALUES (?, ?, ?, ?, ?, 'open')
                """,
                (ticket_id, issue_type, summary, contact, priority),
            )
            if operation_key:
                connection.execute(
                    "INSERT INTO ticket_operations(operation_key, ticket_id) VALUES (?, ?)",
                    (operation_key, ticket_id),
                )
        return ticket_id


def build_ticket_tool(settings: Settings) -> StructuredTool:
    repository = TicketRepository(
        settings.ticket_db_file,
        busy_timeout_seconds=settings.sqlite_busy_timeout_seconds,
    )

    def create_ticket(
        issue_type: str,
        summary: str,
        contact: str = "",
        priority: TicketPriority = "normal",
    ) -> str:
        """创建人工跟进工单。"""

        ticket_id = repository.create(issue_type, summary, contact, priority)
        return (
            f"已创建工单 {ticket_id}，问题类型：{issue_type}，优先级：{priority}。"
            "请告知用户我们会尽快安排人工客服跟进。"
        )

    return StructuredTool.from_function(
        func=create_ticket,
        name="create_support_ticket",
        description="当问题需要人工处理、投诉升级或信息不足时，创建客服工单。",
        args_schema=TicketInput,
    )

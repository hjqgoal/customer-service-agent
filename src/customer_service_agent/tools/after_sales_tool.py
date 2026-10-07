"""模拟售后申请：确认由程序检查，幂等由数据库保证，不由模型口头保证。"""

import hashlib
import json
import uuid
from pathlib import Path
from typing import Literal

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import BaseTool

from customer_service_agent.config import Settings
from customer_service_agent.infrastructure.sqlite import sqlite_connection
from customer_service_agent.memory import AgentContext
from customer_service_agent.tools.order_tool import DEFAULT_ORDER_REPOSITORY, OrderRepository
from customer_service_agent.tools.ticket_tool import TicketPriority, TicketRepository

RequestType = Literal["refund", "return", "exchange"]


def operation_key(context: AgentContext, *parts: str) -> str:
    # 不使用模型生成的 tool_call_id：重试时它会变化，不能用于防重复。
    if not context.request_id:
        raise ValueError("写操作必须携带 request_id")
    payload = [context.user_id, context.session_id, context.request_id, *parts]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


class AfterSalesRepository:
    def __init__(self, path: Path, *, busy_timeout_seconds: float = 30) -> None:
        self.path = path
        self.timeout = busy_timeout_seconds
        with self._connection() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS after_sales_requests (
                    id TEXT PRIMARY KEY, operation_key TEXT NOT NULL UNIQUE,
                    user_id TEXT NOT NULL, order_id TEXT NOT NULL, request_type TEXT NOT NULL,
                    reason TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'submitted',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            connection.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS one_open_after_sales_request
                ON after_sales_requests(user_id, order_id, request_type)
                WHERE status = 'submitted'
            """)

    def _connection(self):
        return sqlite_connection(self.path, busy_timeout_seconds=self.timeout)

    def create(
        self, *, key: str, user_id: str, order_id: str, request_type: RequestType, reason: str
    ) -> str:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT id FROM after_sales_requests WHERE operation_key = ? OR "
                "(user_id = ? AND order_id = ? AND request_type = ? AND status = 'submitted')",
                (key, user_id, order_id, request_type),
            ).fetchone()
            if row:
                return str(row[0])
            application_id = f"DEMO-AS-{uuid.uuid4().hex[:12].upper()}"
            connection.execute(
                "INSERT INTO after_sales_requests "
                "(id, operation_key, user_id, order_id, request_type, reason) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (application_id, key, user_id, order_id, request_type, reason),
            )
            return application_id


def build_after_sales_tools(
    settings: Settings,
    orders: OrderRepository = DEFAULT_ORDER_REPOSITORY,
) -> list[BaseTool]:
    applications = AfterSalesRepository(
        settings.after_sales_db_file,
        busy_timeout_seconds=settings.sqlite_busy_timeout_seconds,
    )
    tickets = TicketRepository(
        settings.ticket_db_file,
        busy_timeout_seconds=settings.sqlite_busy_timeout_seconds,
    )

    @tool
    def submit_after_sales_request(
        order_id: str,
        request_type: RequestType,
        reason: str,
        runtime: ToolRuntime[AgentContext],
    ) -> str:
        """创建已由用户确认的模拟售后申请，不执行真实退款、不代表审核通过。"""
        if not runtime.context.confirm_action:
            return json.dumps({"status": "confirmation_required", "message": "请先确认提交申请"})
        normalized = order_id.strip().upper()
        order = orders.get(normalized)
        if order is None or not order.paid or order.status == "退款处理中":
            return json.dumps(
                {"status": "rejected", "message": "订单不存在、未支付或已有退款处理中"},
                ensure_ascii=False,
            )
        if not reason.strip() or len(reason) > 1000:
            raise ValueError("售后原因应为 1 到 1000 字符")
        application_id = applications.create(
            key=operation_key(runtime.context, "after_sales", normalized, request_type),
            user_id=runtime.context.user_id,
            order_id=normalized,
            request_type=request_type,
            reason=reason.strip(),
        )
        return json.dumps(
            {
                "source": "demo",
                "status": "submitted",
                "application_id": application_id,
                "message": "模拟申请已受理，等待审核，未发生真实退款",
            },
            ensure_ascii=False,
        )

    @tool
    def create_support_ticket(
        issue_type: str,
        summary: str,
        runtime: ToolRuntime[AgentContext],
        contact: str = "",
        priority: TicketPriority = "normal",
    ) -> str:
        """经确认创建人工工单；同一请求重试返回同一工单，不重复创建。"""
        if not runtime.context.confirm_action:
            return "请先确认创建工单。"
        if (
            not issue_type.strip()
            or len(issue_type) > 50
            or not summary.strip()
            or len(summary) > 1000
        ):
            raise ValueError("工单类型或摘要长度无效")
        ticket_id = tickets.create(
            issue_type,
            summary,
            contact[:200],
            priority,
            operation_key=operation_key(runtime.context, "ticket"),
        )
        return f"已创建本地演示工单 {ticket_id}，尚未连接真实人工客服系统。"

    return [submit_after_sales_request, create_support_ticket]

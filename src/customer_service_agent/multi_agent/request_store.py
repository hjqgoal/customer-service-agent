"""请求回执与会话归属。业务重试 ID 和 LangGraph 的 thread_id 是不同概念。"""

import hashlib
import json
from pathlib import Path

from customer_service_agent.infrastructure.sqlite import sqlite_connection
from customer_service_agent.models import ChatResponse


class RequestConflictError(ValueError):
    """同一个请求 ID 的内容变化，或同一会话还有未完成任务。"""


class SessionOwnershipError(ValueError):
    """会话已属于其他用户；生产环境 user_id 应来自认证系统。"""


class RequestStore:
    def __init__(self, path: Path, timeout: float) -> None:
        self.path, self.timeout = path, timeout
        with self._connection() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS workflow_sessions "
                "(session_id TEXT PRIMARY KEY, user_id TEXT NOT NULL)"
            )
            connection.execute("""
                CREATE TABLE IF NOT EXISTS workflow_requests (
                    session_id TEXT NOT NULL, request_id TEXT NOT NULL,
                    fingerprint TEXT NOT NULL, response_json TEXT,
                    abandoned INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (session_id, request_id)
                )
            """)

    def _connection(self):
        return sqlite_connection(self.path, busy_timeout_seconds=self.timeout)

    def claim(
        self, session_id: str, request_id: str, user_id: str, message: str, confirm_action: bool
    ) -> ChatResponse | None:
        fingerprint = hashlib.sha256(json.dumps([message, confirm_action]).encode()).hexdigest()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            owner = connection.execute(
                "SELECT user_id FROM workflow_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if owner and owner[0] != user_id:
                raise SessionOwnershipError("当前用户不能使用此会话")
            connection.execute(
                "INSERT OR IGNORE INTO workflow_sessions VALUES (?, ?)", (session_id, user_id)
            )
            row = connection.execute(
                "SELECT fingerprint, response_json, abandoned FROM workflow_requests "
                "WHERE session_id = ? AND request_id = ?",
                (session_id, request_id),
            ).fetchone()
            if row:
                if row[0] != fingerprint or row[2]:
                    raise RequestConflictError("request_id 已被使用或已取消；新问题请使用新的 ID")
                return ChatResponse.model_validate_json(row[1]) if row[1] else None
            pending = connection.execute(
                "SELECT request_id FROM workflow_requests WHERE session_id = ? "
                "AND response_json IS NULL AND abandoned = 0",
                (session_id,),
            ).fetchone()
            if pending:
                raise RequestConflictError(
                    f"会话尚有未完成请求，请使用原 request_id={pending[0]} 重试"
                )
            connection.execute(
                "INSERT INTO workflow_requests(session_id, request_id, fingerprint) "
                "VALUES (?, ?, ?)",
                (session_id, request_id, fingerprint),
            )
        return None

    def complete(self, response: ChatResponse) -> None:
        with self._connection() as connection:
            connection.execute(
                "UPDATE workflow_requests SET response_json = ? "
                "WHERE session_id = ? AND request_id = ?",
                (response.model_dump_json(), response.session_id, response.request_id),
            )

    def clear(self, session_id: str) -> None:
        # 保留已完成回执，清空聊天记录不应令旧业务请求重新执行。
        with self._connection() as connection:
            connection.execute(
                "UPDATE workflow_requests SET abandoned = 1 "
                "WHERE session_id = ? AND response_json IS NULL",
                (session_id,),
            )

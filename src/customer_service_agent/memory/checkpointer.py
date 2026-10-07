"""LangGraph SQLite checkpointer used for short-term conversation memory."""

from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from customer_service_agent.infrastructure.sqlite import open_sqlite_connection


class SQLiteAgentCheckpointer:
    """Own the SQLite connection required by LangGraph's SqliteSaver."""

    def __init__(self, db_path: str | Path, *, busy_timeout_seconds: float = 30.0) -> None:
        self._connection = open_sqlite_connection(
            db_path,
            busy_timeout_seconds=busy_timeout_seconds,
            check_same_thread=False,
        )
        self.saver = SqliteSaver(self._connection)
        self.saver.setup()

    def delete_thread(self, thread_id: str) -> None:
        self.saver.delete_thread(thread_id)

    def close(self) -> None:
        self._connection.close()

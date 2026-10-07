"""Shared SQLite connection management."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def open_sqlite_connection(
    db_path: str | Path,
    *,
    busy_timeout_seconds: float = 30.0,
    check_same_thread: bool = True,
) -> sqlite3.Connection:
    """Open a configured SQLite connection.

    WAL and busy_timeout make local concurrent reads/writes less fragile. They do
    not turn SQLite into a distributed production database.
    """

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(
        path,
        timeout=busy_timeout_seconds,
        check_same_thread=check_same_thread,
    )
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(f"PRAGMA busy_timeout = {int(busy_timeout_seconds * 1000)}")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


@contextmanager
def sqlite_connection(
    db_path: str | Path,
    *,
    busy_timeout_seconds: float = 30.0,
) -> Iterator[sqlite3.Connection]:
    """Yield a connection and always commit/rollback and close it."""

    connection = open_sqlite_connection(
        db_path,
        busy_timeout_seconds=busy_timeout_seconds,
    )
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

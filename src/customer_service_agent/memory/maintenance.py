"""停服后执行的SQLite维护；默认预览，--apply才备份并写入。

摘要只改变当前状态，旧checkpoint仍占空间。本模块显式裁剪旧快照，
同时处理对应writes及父引用。它依赖sqlite saver 3.x表结构，升级需回归测试。
"""

import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

from customer_service_agent.memory.consolidation import ProfileMemoryPolicy


def _backup(connection: sqlite3.Connection, path: Path) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    target = path.with_name(path.name + f".before-maintenance-{stamp}.bak")
    with closing(sqlite3.connect(target)) as destination:
        connection.backup(destination)
    return str(target)


def _stats(connection: sqlite3.Connection) -> dict:
    page_size = connection.execute("PRAGMA page_size").fetchone()[0]
    return {
        "database_bytes": connection.execute("PRAGMA page_count").fetchone()[0] * page_size,
        "reusable_bytes": connection.execute("PRAGMA freelist_count").fetchone()[0] * page_size,
    }


def maintain_database(
    path: Path,
    *,
    kind: str,
    policy: ProfileMemoryPolicy,
    keep_last: int = 20,
    retention_days: int = 0,
    apply: bool = False,
    vacuum: bool = False,
) -> dict:
    """只支持画像和checkpoint数据库；不删除整条会话或业务回执。"""
    if kind not in {"profiles", "checkpoints"} or keep_last < 2 or retention_days < 0:
        raise ValueError("无效的维护参数。")
    if vacuum and not apply:
        raise ValueError("VACUUM需要--apply，且必须先停止服务。")
    if not path.exists():
        return {"path": str(path), "status": "missing", "applied": False}
    # 预览不建表、不迁移；避免仅查看就改动用户数据。
    mode = "rw" if apply else "ro"
    connection = sqlite3.connect(path.resolve().as_uri() + f"?mode={mode}", uri=True, timeout=30)
    try:
        report = {"path": str(path), "applied": apply, "before": _stats(connection)}
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        required = {"customer_profiles"} if kind == "profiles" else {"checkpoints", "writes"}
        if not required.issubset(tables):
            return {**report, "applied": False, "status": "unsupported_schema"}
        if apply:
            report["backup"] = _backup(connection, path)
        connection.execute("BEGIN IMMEDIATE" if apply else "BEGIN")
        if kind == "profiles":
            report.update(_profiles(connection, policy, retention_days, apply))
        else:
            report.update(_checkpoints(connection, keep_last, apply))
        connection.commit()
        if vacuum:
            # DELETE释放页供复用；VACUUM才把空闲空间归还操作系统，可能需要额外磁盘。
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            connection.execute("VACUUM")
        report["after"] = _stats(connection)
        return report
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def _profiles(connection, policy, retention_days, apply):
    columns = {row[1] for row in connection.execute("PRAGMA table_info(customer_profiles)")}
    has_summary = "memory_summary" in columns
    if apply and not has_summary:
        connection.execute(
            "ALTER TABLE customer_profiles ADD COLUMN memory_summary TEXT NOT NULL DEFAULT ''"
        )
    cutoff = (datetime.now(UTC) - timedelta(days=retention_days)).replace(tzinfo=None)
    summary_column = "memory_summary" if has_summary else "''"
    expired = compacted = total = 0

    def batches():
        # keyset分页：不一次性载入全部用户，也不在未读完的游标上修改源表。
        last_user = None
        while True:
            where = "" if last_user is None else "WHERE user_id > ?"
            rows = connection.execute(
                "SELECT user_id, customer_name, phone, email, order_ids_json, issue_summary, "
                f"preferences_json, updated_at, {summary_column} FROM customer_profiles "
                f"{where} ORDER BY user_id LIMIT 500",
                () if last_user is None else (last_user,),
            ).fetchall()
            if not rows:
                return
            last_user = rows[-1][0]
            yield from rows

    for row in batches():
        total += 1
        updated_at = datetime.fromisoformat(str(row[7])).replace(tzinfo=None)
        if retention_days and updated_at < cutoff:
            expired += 1
            if apply:
                connection.execute("DELETE FROM customer_profiles WHERE user_id = ?", (row[0],))
            continue
        before = {
            "customer_name": row[1],
            "phone": row[2],
            "email": row[3],
            "order_ids": json.loads(row[4]),
            "issue_summary": row[5],
            "preferences": json.loads(row[6]),
            "memory_summary": row[8],
        }
        after = policy.consolidate(before)
        if before != after:
            compacted += 1
            if apply:
                connection.execute(
                    "UPDATE customer_profiles SET order_ids_json=?, issue_summary=?, "
                    "preferences_json=?, memory_summary=? WHERE user_id=?",
                    (
                        json.dumps(after["order_ids"], ensure_ascii=False),
                        after["issue_summary"],
                        json.dumps(after["preferences"], ensure_ascii=False),
                        after["memory_summary"],
                        row[0],
                    ),
                )
    return {
        "profiles_scanned": total,
        "profiles_to_compact": compacted,
        "profiles_to_expire": expired,
    }


def _checkpoints(connection, keep_last, apply):
    # 每个namespace分别保留最近N个。N是快照数，不是对话轮数。
    selection = """
        SELECT thread_id, checkpoint_ns, checkpoint_id FROM (
            SELECT thread_id, checkpoint_ns, checkpoint_id,
                   ROW_NUMBER() OVER (
                       PARTITION BY thread_id, checkpoint_ns ORDER BY checkpoint_id DESC
                   ) AS position
            FROM checkpoints
        ) WHERE position > ?
    """
    count = connection.execute("SELECT COUNT(*) FROM (" + selection + ")", (keep_last,)).fetchone()[
        0
    ]
    if apply and count:
        connection.execute("CREATE TEMP TABLE obsolete_checkpoints AS " + selection, (keep_last,))
        connection.execute(
            "CREATE UNIQUE INDEX obsolete_checkpoint_keys ON "
            "obsolete_checkpoints(thread_id, checkpoint_ns, checkpoint_id)"
        )
        for table in ("writes", "checkpoints"):
            # table只来自上面的固定白名单，不接受用户SQL标识符。
            connection.execute(f"""
                DELETE FROM {table} WHERE EXISTS (
                    SELECT 1 FROM obsolete_checkpoints AS old
                    WHERE old.thread_id={table}.thread_id
                      AND old.checkpoint_ns={table}.checkpoint_ns
                      AND old.checkpoint_id={table}.checkpoint_id
                )
            """)
        connection.execute("""
            UPDATE checkpoints SET parent_checkpoint_id=NULL
            WHERE parent_checkpoint_id IS NOT NULL AND NOT EXISTS (
                SELECT 1 FROM checkpoints AS parent
                WHERE parent.thread_id=checkpoints.thread_id
                  AND parent.checkpoint_ns=checkpoints.checkpoint_ns
                  AND parent.checkpoint_id=checkpoints.parent_checkpoint_id
            )
        """)
    return {"checkpoints_to_prune": count, "keep_last_per_namespace": keep_last}

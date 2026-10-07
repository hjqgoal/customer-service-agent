"""把素材导入专用 SQLite，原有 checkpoint、画像和 RAG 索引完全不动。

fixtures.sqlite3 只存工具可查的事实；cases.sqlite3 存人工验收标准。
每个数据库使用事务。同版本重复导入是幂等的，内容冲突时拒绝覆盖。
"""

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from customer_service_agent.evaluation.dataset import (
    DATASET_DIR,
    canonical_json,
    load_dataset,
)
from customer_service_agent.evaluation.render import casebook, review_csv
from customer_service_agent.infrastructure.sqlite import sqlite_connection


def _seed_database(path: Path, kind: str, rows: list[tuple[str, str, dict]], version: str):
    with sqlite_connection(path) as connection:
        # 静态夹具无需WAL；普通日志模式让只读工具不必在数据库旁创建-shm/-wal。
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute("BEGIN IMMEDIATE")
        tables = {
            r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if tables - {"dataset_meta", "dataset_records"}:
            raise ValueError(f"拒绝向不属于评估模块的数据库写入：{path}")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS dataset_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS dataset_records ("
            "kind TEXT NOT NULL, id TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(kind, id))"
        )
        existing = dict(connection.execute("SELECT key, value FROM dataset_meta"))
        expected = {"kind": kind, "version": version, "synthetic": "true"}
        if existing and existing != expected:
            raise ValueError("数据库用途或版本不匹配，请为新数据集选择新的输出目录")
        for key, value in expected.items():
            connection.execute("INSERT OR IGNORE INTO dataset_meta VALUES (?, ?)", (key, value))
        for record_kind, record_id, payload in rows:
            serialized = canonical_json(payload)
            old = connection.execute(
                "SELECT payload FROM dataset_records WHERE kind=? AND id=?",
                (record_kind, record_id),
            ).fetchone()
            if old and old[0] != serialized:
                raise ValueError(f"{record_id} 内容冲突，未覆盖已有数据")
            connection.execute(
                "INSERT OR IGNORE INTO dataset_records VALUES (?, ?, ?)",
                (record_kind, record_id, serialized),
            )


def seed_dataset(output: Path, dataset: Path = DATASET_DIR) -> dict:
    fixtures, cases = load_dataset(dataset)
    business_rows = [
        *(("order", o.id, o.model_dump()) for o in fixtures.orders),
        *(("logistics", r.tracking_number, r.model_dump()) for r in fixtures.logistics),
        *(("policy", p.id, p.model_dump()) for p in fixtures.policies),
    ]
    # 先校验所有待生成的文档，发现用户编辑过时不要覆盖。
    documents = {}
    for policy in fixtures.policies:
        text = (
            f"# {policy.title}\n\n{fixtures.disclaimer}\n\n"
            f"来源：虚构测试规则 {policy.id}；版本：{fixtures.version}；"
            f"固定时点：{fixtures.as_of}\n\n{policy.content}\n"
        )
        path = output / "knowledge_base" / f"{policy.id}.md"
        if path.exists() and path.read_text("utf-8") != text:
            raise ValueError(f"规则文档已被修改，拒绝覆盖：{path}")
        documents[path] = text
    _seed_database(output / "fixtures.sqlite3", "fixtures", business_rows, fixtures.version)
    _seed_database(
        output / "cases.sqlite3",
        "cases",
        [("case", c.id, c.model_dump()) for c in cases],
        fixtures.version,
    )
    for path, text in documents.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(text, encoding="utf-8")
    # 评分表可能已有人工填写；重跑 seed 绝不能清掉人工结果。
    for filename, content in (
        ("casebook.md", casebook(cases)),
        ("review.csv", review_csv(cases)),
        (".gitignore", "*.sqlite3\n*.sqlite3-*\nreview.csv\nruns/\n"),
    ):
        path = output / filename
        if not path.exists():
            path.write_text(content, encoding="utf-8")
    return {
        "synthetic": True,
        "orders": len(fixtures.orders),
        "logistics": len(fixtures.logistics),
        "policies": len(fixtures.policies),
        "cases": len(cases),
        "evaluated_cases": 0,
        "directory": str(output.resolve()),
    }


class FixtureStore:
    """只读、白名单查询接口；不向 Agent 提供任意 SQL 或访问答案库的工具。"""

    def __init__(self, path: Path):
        self.path = path.resolve()
        with self._connect() as connection:
            metadata = dict(connection.execute("SELECT key, value FROM dataset_meta"))
            if metadata.get("kind") != "fixtures":
                raise ValueError("业务工具不能连接案例答案库")

    def _connect(self):
        # mode=ro 避免路径拼错时静默创建一个空数据库；closing 确保连接释放。
        return closing(sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True))

    def get(self, kind: str, record_id: str) -> dict | None:
        if kind not in {"order", "logistics", "policy"}:
            raise ValueError("只能查询订单、物流或规则")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM dataset_records WHERE kind=? AND id=?",
                (kind, record_id.strip().upper()),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def records(self, kind: str) -> list[dict]:
        if kind not in {"order", "logistics", "policy"}:
            raise ValueError("未知记录类型")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM dataset_records WHERE kind=? ORDER BY id", (kind,)
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

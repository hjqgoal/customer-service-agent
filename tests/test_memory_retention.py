import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from customer_service_agent.agent import build_customer_service_agent
from customer_service_agent.memory import CustomerProfileRepository, CustomerProfileUpdate
from customer_service_agent.memory.consolidation import ProfileMemoryPolicy
from customer_service_agent.memory.maintenance import maintain_database
from tests.fakes import FakeToolCallingChatModel


def test_long_term_memory_replaces_summary_and_keeps_recent_facts(tmp_path):
    policy = ProfileMemoryPolicy(max_orders=2, max_preferences=2, summary_max_chars=100)
    repo = CustomerProfileRepository(tmp_path / "profiles.db", policy=policy)
    repo.upsert("u", CustomerProfileUpdate(order_ids=["A", "B"], preferences=["邮件", "短信"]))
    profile = repo.upsert(
        "u",
        CustomerProfileUpdate(
            order_ids=["A", "C"],
            preferences=["短信", "邮件", "电话"],
            issue_summary="请查退款",
        ),
    )
    assert profile.order_ids == ["A", "C"]
    assert profile.preferences == ["邮件", "电话"]
    assert "请查退款" in profile.memory_summary
    assert len(profile.memory_summary) <= 100
    assert "memory_summary" in profile.prompt_payload()
    with sqlite3.connect(repo.db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM customer_profiles").fetchone()[0] == 1
    repo.upsert("u", CustomerProfileUpdate(issue_summary="问题已解决"))
    assert "请查退款" not in repo.get("u").memory_summary


def test_profile_byte_budget_includes_chinese_and_never_changes_identity(tmp_path):
    policy = ProfileMemoryPolicy(max_bytes=2048)
    repo = CustomerProfileRepository(tmp_path / "profiles.db", policy=policy)
    p = repo.upsert(
        "u",
        CustomerProfileUpdate(
            customer_name="张三",
            phone="+86 123456",
            email="a@example.test",
            order_ids=[f"ORDER-{i}" for i in range(20)],
            preferences=[f"偏好{i}" + "中" * 190 for i in range(20)],
            issue_summary="退款申请" * 200,
        ),
    )
    payload = p.model_dump(exclude={"user_id", "created_at", "updated_at"})
    assert len(json.dumps(payload, ensure_ascii=False).encode()) <= 2048
    assert p.phone == "+86 123456" and p.customer_name == "张三"
    assert len(p.order_ids) <= 20 and len(p.preferences) <= 10


def test_concurrent_profile_writes_do_not_lose_distinct_orders(tmp_path):
    repo = CustomerProfileRepository(tmp_path / "profiles.db")
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(
            pool.map(
                lambda i: repo.upsert("u", CustomerProfileUpdate(order_ids=[str(i)])), range(15)
            )
        )
    assert set(repo.get("u").order_ids) == {str(i) for i in range(15)}


def test_maintenance_dry_run_backup_retention_and_vacuum(tmp_path):
    path = tmp_path / "profiles.db"
    repo = CustomerProfileRepository(path)
    repo.upsert("old", CustomerProfileUpdate(issue_summary="旧问题"))
    repo.upsert("new", CustomerProfileUpdate(issue_summary="新问题"))
    with sqlite3.connect(path) as c:
        c.execute("UPDATE customer_profiles SET updated_at='2000-01-01' WHERE user_id='old'")
    preview = maintain_database(
        path, kind="profiles", policy=ProfileMemoryPolicy(), retention_days=30
    )
    assert preview["profiles_to_expire"] == 1
    assert repo.get("old") is not None
    assert not list(tmp_path.glob("*.bak"))
    result = maintain_database(
        path,
        kind="profiles",
        policy=ProfileMemoryPolicy(),
        retention_days=30,
        apply=True,
        vacuum=True,
    )
    assert repo.get("old") is None and repo.get("new") is not None
    with sqlite3.connect(result["backup"]) as c:
        assert c.execute("SELECT COUNT(*) FROM customer_profiles").fetchone()[0] == 2


def test_retention_disabled_preserves_old_profiles_and_fact_timestamp(tmp_path):
    path = tmp_path / "profiles.db"
    repo = CustomerProfileRepository(path)
    repo.upsert("u", CustomerProfileUpdate(issue_summary="旧问题"))
    with sqlite3.connect(path) as c:
        c.execute("UPDATE customer_profiles SET updated_at='2000-01-01'")
    maintain_database(path, kind="profiles", policy=ProfileMemoryPolicy(), apply=True)
    assert repo.get("u").updated_at.year == 2000


def test_checkpoint_pruning_preserves_latest_state_and_restart(test_settings):
    s = test_settings.model_copy(update={"memory_summary_enabled": False})
    with build_customer_service_agent(s, llm=FakeToolCallingChatModel(), tools=[]) as agent:
        for i in range(5):
            agent.ask(f"第{i}轮", session_id="s")
        config = {"configurable": {"thread_id": "s"}}
        before = agent.graph.get_state(config).values["messages"]
    path = s.memory_db_file
    preview = maintain_database(path, kind="checkpoints", policy=ProfileMemoryPolicy(), keep_last=2)
    assert preview["checkpoints_to_prune"] > 0
    maintain_database(
        path, kind="checkpoints", policy=ProfileMemoryPolicy(), keep_last=2, apply=True
    )
    with sqlite3.connect(path) as c:
        assert c.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] == 2
        assert (
            c.execute(
                "SELECT COUNT(*) FROM writes w WHERE NOT EXISTS (SELECT 1 FROM checkpoints c "
                "WHERE c.thread_id=w.thread_id AND c.checkpoint_ns=w.checkpoint_ns "
                "AND c.checkpoint_id=w.checkpoint_id)"
            ).fetchone()[0]
            == 0
        )
    with build_customer_service_agent(s, llm=FakeToolCallingChatModel(), tools=[]) as agent:
        assert agent.graph.get_state(config).values["messages"] == before
        assert len(list(agent.graph.get_state_history(config))) == 2
        assert agent.ask("继续", session_id="s").answer


def test_checkpoint_pruning_preserves_namespaces_other_tables_and_pending_writes(tmp_path):
    from customer_service_agent.memory.checkpointer import SQLiteAgentCheckpointer

    path = tmp_path / "checkpoints.db"
    SQLiteAgentCheckpointer(path).close()
    with sqlite3.connect(path) as c:
        c.execute("CREATE TABLE request_receipts(id TEXT)")
        c.execute("INSERT INTO request_receipts VALUES ('do-not-delete')")
        for ns in ("", "child"):
            for i in range(4):
                c.execute(
                    "INSERT INTO checkpoints VALUES(?,?,?,?,?,?,?)",
                    ("s", ns, str(i), str(i - 1) if i else None, None, b"", b""),
                )
                c.execute(
                    "INSERT INTO writes VALUES(?,?,?,?,?,?,?,?)",
                    ("s", ns, str(i), "task", 0, "pending", None, b""),
                )
    maintain_database(
        path, kind="checkpoints", policy=ProfileMemoryPolicy(), keep_last=2, apply=True
    )
    with sqlite3.connect(path) as c:
        assert c.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] == 4
        assert c.execute("SELECT COUNT(*) FROM writes").fetchone()[0] == 4
        assert c.execute("SELECT COUNT(*) FROM request_receipts").fetchone()[0] == 1
        assert (
            c.execute(
                "SELECT parent_checkpoint_id FROM checkpoints WHERE checkpoint_id='2'"
            ).fetchone()[0]
            is None
        )


def test_maintenance_legacy_schema_needs_no_rebuild(tmp_path):
    path = tmp_path / "legacy.db"
    repo = CustomerProfileRepository(path)
    repo.upsert("u", CustomerProfileUpdate(issue_summary="待处理"))
    with sqlite3.connect(path) as c:
        c.execute("ALTER TABLE customer_profiles DROP COLUMN memory_summary")
    preview = maintain_database(path, kind="profiles", policy=ProfileMemoryPolicy())
    assert preview["profiles_to_compact"] == 1
    maintain_database(path, kind="profiles", policy=ProfileMemoryPolicy(), apply=True)
    assert CustomerProfileRepository(path).get("u").memory_summary


def test_vacuum_requires_apply(tmp_path):
    with pytest.raises(ValueError):
        maintain_database(
            tmp_path / "none.db", kind="profiles", policy=ProfileMemoryPolicy(), vacuum=True
        )


def test_unrelated_database_is_not_modified_or_backed_up(tmp_path):
    path = tmp_path / "business.db"
    with sqlite3.connect(path) as c:
        c.execute("CREATE TABLE receipts(id TEXT)")
    result = maintain_database(path, kind="checkpoints", policy=ProfileMemoryPolicy(), apply=True)
    assert result["status"] == "unsupported_schema" and not result["applied"]
    assert not list(tmp_path.glob("*.bak"))


def test_cli_maintenance_previews_without_constructing_model(test_settings, monkeypatch, capsys):
    import argparse

    from customer_service_agent import cli

    repo = CustomerProfileRepository(test_settings.profile_db_file)
    repo.upsert("u", CustomerProfileUpdate(issue_summary="待处理"))
    monkeypatch.setattr(cli, "get_settings", lambda: test_settings)
    monkeypatch.setattr(
        cli, "build_customer_service_agent", lambda *a, **k: pytest.fail("no model")
    )
    cli._run(argparse.Namespace(command="memory-maintain", apply=False, vacuum=False))
    reports = json.loads(capsys.readouterr().out)
    assert len(reports) == 3
    assert reports[0]["applied"] is False
    assert repo.get("u") is not None


def test_interrupted_graph_can_resume_after_offline_pruning(tmp_path):
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command, interrupt

    from customer_service_agent.memory.checkpointer import SQLiteAgentCheckpointer

    class State(TypedDict):
        confirmation: str

    def wait_for_confirmation(state):
        return {"confirmation": interrupt("确认吗？")}

    builder = StateGraph(State)
    builder.add_node("confirm", wait_for_confirmation)
    builder.add_edge(START, "confirm")
    builder.add_edge("confirm", END)
    path = tmp_path / "interrupt.db"
    config = {"configurable": {"thread_id": "pending"}}
    checkpointer = SQLiteAgentCheckpointer(path)
    graph = builder.compile(checkpointer=checkpointer.saver)
    for _ in range(4):
        graph.invoke({"confirmation": ""}, config)
        graph.invoke(Command(resume="yes"), config)
    assert graph.invoke({"confirmation": ""}, config).get("__interrupt__")
    checkpointer.close()
    maintain_database(
        path, kind="checkpoints", policy=ProfileMemoryPolicy(), keep_last=2, apply=True
    )
    restarted = SQLiteAgentCheckpointer(path)
    try:
        graph = builder.compile(checkpointer=restarted.saver)
        assert graph.invoke(Command(resume="confirmed"), config)["confirmation"] == "confirmed"
    finally:
        restarted.close()

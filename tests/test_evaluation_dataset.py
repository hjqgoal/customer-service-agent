"""验证案例和夹具的可用性，不给真实模型的回答打分。"""

import csv
import json
import shutil
import sqlite3
from collections import Counter

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from customer_service_agent.agent import build_customer_service_agent
from customer_service_agent.evaluation.cli import main
from customer_service_agent.evaluation.dataset import DATASET_DIR, load_dataset
from customer_service_agent.evaluation.store import FixtureStore, seed_dataset
from customer_service_agent.evaluation.tools import SQLiteFixtureOrders, build_fixture_tools
from customer_service_agent.rag.loader import load_documents
from tests.fakes import FakeToolCallingChatModel


@pytest.fixture
def seeded(tmp_path):
    seed_dataset(tmp_path)
    return tmp_path


def test_all_60_curated_cases_are_complete():
    fixtures, cases = load_dataset()
    assert Counter(c.category for c in cases) == {
        "order": 12,
        "logistics": 10,
        "policy": 12,
        "multi_turn": 10,
        "security": 8,
        "fault": 8,
    }
    assert len(fixtures.orders) == 12
    assert len(fixtures.logistics) == 10
    assert len(fixtures.policies) == 12
    assert all(c.turns and c.expected and c.evidence and c.forbidden for c in cases)
    assert len({t.message for c in cases for t in c.turns}) > 60


def test_days_and_amounts_match_boundary_cases():
    fixtures, _ = load_dataset()
    orders = {o.id: o for o in fixtures.orders}
    assert orders["EVAL-ORDER-1005"].signed_days_ago == 7
    assert orders["EVAL-ORDER-1006"].signed_days_ago == 8
    assert orders["EVAL-ORDER-1001"].amount_fen == 12900
    assert orders["EVAL-ORDER-1011"].tracking_number is None
    assert orders["EVAL-ORDER-1002"].paid is False


def test_seed_is_idempotent_and_answers_are_separate(seeded):
    assert seed_dataset(seeded)["evaluated_cases"] == 0
    with sqlite3.connect(seeded / "fixtures.sqlite3") as connection:
        counts = dict(
            connection.execute("SELECT kind, count(*) FROM dataset_records GROUP BY kind")
        )
    assert counts == {"order": 12, "logistics": 10, "policy": 12}
    with sqlite3.connect(seeded / "cases.sqlite3") as connection:
        assert (
            connection.execute("SELECT count(*) FROM dataset_records WHERE kind='case'").fetchone()[
                0
            ]
            == 60
        )
    assert "ORD-001" in (seeded / "casebook.md").read_text("utf-8")
    with (seeded / "review.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 60 and {r["result"] for r in rows} == {"not_run"}
    (seeded / "review.csv").write_text("用户已经填写的评分", encoding="utf-8")
    seed_dataset(seeded)
    assert (seeded / "review.csv").read_text("utf-8") == "用户已经填写的评分"


def test_rag_documents_contain_only_policies(seeded):
    docs = load_documents(seeded / "knowledge_base")
    assert len(docs) == 12
    assert all("虚构" in d.page_content for d in docs)
    assert all("ORD-001" not in d.page_content and "预期结果" not in d.page_content for d in docs)


def test_refuses_answers_database(seeded):
    with pytest.raises(ValueError, match="答案库"):
        FixtureStore(seeded / "cases.sqlite3")


def test_static_fixture_database_needs_no_wal_sidecar(seeded):
    store = FixtureStore(seeded / "fixtures.sqlite3")
    with store._connect() as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    assert not (seeded / "fixtures.sqlite3-wal").exists()
    assert not (seeded / "fixtures.sqlite3-shm").exists()


def test_seed_refuses_foreign_database(tmp_path):
    with sqlite3.connect(tmp_path / "fixtures.sqlite3") as connection:
        connection.execute("CREATE TABLE user_notes (text TEXT)")
        connection.execute("INSERT INTO user_notes VALUES ('keep')")
    with pytest.raises(ValueError, match="不属于"):
        seed_dataset(tmp_path)
    with sqlite3.connect(tmp_path / "fixtures.sqlite3") as connection:
        assert connection.execute("SELECT text FROM user_notes").fetchone()[0] == "keep"
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE name='dataset_records'"
            ).fetchone()
            is None
        )


def test_seed_conflict_rolls_back_and_preserves_user_content(seeded):
    with sqlite3.connect(seeded / "fixtures.sqlite3") as connection:
        connection.execute("DELETE FROM dataset_records WHERE id='EVAL-ORDER-1001'")
        connection.execute("UPDATE dataset_records SET payload='{}' WHERE id='EVAL-ORDER-1002'")
    with pytest.raises(ValueError, match="冲突"):
        seed_dataset(seeded)
    with sqlite3.connect(seeded / "fixtures.sqlite3") as connection:
        assert (
            connection.execute(
                "SELECT payload FROM dataset_records WHERE id='EVAL-ORDER-1001'"
            ).fetchone()
            is None
        )
        assert (
            connection.execute(
                "SELECT payload FROM dataset_records WHERE id='EVAL-ORDER-1002'"
            ).fetchone()[0]
            == "{}"
        )


def test_document_edits_are_not_overwritten(seeded):
    path = seeded / "knowledge_base" / "P-01.md"
    path.write_text("个人学习笔记", encoding="utf-8")
    with pytest.raises(ValueError, match="拒绝覆盖"):
        seed_dataset(seeded)
    assert path.read_text("utf-8") == "个人学习笔记"


@pytest.mark.parametrize("bad_reference", ["order:EVAL-ORDER-9999", "policy:P-99"])
def test_dataset_rejects_broken_evidence(tmp_path, bad_reference):
    shutil.copytree(DATASET_DIR, tmp_path / "dataset")
    path = tmp_path / "dataset" / "cases.jsonl"
    lines = path.read_text("utf-8").splitlines()
    case = json.loads(lines[0])
    case["evidence"] = [bad_reference]
    lines[0] = json.dumps(case)
    path.write_text("\n".join(lines), encoding="utf-8")
    with pytest.raises(ValueError, match="不存在的证据"):
        load_dataset(path.parent)


def test_fixture_tools_query_sqlite_and_enforce_ownership(seeded):
    tools = {t.name: t for t in build_fixture_tools(seeded / "fixtures.sqlite3")}
    order = json.loads(tools["lookup_order"].invoke({"order_id": " eval-order-1001 "}))
    assert order["source"] == "synthetic_evaluation"
    assert order["order"]["amount_fen"] == 12900
    assert not json.loads(tools["lookup_order"].invoke({"order_id": "EVAL-ORDER-1012"}))["found"]
    assert not json.loads(tools["lookup_logistics"].invoke({"tracking_number": "EVAL-TRACK-1012"}))[
        "found"
    ]
    other = {
        t.name: t for t in build_fixture_tools(seeded / "fixtures.sqlite3", user_id="EVAL-USER-B")
    }
    assert json.loads(other["lookup_order"].invoke({"order_id": "EVAL-ORDER-1012"}))["found"]
    assert (
        json.loads(tools["lookup_logistics"].invoke({"tracking_number": "EVAL-TRACK-1001"}))[
            "logistics"
        ]["status"]
        == "运输中"
    )
    assert (
        json.loads(tools["lookup_policy"].invoke({"topic": "P-01"}))["policies"][0]["id"] == "P-01"
    )
    assert not json.loads(tools["lookup_policy"].invoke({"topic": "终身保修"}))["found"]


def test_sql_input_is_not_executed(seeded):
    store = FixtureStore(seeded / "fixtures.sqlite3")
    assert store.get("order", "EVAL-ORDER-1001' OR '1'='1") is None
    assert len(store.records("order")) == 12
    with pytest.raises(ValueError):
        store.get("case", "ORD-001")
    with store._connect() as connection, pytest.raises(sqlite3.OperationalError):
        connection.execute("DELETE FROM dataset_records")


def test_existing_order_repository_contract(seeded):
    adapter = SQLiteFixtureOrders(FixtureStore(seeded / "fixtures.sqlite3"), "EVAL-USER-A")
    assert adapter.get("EVAL-ORDER-1001").tracking_number == "EVAL-TRACK-1001"
    assert adapter.get("EVAL-ORDER-1002").paid is False
    assert adapter.get("EVAL-ORDER-1012") is None


def test_cli_offline_paths(seeded, capsys):
    main(["--output", str(seeded), "query", "policy", "P-01"])
    assert json.loads(capsys.readouterr().out)["found"]
    main(["show", "MUL-007"])
    assert json.loads(capsys.readouterr().out)["turns"][1]["confirm_action"]
    main(["validate"])
    assert json.loads(capsys.readouterr().out)["agent_evaluation"] == "not_run"
    with pytest.raises(SystemExit):
        main(["show", "NO-SUCH-CASE"])


class FixtureQueryModel(FakeToolCallingChatModel):
    """只测试真实工具调用路径，最终回复不代表真实LLM质量。"""

    def _generate(self, messages, **kwargs):
        if isinstance(messages[-1], ToolMessage):
            message = AIMessage(content=messages[-1].content)
        else:
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup_order",
                        "args": {"order_id": "EVAL-ORDER-1001"},
                        "id": "fixture-order-call",
                        "type": "tool_call",
                    }
                ],
            )
        return ChatResult(generations=[ChatGeneration(message=message)])


def test_tools_work_with_existing_single_agent(test_settings, seeded):
    with build_customer_service_agent(
        test_settings,
        llm=FixtureQueryModel(),
        tools=build_fixture_tools(seeded / "fixtures.sqlite3"),
    ) as agent:
        result = agent.ask(
            "查询测试订单", session_id="fixture", user_id="EVAL-USER-A", include_trace=True
        )
    assert json.loads(result.answer)["order"]["id"] == "EVAL-ORDER-1001"
    assert result.intermediate_steps[0].tool == "lookup_order"

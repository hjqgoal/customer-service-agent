import json
import sqlite3

from customer_service_agent.agent import build_customer_service_agent
from customer_service_agent.memory import CustomerProfileRepository
from customer_service_agent.multi_agent.specialists import build_specialist_tools
from customer_service_agent.rag.service import RAGService
from tests.fakes import FakeEmbeddings
from tests.multi_fakes import CollaborationModel, task


def build(settings, model):
    settings = settings.model_copy(update={"agent_mode": "multi"})
    profiles = CustomerProfileRepository(settings.profile_db_file)
    tools = build_specialist_tools(settings, RAGService(settings, FakeEmbeddings()), profiles)
    return build_customer_service_agent(
        settings, llm=model, specialist_tools=tools, profile_repository=profiles
    )


def test_actual_order_and_logistics_tool_cycle(test_settings):
    model = CollaborationModel(
        plan={"tasks": [task("order"), task("tracking", "logistics", ["order"])]},
        scripted_calls={
            "order": {"name": "lookup_order", "args": {"order_id": "ORDER-1001"}},
            "tracking": {"name": "lookup_logistics", "args": {"tracking_number": "SF123456789"}},
        },
    )
    with build(test_settings, model) as agent:
        response = agent.ask("订单 ORDER-1001 到哪了", include_trace=True)
    assert response.status == "completed"
    assert [s.tool for s in response.intermediate_steps] == ["lookup_order", "lookup_logistics"]
    assert (
        json.loads(response.intermediate_steps[0].observation)["order"]["tracking_number"]
        == "SF123456789"
    )
    assert json.loads(response.intermediate_steps[1].observation)["logistics"]["status"] == "运输中"


def application_model(order_id="ORDER-1001", kind="action"):
    return CollaborationModel(
        plan={"tasks": [task("order"), task("submit", "after_sales", ["order"], kind)]},
        scripted_calls={
            "order": {"name": "lookup_order", "args": {"order_id": order_id}},
            "submit": {
                "name": "submit_after_sales_request",
                "args": {
                    "order_id": order_id,
                    "request_type": "return",
                    "reason": "商品不合适",
                },
            },
        },
    )


def test_confirmed_application_is_idempotent_across_replay_and_new_turn(test_settings):
    model = application_model()
    with build(test_settings, model) as agent:
        first = agent.ask(
            "申请退货 ORDER-1001", request_id="apply-1", confirm_action=True, include_trace=True
        )
        repeat = agent.ask("申请退货 ORDER-1001", request_id="apply-1", confirm_action=True)
        next_turn = agent.ask(
            "申请退货 ORDER-1001", request_id="apply-2", confirm_action=True, include_trace=True
        )
    a = json.loads(first.intermediate_steps[-1].observation)
    b = json.loads(next_turn.intermediate_steps[-1].observation)
    assert a["application_id"] == b["application_id"]
    assert a["source"] == "demo"
    assert not repeat.intermediate_steps  # 回执中有轨迹，普通 API 返回仍应隐藏它。
    with sqlite3.connect(test_settings.after_sales_db_file) as connection:
        assert connection.execute("SELECT count(*) FROM after_sales_requests").fetchone()[0] == 1


def test_unpaid_order_cannot_create_refund_application(test_settings):
    with build(test_settings, application_model("ORDER-1002")) as agent:
        response = agent.ask("退款 ORDER-1002", confirm_action=True, include_trace=True)
    assert json.loads(response.intermediate_steps[-1].observation)["status"] == "rejected"
    with sqlite3.connect(test_settings.after_sales_db_file) as connection:
        assert connection.execute("SELECT count(*) FROM after_sales_requests").fetchone()[0] == 0


def test_query_worker_cannot_access_write_tool_even_when_model_requests_it(test_settings):
    with build(test_settings, application_model(kind="query")) as agent:
        agent.ask("评估退货条件", confirm_action=True, include_trace=True)
    with sqlite3.connect(test_settings.after_sales_db_file) as connection:
        assert connection.execute("SELECT count(*) FROM after_sales_requests").fetchone()[0] == 0


def test_profile_updates_work_inside_specialist(test_settings):
    model = CollaborationModel(
        plan={"tasks": [task("profile", "order")]},
        scripted_calls={
            "profile": {
                "name": "save_customer_profile",
                "args": {
                    "customer_name": "张三",
                    "preferences": ["短信联系"],
                },
            },
        },
    )
    with build(test_settings, model) as agent:
        agent.ask("我叫张三，请短信联系", user_id="customer-1")
        profile = agent.get_customer_profile("customer-1")
    assert profile.customer_name == "张三"
    assert profile.preferences == ["短信联系"]


def test_ticket_operation_key_prevents_duplicate_writes(test_settings):
    from customer_service_agent.tools.ticket_tool import TicketRepository

    repository = TicketRepository(test_settings.ticket_db_file)
    first = repository.create("售后", "需要跟进", "", "normal", operation_key="same-operation")
    second = repository.create("售后", "需要跟进", "", "normal", operation_key="same-operation")
    assert first == second


def test_concurrent_after_sales_writes_create_one_application(test_settings):
    from concurrent.futures import ThreadPoolExecutor

    from customer_service_agent.tools.after_sales_tool import AfterSalesRepository

    repository = AfterSalesRepository(test_settings.after_sales_db_file)

    def create(index):
        return repository.create(
            key=f"different-request-{index}",
            user_id="u",
            order_id="ORDER-1001",
            request_type="return",
            reason="商品不合适",
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(create, range(4)))
    assert len(set(results)) == 1

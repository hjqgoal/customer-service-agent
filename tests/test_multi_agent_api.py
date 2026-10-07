from fastapi.testclient import TestClient

from customer_service_agent.agent import build_customer_service_agent
from customer_service_agent.api import app, get_agent
from tests.multi_fakes import CollaborationModel, task


def test_multi_api_reports_tasks_replays_requests_and_rejects_owner_change(test_settings):
    settings = test_settings.model_copy(update={"agent_mode": "multi"})
    model = CollaborationModel(
        plan={"tasks": [task("order"), task("policy", "after_sales")]}, failures={"policy": "fatal"}
    )
    with build_customer_service_agent(settings, llm=model, specialist_tools={}) as agent:
        app.dependency_overrides[get_agent] = lambda: agent
        try:
            with TestClient(app) as client:
                body = {
                    "message": "查订单及退款政策",
                    "session_id": "api",
                    "user_id": "user-a",
                    "request_id": "api-r1",
                }
                first = client.post("/chat", json=body)
                repeated = client.post("/chat", json=body)
                changed = client.post("/chat", json={**body, "message": "不同请求"})
                wrong_owner = client.post("/chat", json={**body, "user_id": "user-b"})
            assert first.status_code == 200
            assert first.json()["mode"] == "multi"
            assert first.json()["status"] == "partial"
            assert len(first.json()["tasks"]) == 2
            assert first.json()["intermediate_steps"] == []
            assert first.json() == repeated.json()
            assert changed.status_code == 409
            assert wrong_owner.status_code == 403
        finally:
            app.dependency_overrides.clear()


def test_api_provides_retry_id_when_caller_did_not_supply_one(test_settings):
    settings = test_settings.model_copy(update={"agent_mode": "multi"})
    with build_customer_service_agent(
        settings, llm=CollaborationModel(), specialist_tools={}
    ) as agent:
        app.dependency_overrides[get_agent] = lambda: agent
        try:
            with TestClient(app) as client:
                response = client.post("/chat", json={"message": "你好"})
            assert response.status_code == 200
            assert response.json()["request_id"] == response.headers["X-Request-ID"]
        finally:
            app.dependency_overrides.clear()

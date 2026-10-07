from fastapi.testclient import TestClient

from customer_service_agent.agent import build_customer_service_agent
from customer_service_agent.api import app, get_agent, get_rag_service
from customer_service_agent.config import get_settings
from customer_service_agent.memory import CustomerProfileUpdate
from tests.fakes import FakeToolCallingChatModel


class FailingRAGService:
    def search(self, query: str, *, top_k: int):
        raise ValueError("bad query")


def test_health_endpoint(test_settings):
    app.dependency_overrides[get_settings] = lambda: test_settings
    try:
        with TestClient(app) as client:
            response = client.get("/health", headers={"X-Request-ID": "test-request"})
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.headers["X-Request-ID"] == "test-request"
    finally:
        app.dependency_overrides.clear()


def test_rag_errors_use_stable_error_schema(test_settings):
    app.dependency_overrides[get_settings] = lambda: test_settings
    app.dependency_overrides[get_rag_service] = lambda: FailingRAGService()
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post("/rag/search", json={"query": "退款", "top_k": 4})
        assert response.status_code == 400
        assert response.json()["code"] == "invalid_request"
        assert response.json()["request_id"]
    finally:
        app.dependency_overrides.clear()


def test_chat_endpoint_passes_session_and_user_identity(test_settings):
    agent = build_customer_service_agent(
        test_settings,
        llm=FakeToolCallingChatModel(),
        tools=[],
    )
    app.dependency_overrides[get_settings] = lambda: test_settings
    app.dependency_overrides[get_agent] = lambda: agent
    try:
        with TestClient(app) as client:
            response = client.post(
                "/chat",
                json={
                    "message": "你好",
                    "session_id": "chat-001",
                    "user_id": "customer-001",
                },
            )

        assert response.status_code == 200
        assert response.json() == {
            "session_id": "chat-001",
            "user_id": "customer-001",
            "answer": "这是一个测试用客服回复。",
            "intermediate_steps": [],
            "mode": "single",
            "request_id": response.headers["X-Request-ID"],
            "status": "completed",
            "tasks": [],
        }
    finally:
        app.dependency_overrides.clear()
        agent.close()


def test_profile_admin_endpoints_read_and_delete_profile(test_settings):
    agent = build_customer_service_agent(
        test_settings,
        llm=FakeToolCallingChatModel(),
        tools=[],
    )
    agent.profile_repository.upsert(
        "customer-002",
        CustomerProfileUpdate(customer_name="王芳", preferences=["电话联系"]),
    )
    app.dependency_overrides[get_settings] = lambda: test_settings
    app.dependency_overrides[get_agent] = lambda: agent
    try:
        with TestClient(app) as client:
            read_response = client.get("/memory/profiles/customer-002")
            delete_response = client.delete("/memory/profiles/customer-002")
            missing_response = client.get("/memory/profiles/customer-002")

        assert read_response.status_code == 200
        assert read_response.json()["customer_name"] == "王芳"
        assert read_response.json()["preferences"] == ["电话联系"]
        assert delete_response.json()["deleted"] is True
        assert missing_response.status_code == 404
        assert missing_response.json()["code"] == "http_error"
    finally:
        app.dependency_overrides.clear()
        agent.close()

from customer_service_agent.tools.order_tool import lookup_order
from customer_service_agent.tools.policy_tool import lookup_policy
from customer_service_agent.tools.ticket_tool import TicketRepository


def test_lookup_order_known_order():
    result = lookup_order("ORDER-1001")

    assert "已发货" in result
    assert "SF123456789" in result


def test_lookup_policy_refund():
    result = lookup_policy("退款")

    assert "7 天" in result


def test_ticket_repository_create(tmp_path):
    repository = TicketRepository(str(tmp_path / "tickets.sqlite3"))

    ticket_id = repository.create("投诉", "用户要求人工回访", "user@example.com", "high")

    assert ticket_id.startswith("TICKET-")

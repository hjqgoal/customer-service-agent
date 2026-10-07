import pytest

from customer_service_agent.skills import (
    SkillRegistry,
    SkillValidationError,
    render_skill_catalog_for_prompt,
)
from customer_service_agent.tools.skill_tool import build_skill_loader_tool


def test_markdown_skills_are_loaded():
    skills = SkillRegistry().list()
    assert {skill.slug for skill in skills} >= {
        "order_query",
        "refund_after_sales",
        "knowledge_base_qa",
        "human_handoff",
    }


def test_prompt_catalog_does_not_eagerly_load_full_sop():
    catalog = render_skill_catalog_for_prompt()
    assert "order_query" in catalog
    assert "订单查询" in catalog
    assert "## 执行步骤" not in catalog


def test_skill_tool_loads_full_instructions():
    tool = build_skill_loader_tool()
    result = tool.invoke({"skill_id": "order_query"})
    assert "订单查询" in result
    assert "执行步骤" in result
    assert "lookup_order" in result


def test_unknown_skill_has_clear_error():
    with pytest.raises(SkillValidationError, match="未知 skill_id"):
        SkillRegistry().get("missing")

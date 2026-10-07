"""Backward-compatible exports for the old Python skill module.

现在 skill 的单一事实来源已经迁移到 `skills/*/SKILL.md`。
这个文件只保留兼容层，避免旧代码中的 import 立刻失效。
"""

from customer_service_agent.skills.loader import (
    MarkdownSkill as Skill,
)
from customer_service_agent.skills.loader import (
    load_skill_documents,
)
from customer_service_agent.skills.loader import (
    render_skill_catalog_for_prompt as render_skills_for_prompt,
)

CUSTOMER_SERVICE_SKILLS = tuple(load_skill_documents())

__all__ = ["CUSTOMER_SERVICE_SKILLS", "Skill", "render_skills_for_prompt"]

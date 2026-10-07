"""Tool for progressive loading of Markdown skill instructions."""

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from customer_service_agent.skills import SkillRegistry


class SkillInstructionsInput(BaseModel):
    skill_id: str = Field(
        ...,
        min_length=1,
        max_length=80,
        pattern=r"^[a-z0-9_\-]+$",
        description="技能目录中的 skill_id，例如 order_query",
    )


def build_skill_loader_tool(registry: SkillRegistry | None = None) -> StructuredTool:
    skill_registry = registry or SkillRegistry()

    def load_skill_instructions(skill_id: str) -> str:
        """读取一个客服技能的完整 SOP。"""

        return skill_registry.get(skill_id).to_prompt_text()

    return StructuredTool.from_function(
        func=load_skill_instructions,
        name="load_skill_instructions",
        description="按 skill_id 读取完整客服 SOP。执行技能前先调用此工具。",
        args_schema=SkillInstructionsInput,
    )

"""Load and validate Markdown skills."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SKILL_FILE_NAME = "SKILL.md"


class SkillValidationError(ValueError):
    """Raised when a SKILL.md file has invalid metadata."""


@dataclass(frozen=True)
class MarkdownSkill:
    slug: str
    path: Path
    name: str
    description: str
    tools: tuple[str, ...]
    content: str

    def to_prompt_text(self) -> str:
        tools = ", ".join(self.tools) if self.tools else "无"
        return (
            f"# {self.name}\n\n"
            f"skill_id: {self.slug}\n"
            f"description: {self.description}\n"
            f"tools: {tools}\n\n"
            f"{self.content.strip()}"
        )


def parse_front_matter(raw_text: str) -> tuple[dict[str, Any], str]:
    lines = raw_text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, raw_text.strip()

    closing_index = next(
        (index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---"),
        None,
    )
    if closing_index is None:
        raise SkillValidationError("SKILL.md front matter 缺少结束标记 '---'。")

    metadata = yaml.safe_load("\n".join(lines[1:closing_index])) or {}
    if not isinstance(metadata, dict):
        raise SkillValidationError("SKILL.md front matter 必须是 YAML 对象。")
    return metadata, "\n".join(lines[closing_index + 1 :]).strip()


def load_skill_document(skill_file: Path) -> MarkdownSkill:
    metadata, content = parse_front_matter(skill_file.read_text(encoding="utf-8"))
    slug = skill_file.parent.name
    name = str(metadata.get("name") or "").strip()
    description = str(metadata.get("description") or "").strip()
    raw_tools = metadata.get("tools") or []

    if not name or not description:
        raise SkillValidationError(f"{skill_file}: name 和 description 是必填字段。")
    if not content:
        raise SkillValidationError(f"{skill_file}: skill 正文不能为空。")
    if not isinstance(raw_tools, list) or not all(isinstance(tool, str) for tool in raw_tools):
        raise SkillValidationError(f"{skill_file}: tools 必须是字符串列表。")

    return MarkdownSkill(
        slug=slug,
        path=skill_file,
        name=name,
        description=description,
        tools=tuple(raw_tools),
        content=content,
    )


class SkillRegistry:
    """Immutable in-process registry built from `skills/*/SKILL.md`."""

    def __init__(self, skills_root: Path | None = None) -> None:
        root = skills_root or Path(__file__).parent
        loaded = [
            load_skill_document(path) for path in sorted(root.glob(f"*/{DEFAULT_SKILL_FILE_NAME}"))
        ]
        self._skills = {skill.slug: skill for skill in loaded}
        if len(self._skills) != len(loaded):
            raise SkillValidationError("检测到重复的 skill_id。")

    def list(self) -> tuple[MarkdownSkill, ...]:
        return tuple(self._skills.values())

    def get(self, skill_id: str) -> MarkdownSkill:
        try:
            return self._skills[skill_id]
        except KeyError as exc:
            available = ", ".join(sorted(self._skills))
            raise SkillValidationError(f"未知 skill_id: {skill_id}。可用技能：{available}") from exc

    def render_catalog(self) -> str:
        if not self._skills:
            return "当前没有可用技能。"
        return "\n".join(
            f"- {skill.slug}: {skill.name}；{skill.description}；可用工具："
            f"{', '.join(skill.tools) or '无'}"
            for skill in self.list()
        )


def load_skill_documents(skills_root: Path | None = None) -> list[MarkdownSkill]:
    return list(SkillRegistry(skills_root).list())


def render_skill_catalog_for_prompt(skills_root: Path | None = None) -> str:
    return SkillRegistry(skills_root).render_catalog()


def render_skills_for_prompt(skills_root: Path | None = None) -> str:
    """Backward-compatible full rendering used only by old integrations."""

    return "\n\n---\n\n".join(skill.to_prompt_text() for skill in SkillRegistry(skills_root).list())

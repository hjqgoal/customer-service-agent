"""验证人工编写的案例，避免缺标准答案、证据引用失效、物流和订单对不上。

这里验证的是数据集的完整性，不会把“案例格式正确”当成“Agent 回答正确”。
所有日期和规则均属于固定的虚构实验环境，不随电脑当天日期变化。
"""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DATASET_DIR = Path(__file__).resolve().parents[3] / "evals"
AUDIT_REFS = {
    "audit:trace",
    "audit:runtime",
    "audit:history",
    "audit:profile",
    "audit:after_sales",
    "audit:tickets",
    "audit:auth",
    "audit:rag",
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Turn(StrictModel):
    message: str = Field(min_length=1, max_length=8000)
    # 这是可信测试端传入的标志，不是从用户消息中的“已确认”三个字推断。
    confirm_action: bool = False


class Injection(StrictModel):
    target: str = Field(min_length=1)
    method: str = Field(min_length=1)
    verify: str = Field(min_length=1)


class EvaluationCase(StrictModel):
    id: str = Field(pattern=r"^(ORD|LOG|POL|MUL|SEC|FLT)-\d{3}$")
    title: str = Field(min_length=1)
    category: Literal["order", "logistics", "policy", "multi_turn", "security", "fault"]
    execution: Literal["chat", "write_harness", "security_harness", "fault_harness"] = "chat"
    preconditions: list[str] = Field(min_length=1)
    turns: list[Turn] = Field(min_length=1)
    expected: list[str] = Field(min_length=1)
    evidence: list[str] = Field(min_length=1)
    forbidden: list[str] = Field(min_length=1)
    injection: Injection | None = None

    @model_validator(mode="after")
    def check_execution(self):
        if (self.execution == "fault_harness") != (self.injection is not None):
            raise ValueError("故障案例必须描述注入方法，普通案例不能暗藏故障注入")
        if self.category == "multi_turn" and len(self.turns) < 2:
            raise ValueError("多轮案例至少需要两条用户消息")
        if self.category == "fault" and self.execution != "fault_harness":
            raise ValueError("故障案例必须通过测试夹具注入，不能只在聊天框里声称发生故障")
        return self


class FixtureOrder(StrictModel):
    id: str = Field(pattern=r"^EVAL-ORDER-\d{4}$")
    customer_id: str = Field(pattern=r"^EVAL-USER-[AB]$")
    product: str
    amount_fen: int = Field(gt=0)
    status: str
    logistics: str
    eta: str
    tracking_number: str | None = None
    paid: bool = True
    signed_days_ago: int | None = Field(default=None, ge=0)
    category: Literal["ordinary", "custom"] = "ordinary"
    condition: str


class FixtureLogistics(StrictModel):
    tracking_number: str = Field(pattern=r"^EVAL-TRACK-\d{4}$")
    status: str
    events: list[str] = Field(min_length=1)
    eta: str


class FixturePolicy(StrictModel):
    id: str = Field(pattern=r"^P-\d{2}$")
    title: str
    keywords: list[str] = Field(min_length=1)
    content: str = Field(min_length=1)


class Fixtures(StrictModel):
    version: Literal["2026-09-23.v1"]
    synthetic: Literal[True]
    as_of: Literal["2026-09-23T12:00:00+08:00"]
    disclaimer: str
    orders: list[FixtureOrder]
    logistics: list[FixtureLogistics]
    policies: list[FixturePolicy]


def load_dataset(directory: Path = DATASET_DIR) -> tuple[Fixtures, list[EvaluationCase]]:
    fixtures = Fixtures.model_validate_json((directory / "fixtures.json").read_text("utf-8"))
    cases = []
    for number, line in enumerate((directory / "cases.jsonl").read_text("utf-8").splitlines(), 1):
        if line.strip():
            try:
                cases.append(EvaluationCase.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"cases.jsonl 第 {number} 行不合法：{exc}") from exc
    groups = [
        [o.id for o in fixtures.orders],
        [p.id for p in fixtures.policies],
        [r.tracking_number for r in fixtures.logistics],
        [c.id for c in cases],
    ]
    if not cases or any(len(ids) != len(set(ids)) for ids in groups):
        raise ValueError("案例不能为空，各组 ID 必须唯一")
    tracking = set(groups[2])
    used = [o.tracking_number for o in fixtures.orders if o.tracking_number]
    if set(used) != tracking or len(used) != len(set(used)):
        raise ValueError("每条物流必须关联且只关联一个虚构订单")
    known = AUDIT_REFS | {
        *(f"order:{o.id}" for o in fixtures.orders),
        *(f"logistics:{r.tracking_number}" for r in fixtures.logistics),
        *(f"policy:{p.id}" for p in fixtures.policies),
    }
    for case in cases:
        if missing := set(case.evidence) - known:
            raise ValueError(f"{case.id} 引用了不存在的证据：{sorted(missing)}")
    return fixtures, cases


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

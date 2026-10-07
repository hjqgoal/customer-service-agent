"""从同一份 JSONL 生成易读案例册和人工评分表，不维护两份标准答案。"""

import csv
import io

from customer_service_agent.evaluation.dataset import EvaluationCase


def casebook(cases: list[EvaluationCase]) -> str:
    lines = [
        "# 电商客服人工评估案例册",
        "",
        "全部是虚构测试案例。本文含验收答案，严禁导入 RAG。",
        "这是一份待执行案例集，不是已经通过测试的报告。",
        "",
        "每个案例使用独立会话；多轮案例内部保留同一会话，等待上轮回答后再发送下一条。",
        "fault_harness / security_harness / write_harness 需要相应测试环境；条件未满足记 blocked。",
        "不能用只读工具测试写操作然后把零写入算作确认保护通过。",
        "",
    ]
    for case in cases:
        lines.extend(
            [
                f"## {case.id} {case.title}",
                "",
                f"类别：{case.category}；执行方式：{case.execution}",
                "",
            ]
        )
        sections = [
            ("前置条件", case.preconditions),
            (
                "用户消息（依次发送）",
                [
                    f"第{i}轮 [confirm_action={t.confirm_action}]：{t.message}"
                    for i, t in enumerate(case.turns, 1)
                ],
            ),
            ("预期结果（全部满足才通过）", case.expected),
            ("必要证据", case.evidence),
            ("禁止行为（任一发生即失败）", case.forbidden),
        ]
        if case.injection:
            sections.append(
                (
                    "故障注入",
                    [
                        f"目标：{case.injection.target}",
                        f"方法：{case.injection.method}",
                        f"验证：{case.injection.verify}",
                    ],
                )
            )
        for heading, items in sections:
            lines.extend([f"### {heading}", "", *(f"- {item}" for item in items), ""])
    return "\n".join(lines)


def review_csv(cases: list[EvaluationCase]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(["case_id", "title", "execution", "result", "evidence_path", "notes"])
    writer.writerows((c.id, c.title, c.execution, "not_run", "", "") for c in cases)
    return buffer.getvalue()

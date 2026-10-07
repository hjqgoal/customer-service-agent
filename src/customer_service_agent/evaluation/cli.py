"""纯离线入口：导入、浏览案例、直接查询 SQLite；不读取 .env 或调用模型。"""

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

from customer_service_agent.evaluation.dataset import DATASET_DIR, load_dataset
from customer_service_agent.evaluation.store import seed_dataset
from customer_service_agent.evaluation.tools import build_fixture_tools


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_DIR)
    parser.add_argument("--output", type=Path, default=DATASET_DIR.parent / "data" / "evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate", help="校验案例完整性，不运行 Agent")
    sub.add_parser("seed", help="事务导入专用 SQLite，保留原业务数据库")
    sub.add_parser("list", help="列出案例编号、标题和执行类型")
    show = sub.add_parser("show", help="查看一例的输入和人工验收标准")
    show.add_argument("case_id")
    query = sub.add_parser("query", help="不调用模型，直接检验只读业务工具")
    query.add_argument("kind", choices=["order", "logistics", "policy"])
    query.add_argument("value")
    query.add_argument("--user", default="EVAL-USER-A", choices=["EVAL-USER-A", "EVAL-USER-B"])
    args = parser.parse_args(argv)
    try:
        fixtures, cases = load_dataset(args.dataset)
        if args.command == "seed":
            result = seed_dataset(args.output, args.dataset)
        elif args.command == "validate":
            result = {
                "version": fixtures.version,
                "cases": len(cases),
                "categories": dict(Counter(c.category for c in cases)),
                "execution": dict(Counter(c.execution for c in cases)),
                "agent_evaluation": "not_run",
            }
        elif args.command == "list":
            for case in cases:
                print(f"{case.id}\t{case.execution}\t{case.title}")
            return
        elif args.command == "show":
            case = next((c for c in cases if c.id == args.case_id.upper()), None)
            if case is None:
                parser.error(f"未知案例：{args.case_id}")
            result = case.model_dump()
        else:
            tools = {
                t.name: t
                for t in build_fixture_tools(args.output / "fixtures.sqlite3", user_id=args.user)
            }
            name, argument = {
                "order": ("lookup_order", "order_id"),
                "logistics": ("lookup_logistics", "tracking_number"),
                "policy": ("lookup_policy", "topic"),
            }[args.kind]
            result = json.loads(tools[name].invoke({argument: args.value}))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError, sqlite3.Error) as exc:
        parser.exit(1, f"评估素材操作失败：{exc}\n")


if __name__ == "__main__":
    main()

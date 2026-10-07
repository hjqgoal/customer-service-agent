# 电商客服评估起步集

本次素材完全由人工方式编写，未抓取真实客户数据，不引用真实商家承诺。
所有订单、运单、商品和政策均为虚构，固定测试时点为 `2026-09-23 12:00 +08:00`。
金额以分保存，例如 `12900` 表示 129 元。规则不是法律意见。

## 从哪里看

- `cases.jsonl`：60 个完整案例，一行一例，是唯一维护的案例来源。
- `fixtures.json`：12 笔订单、10 条物流、12 条政策，供工具查询。
- 运行导入后看 `data/evaluation/casebook.md`：适合直接阅读的中文案例册。
- `data/evaluation/review.csv`：人工填写通过、失败和证据位置，初始全部 `not_run`。
- `src/customer_service_agent/evaluation/`：校验、SQLite 导入、只读工具和命令入口。
- `tests/test_evaluation_dataset.py`：数据与工具的离线测试，不调用真实模型。

| 分类 | 数量 | 重点 |
| --- | ---: | --- |
| ORD 订单 | 12 | 状态、金额、7/8天边界、普通/定制商品 |
| LOG 物流 | 10 | 正确运单、天气延误、签收争议、包裹关联 |
| POL 规则 | 12 | 退货、运费、退款时效、换货、发票、知识缺失 |
| MUL 多轮 | 10 | 澄清、更正、指代、确认、重复申请、记忆压缩 |
| SEC 安全 | 8 | 授权隔离、提示注入、虚假执行、最少必要信息 |
| FLT 故障 | 8 | 超时、证据源故障、坏计划、崩溃重放、SQLite锁 |

普通单轮、多轮、边界和对抗题都属于开发起步集，不是隐藏验收集。
相近表达和相关案例应按语义分组再划分训练/开发/验收，不能随机逐行切分后宣称泛化能力。
当前只准备数据、提供工具并验证程序；没有真实模型的任务成功率。

## 离线使用

在项目根目录、已安装项目依赖的 Python 环境运行。未安装可使用 `PYTHONPATH=src` 前缀。
这些命令不读取 `.env`，不连接模型、快递或支付服务：

```bash
python -m customer_service_agent.evaluation.cli validate
python -m customer_service_agent.evaluation.cli seed
python -m customer_service_agent.evaluation.cli list
python -m customer_service_agent.evaluation.cli show ORD-005
python -m customer_service_agent.evaluation.cli query order EVAL-ORDER-1001
python -m customer_service_agent.evaluation.cli query logistics EVAL-TRACK-1001
python -m customer_service_agent.evaluation.cli query policy 退货
python -m pytest tests/test_evaluation_dataset.py -q
```

查询特定规则可使用 `query policy P-01`。用户B的数据需用可信测试端参数：
`query order EVAL-ORDER-1012 --user EVAL-USER-B`。
CLI 中的 `--user` 是本地测试开关，不是登录认证，不应直接对外开放。

## SQLite 放什么

导入生成两个独立数据库，避免混进原来的聊天历史和 RAG：

- `data/evaluation/fixtures.sqlite3`：`dataset_records` 中的 order/logistics/policy；Agent 只读工具只查这个库。
- `data/evaluation/cases.sqlite3`：`dataset_records` 中的 case，包含期望答案、禁止行为；只给评估者使用。
- 两个库都有 `dataset_meta` 记录版本、用途和虚构标记，`payload` 是 JSON 文本。

同版本重复导入不增加重复记录，原内容不同则报冲突，不覆盖用户修改。
每个数据库分别使用事务；两个文件不是跨数据库分布式事务，部分失败后可修正问题再幂等重跑。
工具没有任意 SQL 权限，并拒绝连接 `cases.sqlite3`。
旧 checkpoint、画像、工单和 `data/rag_chunks.sqlite3` 不会被改动。

## 怎样接到现有 Agent

普通 `customer_service_agent.cli ask` 仍使用原来的演示工具；导入新库不会悄悄切换数据源。
新增的 `build_fixture_tools()` 返回同名的三个只读 LangChain 工具：
`lookup_order`、`lookup_logistics`、`lookup_policy`。它们可以通过已有 `tools=` 参数注入：

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from customer_service_agent.agent import build_customer_service_agent
from customer_service_agent.config import PROJECT_ROOT, Settings
from customer_service_agent.evaluation.tools import build_fixture_tools

fixtures = PROJECT_ROOT / "data/evaluation/fixtures.sqlite3"
tools = build_fixture_tools(fixtures, user_id="EVAL-USER-A")

# 下面才会调用你配置的真实聊天模型并可能计费；本次没有代你执行。
with TemporaryDirectory(prefix="agent-eval-") as directory:
    root = Path(directory)
    settings = Settings().model_copy(update={
        "agent_mode": "single", "app_env": "test", "memory_summary_enabled": False,
        "memory_db_path": root / "memory.sqlite3",
        "multi_agent_db_path": root / "multi.sqlite3",
        "profile_db_path": root / "profiles.sqlite3",
        "after_sales_db_path": root / "after_sales.sqlite3",
        "ticket_db_path": root / "tickets.sqlite3",
        "rag_db_path": root / "rag.sqlite3", "knowledge_base_dir": root / "kb",
    })
    with build_customer_service_agent(settings, tools=tools) as agent:
        result = agent.ask("查 EVAL-ORDER-1001 的物流", session_id="LOG-001",
                           user_id="EVAL-USER-A", include_trace=True)
        print(result.model_dump_json(indent=2))
```

不要同时把默认演示订单工具和同名夹具工具加入 Agent，否则数据源混杂。
此例只读，不提供售后写工具；不得用它测试确认保护后宣称写操作安全。
`SQLiteFixtureOrders(FixtureStore(path), user_id)` 实现现有 `OrderRepository.get` 契约，
需要测试模拟写入时，可将它传给 `build_after_sales_tools(sandbox_settings, orders=...)`。
原有写工具的业务校验并不等于覆盖全部新政策；不能为了通过评测修改期望答案。

多Agent可将三种工具按专业角色分组，注入 `MultiAgentCustomerService` 的 `tools_by_agent`，
并按需配备沙箱售后、RAG、画像和skill工具。当前仓库 `agent.py` 仍是单Agent工厂，
未接回多Agent分支；不要直接在该函数传 `specialist_tools=`，会产生参数错误。
本次只新增评估模块，没有替换你正在学习的主代码。

## RAG 怎么测

导入同时生成 `data/evaluation/knowledge_base/P-01.md` 到 `P-12.md`，只包含业务政策。
尚未生成任何向量：真实 embedding 需要你配置的模型，不能把随便生成的数字塞进原索引。
创建使用独立 `rag_db_path` 的 `RAGService`，通过原有
`ingest_directory(PROJECT_ROOT / "data/evaluation/knowledge_base")` 建索引，再把
`build_rag_search_tool(rag_service)` 加入测试工具集合。该步骤会调用 embedding 服务并可能计费。

不要把 `cases.jsonl`、`casebook.md`、`review.csv`、`cases.sqlite3` 或整个 `evals/` 导入RAG。
做检索评估时可暂时不提供 `lookup_policy`，否则模型可能从规则工具取得答案，掩盖检索问题。
记录命中文档ID、上下文、回答，再对照每例 `policy:P-xx` 证据检查召回和忠实度。
普通规则工具查询正确，不等于已经验证了向量检索质量。

## 怎样执行和评分

1. 每例使用新的会话和独立运行目录，多轮案例内部连续发送、等待每轮回复；不要批量拼成一句话。
2. 记录模型、提示词、数据版本和运行时配置。同例重复运行应使用新请求ID，幂等重放题除外。
3. 按 `preconditions` 建立环境。`chat` 可做人工对话；其他执行类型必须配相应测试夹具，条件不满足记 `blocked`。
4. 保存回复、工具参数/结果、必要的数据库前后快照。`expected` 全满足且 `forbidden` 无任何触发才记 `pass`。
5. `review.csv` 填 `pass`、`fail`、`blocked`、`not_run`，记录证据路径和失败原因。重跑seed不覆盖已填写评分表。
6. 报告执行覆盖率 `(pass+fail)/60` 和已执行通过率 `pass/(pass+fail)`；分母为0时写未评估，blocked不能当pass。

`audit:trace` 是工具调用轨迹；`audit:history/profile` 是测试会话或画像；
`audit:after_sales/tickets` 是写入前后记录；`audit:auth` 是可信身份及隔离检查；
`audit:rag` 是命中文档和检索结果；`audit:runtime` 是异常、重试和调度事件。
这些是评估者收集的证据，不是给Agent检索的文档。部分检验需要人工判断，不能只做关键词匹配。

故障题写了具体 target/method/verify，但本次没有实现或执行全部故障注入器。
崩溃和SQLite锁测试只能在临时目录及独立测试子进程进行，绝不能关闭当前开发服务或锁住真实数据库。
LangSmith可以以后接入管理这些案例和轨迹；起步阶段不依赖平台账户。

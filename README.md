# customer-service-agent：规范化智能客服 Agent 参考工程

这是一个真实 Agent 工程边界的 LangGraph 项目。它可以在本地完整运行，也明确标注了
SQLite 与互联网生产部署之间的差距。

默认运行 **Supervisor + 订单 Agent + 物流 Agent + 售后 Agent** 的协作模式，支持依赖调度、
并行执行、失败降级和 SQLite 恢复。设置 `AGENT_MODE=single` 可以运行原版单 Agent。
已读过原版的同学，请先看 [新版指南](docs/multi-agent-learning.md)。

长期画像提炼、SQLite空间维护和独立排序模型的参数见[记忆与Rerank](docs/memory-and-rerank.md)。

## 核心模块

- LangChain 1.x `create_agent`：工具调用 Agent，底层运行在 LangGraph。
- 多 Agent：外层 StateGraph + Send 调度，专业 Agent 以最小工具权限运行。
- LangGraph SQLite checkpointer：按 `thread_id` 持久化短期会话状态。
- 三层记忆：保留最近 N 轮、旧历史累计摘要、按 `user_id` 保存结构化客户画像。
- Markdown Skills：单 Agent 只加载技能目录、按需读取 SOP；专业 Agent 直接带上本角色 SOP。
- Tools：订单、物流、政策查询，模拟售后申请与工单，知识库检索和 Skill 加载。
- RAG：原子索引替换、哈希去重、BM25 + 向量混合检索、Rerank（DashScope / Jina 兼容协议）、
  浏览器 MCP 网页降级。
- 评测：60 例电商客服案例集、只读夹具库、真实模型评测脚本与记录。
- FastAPI：聊天、健康检查、就绪检查、RAG 管理和稳定错误响应。
- CLI / MCP：本地调试和外部工具协议入口。
- 模拟售后：调用方确认、幂等申请和请求回执；不执行真实退款。
- 测试与质量：pytest、Ruff、pre-commit、GitHub Actions。

## 安装

需要 Python 3.11 及以上，推荐使用独立环境：

```bash
git clone https://github.com/hjqgoal/customer-service-agent.git
cd customer-service-agent
python -m venv .venv
source .venv/bin/activate  # 或激活你自己的 conda 环境
python -m pip install -e ".[dev]"
python -m pip check
```

`pip check` 应输出 `No broken requirements found.`。项目已经迁移到 LangChain 1.x，不应再安装
`langchain 0.3.x` 与 `langchain-core 1.x` 的混合版本。

## 配置

已有 `.env` 的用户继续使用原配置，不要覆盖密钥。新增配置有默认值，详见 `.env.example`。

```bash
cp .env.example .env
```

聊天和 embedding 使用独立配置：

```env
CHAT_API_KEY="..."
CHAT_BASE_URL="..."
CHAT_MODEL_NAME="..."

EMBEDDING_API_KEY="..."
EMBEDDING_BASE_URL="..."
EMBEDDING_MODEL_NAME="..."
```

对于 DashScope 等只接受 `str/list[str]` 的 OpenAI-compatible embedding 接口，保持：

```env
EMBEDDING_CHECK_CTX_LENGTH="false"
```

记忆相关配置：

```env
MEMORY_RECENT_TURNS="6"
MEMORY_SUMMARY_ENABLED="true"
MEMORY_SUMMARY_MAX_CHARS="4000"
PROFILE_DB_PATH="data/customer_profiles.sqlite3"
```

生产环境必须设置 `ADMIN_API_KEY`，并通过 `X-Admin-Key` 访问 RAG 入库和清空记忆接口。

## 开发检查

```bash
make check
```

也可以分别运行：

```bash
python -m ruff check src tests
python -m pytest
```

启用提交前检查：

```bash
pre-commit install
```

测试全部使用 fake model / fake embeddings，不消耗真实 API。

## RAG

已准备12条规则、20笔订单、16份物流的模拟业务知识，并完成真实向量入库及跨会话记忆验证。
直接测试请看 [业务知识库与长期记忆实测](docs/business-lab.md)。这套实验使用独立的
`python -m examples.business_lab` 入口，避免与旧固定演示订单工具混淆。

先构建索引：

```bash
python -m customer_service_agent.cli rag-ingest --path data/knowledge_base
```

默认行为是：先成功生成全部向量，再在一个 SQLite 事务中替换旧索引。embedding 失败不会清空
原索引。增量入库会按内容哈希去重：

```bash
python -m customer_service_agent.cli rag-ingest --path data/knowledge_base --append
```

搜索：

```bash
python -m customer_service_agent.cli rag-search "退款规则"
```

检索现在并行执行中文关键词BM25与向量检索，每路默认Top3；工具把两组候选交给聊天模型
判断相关性、冲突和引用。本地无结果、检索异常，或显式使用`--force-web`时，通过浏览器MCP
查询公开搜索页。浏览器配置和学习顺序见[混合RAG说明](docs/hybrid-rag.md)。

SQLite向量检索仍是O(n)教学实现，关键词由FTS5倒排索引排序。首次打开旧库会补建关键词
索引，保留原chunk及向量；大知识库请替换为pgvector、Qdrant或Milvus。

## Agent CLI

```bash
python -m customer_service_agent.cli ask "帮我查询订单 ORDER-1001" --session demo --user user-001
python -m customer_service_agent.cli ask "退款规则是什么" --session demo --trace
python -m customer_service_agent.cli chat --session demo --user user-001
```

`--trace` 只用于本地学习。FastAPI `/chat` 默认不返回工具输入和观察结果，避免泄露内部信息。

## 三层记忆

这三个概念不要混在一起：

| 层次 | 主键 | 保存内容 | 代码位置 |
| --- | --- | --- | --- |
| 最近对话 | `session_id/thread_id` | 最近 N 个“用户 + Agent”轮次及工具消息 | `memory/checkpointer.py` |
| 累计摘要 | `session_id/thread_id` | 被移出窗口的旧会话摘要 | `memory/compaction.py` |
| 客户画像 | `user_id` | 姓名、电话、邮箱、订单号、问题摘要、服务偏好 | `memory/profile.py` |

每次收到新消息前，`ConversationCompactor` 读取当前 LangGraph 状态。若加入本轮后会超过
`MEMORY_RECENT_TURNS`，它会把超长旧历史完整分段，分别生成摘要，再分层合并为包含
`confirmed_facts`、`current_request`、`tool_results`、`promises` 和 `pending_actions` 的
结构化 JSON，最后用一条带 `[conversation_summary]` 标记的 `SystemMessage` 替换旧消息。
已有摘要也会参与下一次压缩，因此摘要是累计的。模型服务或 JSON 校验失败时，对应片段会
退化为有长度上限的结构化确定性摘要，聊天仍可继续。摘要被明确标记为“不可信数据”，其中
出现的命令或角色设定不能作为指令执行。

客户画像不依赖摘要。Agent 通过 `save_customer_profile` tool 只写入白名单字段；
`ToolRuntime` 从可信运行时注入 `user_id`，不会把它暴露给模型填写。下一次模型调用时，
动态 prompt middleware 会按 `user_id` 读取最新画像。相同用户换一个 `session_id` 仍可读取画像，
清空某段会话也不会误删长期画像。

不要把密码、验证码、支付卡号或完整证件号放进画像。真实生产系统还需要用户鉴权、字段加密、
数据保留期限、审计与删除流程。

## FastAPI

```bash
python -m uvicorn customer_service_agent.api:app --reload --host 127.0.0.1 --port 8000
```

打开 `http://127.0.0.1:8000/docs`。主要接口：

- `GET /health`：进程存活检查。
- `GET /ready`：配置与本地数据目录就绪检查；未就绪返回 503。
- `POST /chat`：对话接口。
- `POST /rag/search`：知识库搜索。
- `POST /rag/ingest`：管理接口，只读取配置的知识库目录。
- `POST /memory/{session_id}/clear`：管理接口。
- `GET /memory/profiles/{user_id}`：读取结构化客户画像，管理接口。
- `DELETE /memory/profiles/{user_id}`：删除结构化客户画像，管理接口。

聊天请求示例；`user_id` 省略时会回退为 `session_id`：

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"我叫张三，想查询 ORDER-1001","session_id":"chat-001","user_id":"user-001"}'
```

管理接口示例：

```bash
curl -X POST http://127.0.0.1:8000/rag/ingest \
  -H "Content-Type: application/json" \
  -H "X-Admin-Key: $ADMIN_API_KEY" \
  -d '{"replace": true}'
```

## Skill 规范

每个 Skill 使用独立目录：

```text
skills/
  order_query/SKILL.md
  logistics_query/SKILL.md
  refund_after_sales/SKILL.md
  knowledge_base_qa/SKILL.md
  human_handoff/SKILL.md
```

`SKILL.md` front matter 由 `yaml.safe_load` 解析并校验。单 Agent 的初始 prompt 只包含名称、描述和
工具目录，需要执行技能时调用 `load_skill_instructions` 加载完整 SOP。多 Agent 中订单、物流、售后
Agent 的领域固定，构建时直接把本角色 SOP 写进系统提示词，省掉每个任务一次加载往返
（见 `multi_agent/specialists.py`）。

## 评测

`evals/` 是 60 例电商客服评测集，覆盖订单、物流、政策、多轮、安全、故障六类，配套只读夹具库。
离线校验不调用模型：

```bash
python -m customer_service_agent.evaluation.cli validate
```

`python -m customer_service_agent.evaluation.runner` 用真实模型逐例执行并判分，会产生模型费用。
第一次真实评测记录见 [docs/evaluation-2026-11-04.md](docs/evaluation-2026-11-04.md)：已执行 31 例、
通过 30 例；政策、多轮、安全三类执行得还不够，不能据此评价这三类能力。

## 文档

| 文档 | 内容 |
| --- | --- |
| [docs/multi-agent-learning.md](docs/multi-agent-learning.md) | 从单 Agent 读到协作系统：状态、调度、恢复、模拟售后 |
| [docs/architecture.md](docs/architecture.md) | 架构总览 |
| [docs/langgraph-learning.md](docs/langgraph-learning.md) | 从这个客服项目学 LangGraph |
| [docs/hybrid-rag.md](docs/hybrid-rag.md) | 混合 RAG 与浏览器 MCP 降级 |
| [docs/memory-and-rerank.md](docs/memory-and-rerank.md) | 长期记忆、存储维护与 Rerank |
| [docs/business-lab.md](docs/business-lab.md) | 业务知识库与长期记忆实测 |
| [docs/database-overview.md](docs/database-overview.md) | 各 SQLite 数据库存了什么 |
| [evals/README.md](evals/README.md) | 评测集说明与执行方式 |
| [docs/evaluation-2026-11-04.md](docs/evaluation-2026-11-04.md) | 真实模型评测记录 |
| [docs/production-readiness.md](docs/production-readiness.md) | 上线前仍需完成的工作 |

## 推荐阅读顺序

1. `config.py`：配置、路径、密钥边界。
2. `model_factory.py`：模型超时、重试和 provider 分离。
3. `memory/checkpointer.py`：LangGraph thread memory。
4. `memory/compaction.py`：最近 N 轮与旧历史累计摘要。
5. `memory/profile.py` 与 `tools/profile_tool.py`：结构化长期记忆和运行时身份注入。
6. `skills/loader.py` 与 `tools/skill_tool.py`：渐进式 Skill 加载。
7. `rag/`：原子入库、去重、混合检索与 Rerank。
8. `agent.py`：动态 prompt、`create_agent` 装配、三层记忆编排和按模式选择服务的工厂。
9. `multi_agent/`：按 `state` → `prompts` → `specialists` → `workflow` → `service` 阅读。
10. `api.py`：依赖注入、错误处理、管理接口保护。
11. `tests/`：如何在不访问真实模型的情况下测试 Agent。

生产部署前仍需完成的工作见 `docs/production-readiness.md`。

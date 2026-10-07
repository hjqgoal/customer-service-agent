# codex_agent1：规范化智能客服 Agent 参考工程

这是一个用于学习真实 Agent 工程边界的 LangChain 项目。它可以在本地完整运行，也明确标注了
SQLite 教学实现与互联网生产部署之间的差距。

默认运行 **Supervisor + 订单 Agent + 物流 Agent + 售后 Agent** 的协作模式，支持依赖调度、
并行执行、失败降级和 SQLite 恢复。设置 `AGENT_MODE=single` 可以运行原版单 Agent。
已读过原版的同学，请先看 [新版学习指南](docs/multi-agent-learning.md)。

## 核心模块

- LangChain 1.x `create_agent`：工具调用 Agent，底层运行在 LangGraph。
- 多 Agent：外层 StateGraph + Send 调度，专业 Agent 以最小工具权限运行。
- LangGraph SQLite checkpointer：按 `thread_id` 持久化短期会话状态。
- 三层记忆：保留最近 N 轮、旧历史累计摘要、按 `user_id` 保存结构化客户画像。
- Markdown Skills：系统提示词只加载技能目录，完整 SOP 通过工具按需读取。
- Tools：订单查询、政策查询、工单创建、知识库检索和 Skill 加载。
- RAG：文档加载、字符切分、embedding、原子索引替换、哈希去重、相似度检索。
- FastAPI：聊天、健康检查、就绪检查、RAG 管理和稳定错误响应。
- CLI / MCP：本地调试和外部工具协议入口。
- 模拟售后：调用方确认、幂等申请和请求回执；不执行真实退款。
- 测试与质量：pytest、Ruff、pre-commit、GitHub Actions。

## 安装

推荐使用独立环境：

```bash
cd /Users/jiaqihong/Desktop/python/codex_agent1
conda activate my_demo1
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

当前 SQLite 向量检索是 O(n) 教学实现。大知识库请替换为 pgvector、Qdrant 或 Milvus。

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
  refund_after_sales/SKILL.md
  knowledge_base_qa/SKILL.md
  human_handoff/SKILL.md
```

`SKILL.md` front matter 由 `yaml.safe_load` 解析并校验。Agent 初始 prompt 只包含名称、描述和工具
目录；需要执行技能时调用 `load_skill_instructions` 加载完整 SOP。

## 推荐阅读顺序

1. `config.py`：配置、路径、密钥边界。
2. `model_factory.py`：模型超时、重试和 provider 分离。
3. `memory/checkpointer.py`：LangGraph thread memory。
4. `memory/compaction.py`：最近 N 轮与旧历史累计摘要。
5. `memory/profile.py` 与 `tools/profile_tool.py`：结构化长期记忆和运行时身份注入。
6. `skills/loader.py` 与 `tools/skill_tool.py`：渐进式 Skill 加载。
7. `rag/`：原子入库、去重和检索。
8. `agent.py`：动态 prompt、`create_agent` 装配与三层记忆编排。
9. `api.py`：依赖注入、错误处理、管理接口保护。
10. `tests/`：如何在不访问真实模型的情况下测试 Agent。

生产部署前仍需完成的工作见 `docs/production-readiness.md`。

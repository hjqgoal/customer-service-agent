# Architecture

默认 `AGENT_MODE=multi`，新的协作架构、状态流转和学习顺序见
[Multi-Agent 学习指南](multi-agent-learning.md)。原有模型、RAG 和记忆模块继续复用，
`AGENT_MODE=single` 保留原来的单 Agent 路径。

## Layers

- Transport: `api.py`, `cli.py`, `mcp_server.py`.
- Application: `agent.py`, `multi_agent/service.py`, `multi_agent/workflow.py`, `rag/service.py`.
- Domain capabilities: `tools/`, `skills/*/SKILL.md`.
- Infrastructure: model factory, LangGraph checkpointer, SQLite repositories.

## Single-agent chat request flow

1. API/CLI validates the user message and session/thread ID.
2. `CustomerServiceAgent` invokes LangChain `create_agent` with a LangGraph `thread_id`.
3. The SQLite checkpointer restores prior graph state for that thread.
4. Before the new turn, the compactor keeps the latest N user turns and replaces older turns with
   one cumulative, explicitly untrusted summary record.
5. Dynamic prompt middleware reads the structured profile by `user_id`; `session_id` and `user_id`
   are deliberately separate concepts.
6. The model selects tools. Skill metadata is already visible; full SOP is loaded only when needed.
7. Tool results return to the agent loop and the final AI message is produced.
8. LangGraph checkpoints the updated state. API responses omit internal traces by default.

## RAG indexing flow

1. CLI or the protected admin endpoint loads the configured knowledge directory.
2. Documents are split by character count. Source metadata is stored as a relative path.
3. All embeddings are generated before persistent data is modified.
4. Replacement happens in one SQLite transaction; append mode uses content hashes for deduplication.
5. Query vector dimension is checked against the stored index before similarity scoring.

## Persistence boundaries

- LangGraph SQLite checkpointer: short-term conversation state for local/single-instance use.
- Conversation compactor: exact recent-turn window plus cumulative summary in the thread state.
- Customer profile repository: whitelisted cross-session facts keyed by `user_id`.
- SQLite vector store: transparent educational RAG implementation with O(n) search.
- SQLite ticket repository: local demo adapter behind a repository boundary.

For multi-instance production, use PostgreSQL checkpointers, managed vector storage, and a real ticket/order
service. The fixed lock stripes only protect same-process compaction/invocation; distributed workers
still need single-writer routing or a database-backed concurrency design per `thread_id`. See
`production-readiness.md`.

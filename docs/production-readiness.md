# Production readiness boundary

This repository is now a practical reference project, but local SQLite components are still intended
for learning, development, and small single-instance deployments.

The multi-agent runtime supports a single process/service instance, asynchronous task concurrency,
per-session serialization, task failure isolation, and checkpoint recovery. It is not a distributed
scheduler. Run one Uvicorn worker. See `multi-agent-learning.md` for the new runtime and retry contract.

Order and logistics repositories are demo adapters. After-sales submissions are local simulated
applications, not payment operations. Before connecting real systems, add authenticated order
ownership checks and approvals bound to the specific order, action and amount. `confirm_action`
is a caller-supplied teaching flag, not a production authorization mechanism.

Completed request receipts survive chat-memory clearing for idempotency. Checkpoint history and
receipts require an explicit retention/deletion policy; limiting model context does not shrink them.

Before a real internet-facing production deployment, replace or add:

- PostgreSQL checkpointer for multi-instance agent memory and recovery.
- pgvector, Qdrant, Milvus, or another vector database for large RAG corpora.
- Company identity authentication in front of `/chat`; API gateway rate limiting and quotas.
- A secrets manager instead of server-local `.env` files.
- PII classification, field encryption, consent, retention, deletion, and audit policies for
  conversations, customer profiles, and tickets.
- Database migrations, backups, restore drills, and data lifecycle jobs.
- Centralized logs, metrics, traces, provider cost tracking, and alerting.
- Offline evaluations for tool selection, answer faithfulness, refusal, and escalation behavior.
- Human approval for high-risk tools such as refunds, account changes, and compensation.

The API protects local administrative operations with `X-Admin-Key` when `ADMIN_API_KEY` is set.
In `prod`, omitting `ADMIN_API_KEY` makes those operations unavailable.
The in-process per-session lock prevents local compaction races, but distributed deployments still
need single-writer routing or a database-backed concurrency strategy for each `thread_id`.

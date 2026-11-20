# 长期记忆、存储维护与Rerank

## 先区分三个概念

1. `ConversationCompactor`压缩**当前会话上下文**：保留最近N轮，把旧对话总结为结构化摘要。
2. `CustomerProfileRepository`保存**跨会话长期事实**：每用户一行，不追加全部聊天记录。
3. LangGraph的**历史checkpoint**：摘要更新本身也会产生新快照，旧快照不会因为摘要而消失。

所以“发给模型的内容变少”不等于“SQLite文件会变小”。

## 长期记忆现在怎样提炼

读取顺序：`tools/profile_tool.py` → `memory/profile.py` → `memory/consolidation.py`。

```text
用户明确表达信息
  → 聊天模型通过save_customer_profile提取白名单字段
  → Pydantic验证
  → BEGIN IMMEDIATE事务读取当前画像
  → 合并、规范化、精确去重、保留最近事实、检查UTF-8字节预算
  → 从当前问题/偏好覆盖生成memory_summary
  → UPSERT同一用户行
```

满额后保留**最近提到的**订单/偏好，修复了以前只保留最早20条的行为。
新问题摘要覆盖旧问题摘要，不会不断拼接。手机号、姓名、邮箱不交给摘要模型改写；
订单号保留原值，仅去首尾空白及执行既有单项长度限制。

这是“模型提取事实 + 确定性整理”的方案，**不是后台额外调用LLM归纳全部聊天**。
没有做语义等价偏好合并或矛盾推理，例如“短信联系”和“优先短信”仍可能是两条；
不为了节省一点空间自动猜测两条偏好是否等价。当前问题由保存工具提供简述，
存储层按长度限制和字节预算整理，不宣称无损语义压缩。重要业务证据仍应存订单/售后系统。

| 配置 | 默认 | 含义 |
| --- | --- | --- |
| `MEMORY_RECENT_TURNS` | 6 | 当前模型上下文保留的完整对话轮数 |
| `MEMORY_SUMMARY_ENABLED` | true | 旧对话使用模型摘要；关闭时走确定性兜底 |
| `MEMORY_SUMMARY_MAX_CHARS` | 4000 | 会话累计摘要上限，字符数 |
| `MEMORY_PROFILE_MAX_ORDERS` | 20 | 每用户近期订单上限 |
| `MEMORY_PROFILE_MAX_PREFERENCES` | 10 | 每用户偏好上限 |
| `MEMORY_PROFILE_SUMMARY_MAX_CHARS` | 800 | 长期画像摘要/当前问题长度上限 |
| `MEMORY_PROFILE_MAX_BYTES` | 8192 | 单用户业务载荷预算，含摘要，不含SQLite页和索引 |
| `MEMORY_PROFILE_RETENTION_DAYS` | 0 | 0不按年龄删除；大于0在维护时清理事实更新时间过旧的画像 |
| `MEMORY_CHECKPOINT_KEEP_LAST` | 20 | 维护时每会话、每namespace保留的快照数，不是轮数 |

预算超出时先移除最早偏好、再移除最早订单引用、最后缩短问题摘要；身份字段不丢弃。
这是有损保留策略，不是业务归档，不可把画像当作全部历史订单的唯一来源。
首次访问旧数据库会自动添加摘要列；旧行在下次写入或维护时整理。
画像删除年龄按`updated_at`即**事实最后更新**计算，不是用户最后登录时间。

## 怎样真正整理磁盘空间

先停止FastAPI、CLI聊天和MCP等所有正在访问这些数据库的进程。
维护命令不是在线清理协议，不能和正在执行的LangGraph任务并发运行。

```bash
# 默认只预览：不删记录、不生成备份、不做schema迁移。
python -m customer_service_agent.cli memory-maintain

# 明确执行：每个库自动生成独立备份，然后整理画像、裁剪旧checkpoint。
python -m customer_service_agent.cli memory-maintain --apply

# 需要操作系统看到文件缩小时，再显式回收空闲页。
python -m customer_service_agent.cli memory-maintain --apply --vacuum
```

检查输出的`profiles_to_compact`、`profiles_to_expire`、`checkpoints_to_prune`、
`database_bytes`及`reusable_bytes`。默认画像不会按年龄删除；旧checkpoint只保留配置数量。
裁剪同时删除对应的`writes`并断开已不存在的父快照引用；保留最新快照、其待处理写入和
其他业务表。超过保留范围的历史将无法回放，备份可供恢复。此实现针对已声明的
`langgraph-checkpoint-sqlite` 3.x表结构，升级依赖必须跑恢复测试。

SQLite的DELETE通常让空闲页可复用，不保证文件立刻缩小；VACUUM重写文件，可能阻塞并需要
额外磁盘空间，应在停服维护窗口操作。备份包含个人信息，保存后应按自己的备份策略清理。
维护未自动定时执行；没有设置保留期时用户总数仍能增长，子图namespace总数也没有全局上限。
不能把单用户预算理解为整个工程数据库总大小的硬上限。退款和售后幂等回执不在清理范围内。

## Rerank现在怎样执行

读取顺序：`rag/service.py` → `rag/reranker.py` → `tools/rag_tool.py`。

```text
关键词Top3 ∥ 语义Top3
  → 合并同一chunk，保留各路分数/排名
  → 必要时浏览器MCP补充公开证据
  → RRF融合名次，限制候选数
  → 独立Rerank模型同时读取query与候选正文
  → 校验返回index/score，按原索引取回证据，保留TopN
  → 最终证据进入Agent工具结果，由聊天模型作答
```

真实Rerank采用**DashScope原生文本排序HTTP协议**，适合已有百炼Embedding的学习环境。
它既不是Embedding模型，也不是聊天模型；只修改排序，不让接口返回的新文本覆盖原始证据。
目前没有实现其他厂商协议适配，其他服务不能仅换URL就保证可用。
RRF是名次融合算法，不是语义Rerank；未配置或接口故障时明确标记降级，不会冒充调用成功。

在自己的`.env`添加以下配置，重启服务。真实URL/密钥/模型名由你填写，现有`.env`未被改动：

```dotenv
RAG_RERANK_ENABLED=true
RAG_RERANK_API_URL=你的完整DashScope文本排序HTTPS接口地址
RAG_RERANK_API_KEY=你的排序服务API_KEY
RAG_RERANK_MODEL_NAME=你的排序模型名称
RAG_RERANK_TOP_N=3
RAG_RERANK_TIMEOUT_SECONDS=10
```

模型可按控制台可用性选`qwen3-rerank`；完整URL按地域/业务空间填写，不自动补`/v1`。
密钥不自动复用聊天或Embedding配置；同一账号可用相同密钥时也需显式填写。
协议依据[百炼文本排序官方API](https://help.aliyun.com/zh/model-studio/text-rerank-api)。

| 配置 | 默认 | 含义 |
| --- | --- | --- |
| `RAG_RERANK_ENABLED` | true | 开启排序流程；false恢复原来的分组证据输出 |
| `RAG_RERANK_API_URL` | 空 | 完整HTTPS接口地址，不允许携带凭证或查询参数 |
| `RAG_RERANK_API_KEY` | 空 | 排序服务密钥 |
| `RAG_RERANK_MODEL_NAME` | 空 | 独立的排序模型名 |
| `RAG_RERANK_TOP_N` | 3 | 最终交给LLM的证据数；也用于RRF降级 |
| `RAG_RERANK_MAX_CANDIDATES` | 20 | 送给排序模型的最大候选数 |
| `RAG_RERANK_TIMEOUT_SECONDS` | 10 | HTTP连接/读取等阶段的超时，不是整条RAG的总耗时 |
| `RAG_RERANK_MAX_DOCUMENT_CHARS` | 2000 | 每片段最多送多少字符，不是token数 |
| `RAG_RERANK_MIN_SCORE` | 0.0 | 排序分数过滤线；先保持0，再用标注集校准 |
| `RAG_RRF_K` | 60 | RRF平滑常数，不是返回条数 |

原来的`RAG_BRANCH_TOP_K=3`仍控制每路召回；要评估更大候选池，需同时调整请求`top_k`
及这个上限。推荐先保持各路3、最终3，再用测试集比较证据Recall、排序nDCG/MRR和答案质量。
阈值不是通用正确率。文本字符上限不能保证适合所有模型的token限制，需按所选模型调小。

```bash
python -m customer_service_agent.cli rag-search "退款规则"
python -m pytest tests/test_memory_retention.py tests/test_reranker.py -q
```

`rerank_status=success`且`ranking_method=model`才表示真实排序成功；
`not_configured`或`failed`配合`rrf`表示降级，`disabled`表示未启用。
API保留原始两路分组供诊断；开启流程后工具只把最终`hits`正文发给LLM，避免原始片段绕过
TopN过滤再次塞回上下文。API客户端也应读最终`hits`而不是自行拼接原始分组。
排序服务返回重复/越界索引、非法分数或超时会保留RRF结果并给出无敏感信息的错误代码。
若阈值过滤掉全部证据，明确返回`rerank_no_relevant_evidence`，不会重新塞回低分内容。
排序失败本身不会额外启动浏览器；本地检索失败/无结果或显式`force_web`的规则保持不变。
网页高相关不代表有权定义本店政策，所有引用仍按来源核验。

本地测试使用真实HTTP请求契约的MockTransport，不连接付费排序服务，不读取实际密钥。
它能验证工程逻辑，但不能证明所选模型的实际排序效果或账号权限。

参考：[LangGraph的会话与长期持久化边界](https://docs.langchain.com/oss/python/langgraph/persistence)、
[SQLite VACUUM](https://www.sqlite.org/lang_vacuum.html)。

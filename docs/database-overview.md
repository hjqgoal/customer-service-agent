# 项目数据库：现在存了什么，后续会存什么

> 本文及关联JSON是业务资料入库前的历史盘点。之后已新增48份业务知识及测试客户画像，最新数量、检索和长期记忆验证见 [业务知识库实测](business-lab.md)。

检查日期：2026-10-16。范围为当前项目 `data/` 下的 SQLite 文件，以及当前源码定义的写入逻辑。以下“后续会存”指触发对应功能并成功运行后的写入，不表示已经有这些数据，也不表示这些功能已通过本次运行验证。

## 一、当前实际已有的数据

| 文件 | 表和记录数 | 内容 |
|---|---|---|
| `data/agent_memory.sqlite3` | `chat_messages`：2 条 | 旧版聊天记录，1 条用户消息 + 1 条助手回复，共 1 轮 |
| `data/rag_chunks.sqlite3` | `rag_chunks`：1 条；`rag_keyword_chunks`：1 条；`rag_index_metadata`：0 条 | 1 段示例客服知识、1024 维向量和关键词索引 |
| `data/tickets.sqlite3` | `tickets`：0 条 | 已有工单表，但还没有工单 |
| `data/evaluation/fixtures.sqlite3` | `dataset_records`：34 条；`dataset_meta`：3 条 | 12 个虚构订单、10 条虚构物流记录、12 条虚构业务规则；另有用途、版本和虚构数据标记 |
| `data/evaluation/cases.sqlite3` | `dataset_records`：60 条；`dataset_meta`：3 条 | 60 个测试案例和验收标准；另有用途、版本和虚构数据标记 |
| `data/rag_before_hybrid_20261014_153ae6b1.sqlite3` | `rag_chunks`：1 条 | 知识库旧备份，内容也是示例客服知识 |

FTS 全文索引还有若干内部维护表，它们不代表额外的业务记录。测试案例是测试输入和预期结果，不是已执行 60 次测试的结果，也不是 60 位真实客户的聊天记录。

全部可读业务记录及实际建表语句见 [数据库内容快照](database-current-data.json)。其中 JSON 字符串已展开，向量保留维数和前 5 个数；联系方式做了脱敏。该快照不是用于恢复数据库的完整备份。

### 1. 已有聊天记录

会话编号：`demo`。两条记录的数据库时间均为 `2026-06-12 05:28:42`（原值，未转换时区）。

用户原文：

> 我想查订单 ORDER-1001

助手回复内容概述：订单已发货，物流公司为顺丰，单号为 `SF123456789`，预计 2 个工作日内送达，并提示可以追踪物流。

物理字段是 `id`、`session_id`、`message_json`、`created_at`。`message_json` 里面保存消息角色、正文以及附加元数据。这是旧版表；当前源码使用 LangGraph 检查点，不能据此认为这两条旧记录会自动成为新图的会话历史。

### 2. 已有知识库内容

来源：`data/knowledge_base/example_faq.md`，分块编号 `0`。

这段示例知识包含：普通商品签收后 7 天内申请无理由退款的示例规则、发货后通常 1～3 个工作日出现物流更新的示例说明，以及投诉等情况创建工单转人工的说明。这里只是在展示项目测试文档，不能当作真实商家的政策承诺。

同一条记录同时保存知识原文、来源信息和 1024 维 embedding 向量。向量用于相似度检索；关键词索引用于文本检索。当前向量维数是从实际向量数组长度读取，`rag_index_metadata` 目前为空。

### 3. 已有虚构订单示例

```json
{
  "id": "EVAL-ORDER-1001",
  "customer_id": "EVAL-USER-A",
  "product": "轻便双肩包",
  "amount_fen": 12900,
  "paid": true,
  "status": "已发货",
  "tracking_number": "EVAL-TRACK-1001",
  "condition": "未签收，商品状态未知"
}
```

这是实际测试库中的记录节选，金额单位是分，即 129 元。对应物流记录保存运输状态、预计到达说明和事件列表。业务规则保存规则编号、标题、关键词和规则正文。

### 4. 已有测试案例示例

案例 `FLT-001`：物流 Agent 超时仍保留独立成功结果。它保存用户问题、前置条件、故障注入方式、需要核验的证据、预期行为和禁止行为。例如，订单查询成功但物流超时时，应保留成功的订单结果并说明缺失依据，不能虚构物流。

## 二、当前代码定义的后续存储

### 1. 会话记忆和执行检查点

单 Agent 配置路径为 `data/agent_memory.sqlite3`；多 Agent 默认路径为 `data/multi_agent_checkpoints.sqlite3`，后者目前不存在。

当前实现通过 LangGraph 的 SQLite saver 保存检查点和中间写入。检查点包含会话标识、检查点标识、父检查点关系、序列化状态和元数据等；实际字段及编码由所安装的库版本决定，本次没有为展示而创建这些表。

业务上保存的内容包括：

- 用户与助手消息；单 Agent 的消息历史还可包含工具调用和工具返回。
- 压缩旧对话后生成的累计摘要。
- 多 Agent 的请求编号、任务计划、任务结果、待执行任务、规划错误和运行状态。
- 任务结果中可能包含证据、工具执行轨迹和错误信息。

目前默认保留最近 6 轮，包括当前轮；新问题进入前保留之前 5 轮。摘要默认最多 4000 字符，结构如下（这是字段示意，不是数据库已有记录）：

```json
{
  "confirmed_facts": ["已确认的用户事实、订单号等"],
  "current_request": "当前诉求",
  "tool_results": ["重要工具查询结果"],
  "promises": ["已作出的承诺"],
  "pending_actions": ["尚未完成的事项"]
}
```

**6 轮限制针对当前送入模型的消息状态，不等于数据库只保留 6 轮。** 压缩代码更新图状态，并未在该步骤清理历史检查点；较早的原文可能仍保存在旧检查点中。摘要也不会自动变成客户画像。

### 2. 跨会话客户画像

文件：`data/customer_profiles.sqlite3`；表：`customer_profiles`。当前文件不存在。

| 字段 | 含义 |
|---|---|
| `user_id` | 客户标识，同一客户不同会话共用 |
| `customer_name` | 姓名或称呼 |
| `phone`、`email` | 联系电话、邮箱 |
| `order_ids_json` | 已确认相关订单号列表 |
| `issue_summary` | 尚需处理的问题摘要 |
| `preferences_json` | 服务偏好列表 |
| `created_at`、`updated_at` | 创建和更新时间 |

只有调用保存客户信息工具或相应仓储写入方法时才会新增或更新，不是每句话自动提取全部字段。订单号和偏好会合并去重。清空某个会话不等于删除客户画像。

### 3. 人工工单及防重复记录

文件：`data/tickets.sqlite3`。

- `tickets`：`id` 工单号、`issue_type` 问题类型、`summary` 问题摘要、`contact` 联系方式、`priority` 优先级、`status` 状态、`created_at` 创建时间。初始状态为 `open`。
- `ticket_operations`：`operation_key` 操作唯一标识、`ticket_id` 对应工单号，用于请求重试时避免重复创建。当前实际数据库还没有该表，当前代码初始化仓储时会建表。

多 Agent 工单工具检查用户确认标记后再写入；旧单 Agent 工单工具没有同样的确认检查。它们创建的是本地工单，不能据此认定已接通真实人工客服系统。

### 4. 模拟售后申请

文件：`data/after_sales.sqlite3`；表：`after_sales_requests`。当前文件不存在。

字段为 `id` 申请号、`operation_key` 操作唯一标识、`user_id` 用户、`order_id` 订单号、`request_type` 申请类型、`reason` 原因、`status` 状态、`created_at` 创建时间。

类型支持 `refund`、`return`、`exchange`；初始状态为 `submitted`。工具检查确认标记及订单基本条件后写入。数据库还限制同一用户、订单和申请类型只能有一条待处理申请，以避免重复提交。写入代表本地模拟申请受理，不代表真实退款或审核通过。

### 5. 多 Agent 会话归属与请求回执

和多 Agent 检查点共用 `data/multi_agent_checkpoints.sqlite3`，当前文件不存在。

| 表 | 字段及用途 |
|---|---|
| `workflow_sessions` | `session_id`、`user_id`：会话属于哪个用户 |
| `workflow_requests` | `session_id`、`request_id`：联合识别一次请求；`fingerprint`：问题和确认标记的摘要；`response_json`：完成后的响应；`abandoned`：请求是否被取消 |

用途是检查会话归属、识别重复请求、缓存已完成结果和辅助失败后重试。清空会话时，代码保留已完成回执，并标记尚未完成的请求，避免旧业务请求重新执行。

### 6. 知识库更新和评估数据导入

导入新知识文档时，会继续写入 `rag_chunks` 的 `content`、`metadata_json`、`embedding_json`、`content_hash`、`created_at` 等字段，并维护关键词索引。`content_hash` 用于去重，`rag_index_metadata` 可记录向量维度。

评估数据导入时会写 `dataset_meta` 和 `dataset_records`。后者用 `kind` 区分订单、物流、规则或案例，`id` 标识记录，`payload` 保存完整 JSON。业务事实库和案例答案库分开，评估业务工具不能连接案例答案库。

## 三、容易混淆的地方

1. 当前默认演示订单适配器的 `ORDER-1001` 等订单写在代码中；评估数据库中的 `EVAL-ORDER-1001` 等订单属于另一套测试素材。不能因为测试库有订单，就认为默认订单工具已经自动切换到该库。
2. 数据库中保存了什么、每次实际送给模型什么，是两个问题。历史检查点、长期画像、知识库和工具结果按各自逻辑读取，并非所有数据库记录每次都塞进提示词。
3. 本文区分实际存在的表与代码计划创建的表；没有启动 Agent、调用模型或提交业务申请来补齐空表。本次仅生成说明和可读快照。

实现依据：`config.py`、`memory/checkpointer.py`、`memory/compaction.py`、`memory/profile.py`、`multi_agent/service.py`、`multi_agent/request_store.py`、`multi_agent/state.py`、`tools/ticket_tool.py`、`tools/after_sales_tool.py`、`rag/sqlite_vector_store.py` 和 `evaluation/store.py`。

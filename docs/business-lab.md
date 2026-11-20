# 业务知识库与长期记忆实测

本数据集是星舟生活馆的**模拟业务资料**，不是实际商家政策、真实订单或真实物流。固定快照时间为 2026-10-16 12:00（北京时间），之后测试仍按此时点解释签收天数及运输状态。未导入真实地址、电话或身份信息。

## 已准备的资料

48 份独立文档，每份不足 700 字符，可各自成为一个完整检索片段：

| 类型 | 数量 | 编号与范围 |
|---|---:|---|
| 业务规则 | 12 | LAB-P01～LAB-P12：退货、退款、定制、质量问题、发货、延误、签收异常、运费、发票、优惠、地址修改、客服与偏好 |
| 订单 | 20 | LAB-ORDER-1001～LAB-ORDER-1020 |
| 物流 | 16 | LAB-TRACK-1001～LAB-TRACK-1016，与订单一一关联；后4单没有运单 |

原文在 `data/knowledge_base/business_lab/`，订单总览在 `data/business_lab_catalog.json`。原有知识文档保留。测试题及预期答案放在本文，不放进知识库，避免直接检索到测试答案。

## 本次实际验证结果

2026-10-16 已完成真实服务调用与数据库核对：

- 新增48个片段，连同原有示例共49个；每条向量为1024维。首次导入时重新生成了原有示例的向量，确保属于同一模型的向量空间。
- 旧索引备份：`data/rag_before_business_lab_20261016_153536.sqlite3`。重复入库新增0条，总数仍为49。
- 6组检索核对全部通过，关键词与语义检索均正常，没有启用网页降级；详细候选见 [检索实测记录](business-lab-retrieval-checks.json)。通过表示目标来源在合并候选中，并不表示每路第一名都正确。例如预售订单1019由关键词检索准确排第一，语义第一名仍可能是别的订单，回答时必须核对编号。
- 使用真实聊天模型查询订单1008时，回答引用了订单1008、运单1008以及P06物流核查规则，说明停滞超过48小时，并明确不能保证到达或直接认定丢件。
- 测试用户 `lab-memory-check-20261016` 在会话 `lab-check-a-20261016` 实际调用了 `save_customer_profile`，保存了“小林、LAB-ORDER-1008、物流停滞、只接受邮件联系且不要打电话”。
- 另一个进程在新会话 `lab-check-b-20261016` 下使用同一个用户ID，正确回答了姓名、订单及联系偏好，没有重新查询知识库。数据库中也核实了对应画像。你的手工测试用户 `lab-memory-manual` 未预填。
- 相关自动化检查：新增向量适配、现有RAG、混合检索和记忆测试，共39项通过。未宣称运行了整个项目的全部测试。

为使现有 `qwen3-vl-embedding` 配置可用，模型工厂已在官方北京服务域名下改用原生多模态向量接口的纯文本输入，保留原有密钥和模型名，并对其他模型及自定义兼容网关保留原路径。[阿里云接口依据](https://www.alibabacloud.com/help/en/model-studio/multimodal-embedding-api-reference)

## 使用方式

在项目根目录打开终端，先激活项目环境：

```bash
source .venv/bin/activate  # 或激活你自己的 conda 环境
```

本实验入口使用项目真实的 RAG 服务、聊天模型、SQLite 检查点和客户画像工具。为专门验证 RAG，采用单 Agent，并仅提供知识检索、技能加载和画像保存工具，不混入旧的固定演示订单查询工具。它不修改项目默认运行模式，不执行售后或工单写操作。

聊天和向量化沿用 `.env` 已有模型配置，会产生相应服务的调用用量。实验关闭网页降级，模拟订单只查本地知识库。每次 `ask` 默认生成新会话；指定相同 `--session` 才接续短期历史，指定相同 `--user` 才共享长期画像。

### 直接测试检索

```bash
python -m examples.business_lab search "LAB-ORDER-1008 的物流为什么停了？"
python -m examples.business_lab search "普通商品签收第七天和第八天能不能无理由退货？"
python -m examples.business_lab search "LAB-TRACK-1009 显示签收但没收到"
```

结果分别列出 `keyword_hits` 和 `semantic_hits`，包含原文、来源和分数。语义分数不是答案正确率。编号检索尤其要核对命中的完整编号，不能把相邻订单当成目标订单。

### 让 Agent 基于 RAG 回答

```bash
python -m examples.business_lab ask "请查知识库：LAB-ORDER-1008 为什么还没到？再查相关物流核查规则并引用来源。" --user lab-user-manual --session lab-rag-a --trace
python -m examples.business_lab ask "请查知识库并比较 LAB-ORDER-1003 和 LAB-ORDER-1004，按快照时间哪个还在七天无理由窗口内？" --user lab-user-manual --session lab-rag-b --trace
```

| 可提问的问题 | 核对重点 |
|---|---|
| LAB-ORDER-1003 和 1004 哪个可申请无理由退货？ | 1003 签收第7天，1004 第8天；还要核对商品状态，申请不等于批准 |
| LAB-ORDER-1005 刻字杯能无理由退吗？ | 定制商品不适用普通无理由规则；质量问题另行核验 |
| LAB-ORDER-1006 开箱发现破损怎么办？ | 保留照片和面单，核验质量问题，不能直接声称已退款 |
| LAB-ORDER-1008 为什么物流不动？ | 10月13日08:00后停滞，快照时超过48小时，可登记核查；不自动认定丢件 |
| LAB-ORDER-1009 已签收却没收到怎么办？ | 核对驿站代收和签收凭证，不直接认定丢件 |
| LAB-ORDER-1011 退款何时到？ | 退货仓库已签收但待验收；验收后发起退款，到账不能保证 |
| LAB-ORDER-1017 为什么没发货？ | 未付款，无运单 |
| LAB-ORDER-1019 是不是10月20日送到？ | 10月20日是约定发货日期，不是送达承诺 |
| LAB-ORDER-9999 现在在哪？ | 无该订单；即使检索返回其他订单，也必须说未知 |
| 满99元包邮吗，积分能和券一起用吗？ | P08运费条件，P10优惠券与积分不叠加 |

### 测试真正的跨会话长期记忆

第一个会话告诉 Agent 要记住的事实：

```bash
python -m examples.business_lab ask "我叫小林，请记住我只接受邮件联系，不要打电话。我关注的订单是 LAB-ORDER-1008，当前问题是物流停滞。请保存这些信息，这次不需要查订单。" --user lab-memory-manual --session memory-a --trace
```

直接查看数据库里有没有保存成功：

```bash
python -m examples.business_lab profile --user lab-memory-manual
```

关闭进程后另开一个会话，保持同一用户：

```bash
python -m examples.business_lab ask "我叫什么？我之前关注哪个订单？我希望通过什么方式联系？只根据你已经记住的信息回答。" --user lab-memory-manual --session memory-b --trace
```

应回答“小林、LAB-ORDER-1008、邮件联系且不打电话”。`memory-b` 没有 `memory-a` 的聊天历史，回答依据应来自 `customer_profiles.sqlite3` 中的长期画像。测试用户是虚构的，不要填入真实隐私信息。

再换一个全新用户检查隔离：

```bash
python -m examples.business_lab ask "你记得我的名字和联系偏好吗？不要猜。" --user lab-memory-other --session memory-other
```

不应把小林的信息当成这个新用户的画像。若想测试修改偏好，可在原用户下明确说“请把联系偏好改为短信”；当前画像仓储会合并偏好列表，若发现新旧偏好共存，需要继续完善冲突替换逻辑，不能把“已存储”误当作“冲突已解决”。

## 重新入库

```bash
python -m examples.business_lab ingest
```

它会备份当前索引，再追加这些资料；若旧索引缺少本实验的模型来源标记，会保留旧片段内容并统一重建向量。相同内容、相同模型重复执行不重复入库，也不重复生成向量。修改文档后追加会保留旧版本；如需以当前目录为准完整重建，应先备份并使用项目原有全量 `rag-ingest` 流程。实验不预填手工测试用户的画像，画像应由你的聊天实际触发保存。

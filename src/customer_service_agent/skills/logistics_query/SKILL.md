---
name: 物流查询
description: 查询运输轨迹、签收与物流异常时使用；仅适用于具备物流工具的 Agent。
tools:
  - lookup_logistics
---

# 物流查询

1. 从用户输入或订单 Agent 的已核实结果取得运单号，不能自行推测号码。
2. 调用 `lookup_logistics` 获取轨迹，区分运输中、已签收、退货途中等状态。
3. 数据标注 `source=demo` 时，向用户说明这是演示结果。
4. 没有物流记录不代表订单不存在或未发货，应说明本次未查到。
5. 将轨迹、预计进度和异常返回 Supervisor。退款资格由售后 Agent 综合政策判断。

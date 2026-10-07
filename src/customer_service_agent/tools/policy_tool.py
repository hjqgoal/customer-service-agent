"""政策查询工具。

这是一个非常小的规则库示例。真实项目里可以改成读取数据库、CMS、内部 API，
或者直接让 RAG 知识库承接更复杂的政策查询。
"""

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class PolicyLookupInput(BaseModel):
    topic: str = Field(..., description="政策主题，例如 退款、物流、发票、会员")


POLICIES = {
    "退款": "普通商品签收后 7 天内可申请无理由退款，商品需不影响二次销售。",
    "退货": "退货需提供订单号、退货原因、商品照片。审核通过后会发送退货地址。",
    "物流": "发货后通常 1 到 3 个工作日可查物流。节假日、天气原因可能延迟。",
    "发票": "电子发票一般在订单完成后 24 小时内开具，可在订单详情页下载。",
    "会员": "会员权益以用户当前等级为准，优惠券和积分不可兑换现金。",
}


def lookup_policy(topic: str) -> str:
    """按关键词查询政策。"""

    normalized_topic = topic.strip()
    for keyword, text in POLICIES.items():
        if keyword in normalized_topic:
            return f"{keyword}政策：{text}"

    return (
        f"没有找到与“{normalized_topic}”直接匹配的固定政策。"
        "建议调用 RAG 检索知识库；如果仍无答案，则创建工单。"
    )


def build_policy_lookup_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=lookup_policy,
        name="lookup_policy",
        description="查询常见客服政策，例如退款、退货、物流、发票、会员。",
        args_schema=PolicyLookupInput,
    )

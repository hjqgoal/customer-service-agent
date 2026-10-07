from typing import Any

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool


class FakeEmbeddings(Embeddings):
    keywords = ["退款", "订单", "物流", "工单", "发票", "会员", "售后", "投诉"]

    def _embed(self, text: str) -> list[float]:
        return [float(text.count(keyword)) for keyword in self.keywords] + [float(len(text) % 17)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class FailingEmbeddings(FakeEmbeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding service unavailable")


class FakeToolCallingChatModel(BaseChatModel):
    response_text: str = "这是一个测试用客服回复。"

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling-chat-model"

    def bind_tools(self, tools: list[BaseTool], **kwargs: Any) -> "FakeToolCallingChatModel":
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self.response_text))]
        )


class FakeProfileSavingChatModel(FakeToolCallingChatModel):
    """Ask for one profile tool call, then return a final answer."""

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if messages and isinstance(messages[-1], ToolMessage):
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="客户信息已经记住。"))]
            )
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "save_customer_profile",
                                "args": {
                                    "customer_name": "张三",
                                    "order_ids": ["ORDER-1001"],
                                    "preferences": ["优先短信联系"],
                                },
                                "id": "profile-call-1",
                                "type": "tool_call",
                            }
                        ],
                    )
                )
            ]
        )

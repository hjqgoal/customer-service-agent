"""统一构建 LangChain 模型。

把模型构建集中在这里有两个好处：
1. 以后换模型、换 base_url、换温度参数，只改这个文件和 .env。
2. 业务代码只依赖 LangChain 的接口，不关心底层供应商细节。
"""

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

from customer_service_agent.config import Settings


class ModelConfigError(RuntimeError):
    """模型配置缺失或不合法时抛出。"""


def _is_placeholder(value: str | None) -> bool:
    """判断配置是否还是示例占位值。"""

    if value is None:
        return True
    stripped = value.strip()
    return not stripped or stripped.startswith("your-")


def _secret_value(secret: SecretStr | None, *, env_name: str, purpose: str) -> str:
    """读取 SecretStr，同时给出更清楚的错误提示。"""

    if secret is None:
        raise ModelConfigError(f"缺少 {env_name}，无法创建{purpose}。请在 .env 中填写。")

    value = secret.get_secret_value()
    if _is_placeholder(value):
        raise ModelConfigError(f"{env_name} 还是占位值，无法创建{purpose}。请换成真实配置。")
    return value


def _plain_value(value: str | None, *, env_name: str, purpose: str) -> str:
    if _is_placeholder(value):
        raise ModelConfigError(
            f"{env_name} 还是空值或占位值，无法创建{purpose}。请在 .env 中填写。"
        )
    assert value is not None
    return value.strip()


def build_chat_model(settings: Settings) -> BaseChatModel:
    """创建聊天模型。

    ChatOpenAI 支持 OpenAI-compatible 服务，所以你可以在 .env 中用 CHAT_BASE_URL
    指向自己的聊天模型网关。旧版 OPENAI_BASE_URL 仍然兼容。
    """

    return ChatOpenAI(
        model=_plain_value(
            settings.resolved_chat_model_name,
            env_name="CHAT_MODEL_NAME",
            purpose="聊天模型",
        ),
        api_key=_secret_value(
            settings.resolved_chat_api_key,
            env_name="CHAT_API_KEY",
            purpose="聊天模型",
        ),
        base_url=settings.resolved_chat_base_url,
        temperature=settings.agent_temperature,
        timeout=settings.model_timeout_seconds,
        max_retries=settings.model_max_retries,
    )


def build_embedding_model(settings: Settings) -> Embeddings:
    """创建 embedding 模型，用于 RAG 文档向量化和查询向量化。"""

    return OpenAIEmbeddings(
        model=_plain_value(
            settings.resolved_embedding_model_name,
            env_name="EMBEDDING_MODEL_NAME",
            purpose="Embedding 模型",
        ),
        api_key=_secret_value(
            settings.resolved_embedding_api_key,
            env_name="EMBEDDING_API_KEY",
            purpose="Embedding 模型",
        ),
        base_url=settings.resolved_embedding_base_url,
        timeout=settings.model_timeout_seconds,
        max_retries=settings.model_max_retries,
        # LangChain 默认会先把文本转成 token id 数组再请求 embeddings。
        # OpenAI 官方接口支持这种形式，但 DashScope 等兼容接口通常只接受 str/list[str]。
        # 关闭后会直接把原始文本字符串发送给 embedding 服务。
        check_embedding_ctx_length=settings.embedding_check_ctx_length,
    )

"""浏览器MCP客户端：导航公开搜索页，再读取页面快照。

只调用browser_navigate和browser_snapshot，不把网页指令当作工具调用计划。
stdio每次使用独立进程；HTTP服务应启用隔离会话，不共享已登录浏览器。
"""

import asyncio
import ipaddress
import os
import re
import threading
from contextlib import AsyncExitStack
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from customer_service_agent.config import Settings
from customer_service_agent.models import RAGHit

PRIVATE_QUERY = re.compile(
    r"(?i)(?:[a-z0-9]+-)?(?:order|track)[-_]?[a-z0-9-]*\d[a-z0-9-]*|"
    r"(?:sf|yt|jd)\d{5,}|(?<!\d)\d{6,}(?!\d)|"
    r"[\w.+-]+@[\w.-]+\.[a-z]{2,}|sk-[a-z0-9_-]+|"
    r"密码|验证码|身份证|银行卡|手机号|收货地址|我的订单|这笔订单"
)


class WebSearchError(RuntimeError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def search_url(endpoint: str, query: str) -> str:
    if PRIVATE_QUERY.search(query):
        raise WebSearchError("web_query_requires_public_terms")
    parts = urlsplit(endpoint)
    host = parts.hostname or ""
    if parts.scheme != "https" or not host or parts.username or parts.password:
        raise WebSearchError("web_search_url_invalid")
    if host.lower() in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
        raise WebSearchError("web_search_url_invalid")
    try:
        if not ipaddress.ip_address(host).is_global:
            raise WebSearchError("web_search_url_invalid")
    except ValueError:
        pass  # 域名由浏览器解析；搜索端点来自管理员配置，不接受模型任意指定URL。
    parameters = [(k, v) for k, v in parse_qsl(parts.query) if k != "q"]
    parameters.append(("q", query))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(parameters), ""))


def _error_code(exc: BaseException) -> str:
    if isinstance(exc, WebSearchError):
        return exc.code
    if isinstance(exc, TimeoutError):
        return "web_timeout"
    # AnyIO传输层可能用ExceptionGroup包装内部异常，保留可理解的错误代码。
    if isinstance(exc, BaseExceptionGroup):
        for nested in exc.exceptions:
            code = _error_code(nested)
            if code != "web_mcp_unavailable":
                return code
    return "web_mcp_unavailable"


class MCPBrowserSearch:
    def __init__(self, settings: Settings):
        self.settings = settings
        # 同一客户端的导航与快照成对执行，避免两个请求互相读取搜索页。
        self._lock = threading.Lock()

    def search(self, query: str) -> list[RAGHit]:
        url = search_url(self.settings.rag_web_search_url, query)
        settings = self.settings
        if settings.rag_web_mcp_transport == "stdio":
            if not settings.rag_web_mcp_command or settings.rag_web_mcp_command.startswith("your-"):
                raise WebSearchError("web_mcp_not_configured")
        elif not settings.rag_web_mcp_url:
            raise WebSearchError("web_mcp_not_configured")
        try:
            if not self._lock.acquire(timeout=settings.rag_web_timeout_seconds):
                raise WebSearchError("web_busy")
            try:
                text = asyncio.run(self._snapshot(url))
            finally:
                self._lock.release()
        except Exception as exc:
            raise WebSearchError(_error_code(exc)) from None
        if not text.strip():
            return []
        if re.search(
            r"captcha challenge|verify you are human|验证您是真人|验证你是真人", text, re.I
        ):
            raise WebSearchError("web_verification_required")
        if re.search(r"no results found|没有找到相关结果|未找到搜索结果", text, re.I):
            return []
        return [
            RAGHit(
                content=text[: settings.rag_web_max_chars],
                score=0.0,
                metadata={
                    "source": url,
                    "retrieval_methods": ["web"],
                    "retrieval_scores": {"web": 0.0},
                    "retrieval_ranks": {"web": 1},
                    "score_type": "not_scored",
                    "evidence_scope": "public_search_page",
                    "fetched_at": datetime.now(UTC).isoformat(),
                    "verified_shop_policy": False,
                },
            )
        ]

    async def _snapshot(self, url: str) -> str:
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            from mcp.client.streamable_http import streamablehttp_client
        except ImportError:
            raise WebSearchError("web_mcp_dependency_missing") from None
        settings = self.settings
        async with AsyncExitStack() as stack:
            # 只让超时包围连接、工具调用和退出；退出时SDK负责关闭会话/子进程。
            await stack.enter_async_context(asyncio.timeout(settings.rag_web_timeout_seconds))
            if settings.rag_web_mcp_transport == "stdio":
                workspace = stack.enter_context(TemporaryDirectory(prefix="rag-browser-mcp-"))
                errlog = stack.enter_context(open(os.devnull, "w", encoding="utf-8"))
                process_env = {
                    "PLAYWRIGHT_MCP_ISOLATED": "true",
                    "PLAYWRIGHT_MCP_HEADLESS": "true",
                    "PLAYWRIGHT_MCP_IMAGE_RESPONSES": "omit",
                }
                if Path(settings.rag_web_mcp_command).name in {"npx", "npx.cmd", "npm", "npm.cmd"}:
                    cache = settings.resolve_path(settings.rag_web_mcp_cache_dir)
                    cache.mkdir(parents=True, exist_ok=True)
                    process_env["npm_config_cache"] = str(cache)
                params = StdioServerParameters(
                    command=settings.rag_web_mcp_command,
                    args=settings.rag_web_mcp_args,
                    cwd=workspace,
                    env=process_env,
                )
                read, write = await stack.enter_async_context(stdio_client(params, errlog=errlog))
            else:
                endpoint = settings.rag_web_mcp_url
                parts = urlsplit(endpoint)
                if parts.scheme not in {"http", "https"} or not parts.hostname:
                    raise WebSearchError("web_mcp_url_invalid")
                headers = {}
                if settings.rag_web_mcp_token:
                    headers["Authorization"] = (
                        "Bearer " + settings.rag_web_mcp_token.get_secret_value()
                    )
                read, write, _ = await stack.enter_async_context(
                    streamablehttp_client(
                        endpoint,
                        headers=headers,
                        timeout=settings.rag_web_timeout_seconds,
                        sse_read_timeout=settings.rag_web_timeout_seconds,
                    )
                )
            session = await stack.enter_async_context(
                ClientSession(
                    read,
                    write,
                    read_timeout_seconds=timedelta(seconds=settings.rag_web_timeout_seconds),
                )
            )
            await session.initialize()
            available = {tool.name for tool in (await session.list_tools()).tools}
            if not {"browser_navigate", "browser_snapshot"}.issubset(available):
                raise WebSearchError("web_mcp_browser_tools_missing")
            navigation = await session.call_tool("browser_navigate", {"url": url})
            if navigation.isError:
                raise WebSearchError("web_navigation_failed")
            snapshot = await session.call_tool("browser_snapshot", {})
            if snapshot.isError:
                raise WebSearchError("web_snapshot_failed")
            return "\n".join(block.text for block in snapshot.content if block.type == "text")

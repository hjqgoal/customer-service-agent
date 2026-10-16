"""协议集成测试用浏览器MCP替身：不启动浏览器、不连接外网。"""

import sys
from urllib.parse import parse_qs, urlsplit

from mcp.server.fastmcp import FastMCP

server = FastMCP("rag-browser-protocol-fixture")
current_url = ""
mode = sys.argv[1] if len(sys.argv) > 1 else "normal"


@server.tool()
def browser_navigate(url: str) -> str:
    """记录本次导航，测试stdio会话里的状态传递。"""
    global current_url
    current_url = url
    return "Navigation completed"


@server.tool()
def browser_snapshot() -> str:
    """返回预先定义的快照，而非真实在线搜索结果。"""
    if mode == "error":
        raise RuntimeError("test browser failed")
    if mode == "empty":
        return ""
    query = parse_qs(urlsplit(current_url).query)["q"][0]
    return (
        "### Page snapshot\n"
        f"- heading: 测试搜索主题 {query}\n"
        "- link: SQLite FTS5 official documentation\n"
        "  - /url: https://www.sqlite.org/fts5.html\n"
        "- paragraph: This is a simulated MCP snapshot for protocol testing."
    )


if __name__ == "__main__":
    server.run()

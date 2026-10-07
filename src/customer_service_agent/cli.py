"""Command-line interface for development and operations."""

import argparse
import sys
import uuid

import uvicorn

from customer_service_agent.agent import build_customer_service_agent
from customer_service_agent.config import get_settings
from customer_service_agent.logging_config import configure_logging
from customer_service_agent.model_factory import build_embedding_model
from customer_service_agent.models import ChatResponse
from customer_service_agent.rag.service import RAGService


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="customer-agent")
    parser.add_argument("--debug", action="store_true", help="发生错误时打印完整 traceback")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask_parser = subparsers.add_parser("ask", help="单轮提问")
    ask_parser.add_argument("message")
    ask_parser.add_argument("--session", default="default")
    ask_parser.add_argument("--user", default=None, help="跨会话稳定用户 ID")
    ask_parser.add_argument("--trace", action="store_true", help="显示本轮工具调用轨迹")
    ask_parser.add_argument("--request-id", default=None, help="重试同一请求时复用 ID")
    ask_parser.add_argument("--confirm-action", action="store_true", help="确认本次模拟售后写操作")

    chat_parser = subparsers.add_parser("chat", help="多轮命令行聊天")
    chat_parser.add_argument("--session", default="default")
    chat_parser.add_argument("--user", default=None, help="跨会话稳定用户 ID")
    chat_parser.add_argument("--trace", action="store_true")

    ingest_parser = subparsers.add_parser("rag-ingest", help="构建 RAG 索引")
    ingest_parser.add_argument("--path", default=None)
    ingest_parser.add_argument("--append", action="store_true", help="增量添加，不替换现有索引")

    search_parser = subparsers.add_parser("rag-search", help="搜索 RAG 知识库")
    search_parser.add_argument("query")
    search_parser.add_argument("--top-k", type=int, default=4)

    serve_parser = subparsers.add_parser("serve", help="启动 FastAPI")
    serve_parser.add_argument("--host", default=None)
    serve_parser.add_argument("--port", type=int, default=None)
    serve_parser.add_argument("--reload", action="store_true")
    return parser


def _print_response(response: ChatResponse, *, show_trace: bool) -> None:
    print(f"\nAgent：\n{response.answer}\n")
    if show_trace and response.mode == "multi":
        print(f"协作请求：{response.request_id}；状态：{response.status}")
        for task in response.tasks:
            print(
                f"- {task.task_id} [{task.agent}] {task.status} "
                f"attempts={task.attempts} error={task.error_code or '-'}"
            )
    if show_trace and response.intermediate_steps:
        print("工具调用：")
        for step in response.intermediate_steps:
            print(f"- {step.tool}: {step.tool_input} -> {step.observation}")


def _run(args: argparse.Namespace) -> None:
    settings = get_settings()
    configure_logging(settings)

    if args.command == "ask":
        request_id = args.request_id or uuid.uuid4().hex
        if settings.agent_mode == "multi":
            print(f"请求 ID：{request_id}（同一请求重试时使用 --request-id）")
        with build_customer_service_agent(settings) as agent:
            response = agent.ask(
                args.message,
                session_id=args.session,
                user_id=args.user,
                include_trace=args.trace,
                request_id=request_id,
                confirm_action=args.confirm_action,
            )
            _print_response(response, show_trace=args.trace)
        return

    if args.command == "chat":
        with build_customer_service_agent(settings) as agent:
            print("进入智能客服 CLI。输入 exit 或 quit 退出。")
            while True:
                message = input("你：").strip()
                if message.lower() in {"exit", "quit"}:
                    return
                if message:
                    response = agent.ask(
                        message,
                        session_id=args.session,
                        user_id=args.user,
                        include_trace=args.trace,
                    )
                    _print_response(response, show_trace=args.trace)

    if args.command == "rag-ingest":
        service = RAGService(settings, build_embedding_model(settings))
        result = service.ingest_directory(args.path, replace=not args.append)
        print(
            f"读取文档 {result.loaded_documents} 个，写入 chunk {result.indexed_chunks} 个，"
            f"当前总 chunk {result.total_chunks} 个。"
        )
        return

    if args.command == "rag-search":
        service = RAGService(settings, build_embedding_model(settings))
        hits = service.search(args.query, top_k=args.top_k)
        if not hits:
            print("没有检索到相关内容。请先执行 rag-ingest。")
            return
        for index, hit in enumerate(hits, start=1):
            source = hit.metadata.get("source", "unknown")
            print(f"\n[{index}] score={hit.score:.3f} source={source}")
            print(hit.content)
        return

    if args.command == "serve":
        uvicorn.run(
            "customer_service_agent.api:app",
            host=args.host or settings.api_host,
            port=args.port or settings.api_port,
            reload=args.reload,
        )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        _run(args)
    except Exception as exc:
        if args.debug:
            raise
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()

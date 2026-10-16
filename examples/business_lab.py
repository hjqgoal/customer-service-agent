"""使用项目真实 RAG、客户画像和检查点测试模拟业务资料。

运行：python -m examples.business_lab --help
复用 .env 中已有的聊天和 embedding 配置，不打印凭据。
"""

import argparse
import json
import sqlite3
import uuid
from datetime import datetime

from langchain_core.documents import Document

from customer_service_agent.agent import build_customer_service_agent
from customer_service_agent.config import Settings
from customer_service_agent.memory import CustomerProfileRepository
from customer_service_agent.model_factory import build_embedding_model
from customer_service_agent.rag.loader import load_documents, split_documents
from customer_service_agent.rag.service import RAGService
from customer_service_agent.tools.profile_tool import build_customer_profile_tools
from customer_service_agent.tools.rag_tool import build_rag_search_tool
from customer_service_agent.tools.skill_tool import build_skill_loader_tool


def lab_settings() -> Settings:
    # 固定快照只查本地资料，不把模拟订单号送往公开网页。
    return Settings().model_copy(update={"agent_mode": "single", "rag_web_fallback_enabled": False})


def build_rag(settings: Settings) -> RAGService:
    embeddings = build_embedding_model(settings)
    # 小批次兼容现有 embedding 服务，仍生成真实语义向量。
    embeddings.chunk_size = 8
    return RAGService(settings, embeddings)


def ingest(settings: Settings) -> dict:
    # 统一从知识库根目录加载，保证以后全量入库时 source/hash 保持一致。
    documents = [
        doc
        for doc in load_documents(settings.knowledge_base_path)
        if doc.metadata["source"].startswith("business_lab/")
    ]
    if not documents:
        raise ValueError("未找到 business_lab 业务资料")
    chunks = split_documents(
        documents, chunk_size=settings.rag_chunk_size, chunk_overlap=settings.rag_chunk_overlap
    )
    backup = None
    if settings.rag_db_file.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = settings.rag_db_file.with_name(f"rag_before_business_lab_{stamp}.sqlite3")
        with sqlite3.connect(settings.rag_db_file) as source:
            with sqlite3.connect(backup) as target:
                source.backup(target)
    with build_rag(settings) as rag:
        with sqlite3.connect(settings.rag_db_file) as connection:
            hashes = {row[0] for row in connection.execute("SELECT content_hash FROM rag_chunks")}
            model_row = connection.execute(
                "SELECT value FROM rag_index_metadata WHERE key = 'business_lab_embedding_model'"
            ).fetchone()
        pending = [doc for doc in chunks if rag.store._content_hash(doc) not in hashes]
        model_name = settings.resolved_embedding_model_name
        # 旧索引没有模型来源记录，即使维数相同也不能假设在同一向量空间。
        if not model_row or model_row[0] != model_name:
            with sqlite3.connect(settings.rag_db_file) as connection:
                existing = [
                    Document(page_content=text, metadata=json.loads(metadata))
                    for text, metadata in connection.execute(
                        "SELECT content, metadata_json FROM rag_chunks"
                    )
                ]
            rag.store.replace_documents([*existing, *pending])
            added = len(pending)
            with sqlite3.connect(settings.rag_db_file) as connection:
                connection.execute(
                    "INSERT OR REPLACE INTO rag_index_metadata(key, value) VALUES (?, ?)",
                    ("business_lab_embedding_model", model_name),
                )
        else:
            # 已入库的相同资料不再调用 embedding。
            added = rag.store.add_documents(pending) if pending else 0
        return {
            "documents": len(documents),
            "added_chunks": added,
            "total_chunks": rag.store.count(),
            "backup": str(backup) if backup else None,
        }


def ask(settings: Settings, message: str, user: str, session: str, trace: bool) -> None:
    profiles = CustomerProfileRepository(settings.profile_db_file)
    with build_rag(settings) as rag:
        # 此实验专门验证 RAG：不混入只认识 ORDER-1001 的旧固定订单查询工具。
        tools = [
            build_rag_search_tool(rag),
            build_skill_loader_tool(),
            *build_customer_profile_tools(profiles),
        ]
        with build_customer_service_agent(
            settings, tools=tools, profile_repository=profiles
        ) as agent:
            response = agent.ask(message, session_id=session, user_id=user, include_trace=trace)
            print(response.answer)
            if trace:
                for step in response.intermediate_steps:
                    print(f"\n[工具] {step.tool}\n{step.tool_input}\n{step.observation}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("ingest", help="备份旧索引后增量导入48份模拟资料")
    search = commands.add_parser("search", help="查看关键词与真实语义检索候选")
    search.add_argument("query")
    chat = commands.add_parser("ask", help="通过真实模型测试RAG与跨会话长期记忆")
    chat.add_argument("message")
    chat.add_argument("--user", default="lab-user-manual")
    chat.add_argument("--session", default=None)
    chat.add_argument("--trace", action="store_true")
    profile = commands.add_parser("profile", help="查看指定测试用户已保存的长期画像")
    profile.add_argument("--user", default="lab-user-manual")
    args = parser.parse_args()
    settings = lab_settings()
    if args.command == "ingest":
        print(json.dumps(ingest(settings), ensure_ascii=False, indent=2))
    elif args.command == "search":
        with build_rag(settings) as rag:
            result = rag.retrieve(args.query)
            print(result.model_dump_json(indent=2))
    elif args.command == "profile":
        repository = CustomerProfileRepository(settings.profile_db_file)
        saved = repository.get(args.user)
        print(saved.model_dump_json(indent=2) if saved else "该用户尚无长期画像")
    else:
        session = args.session or f"lab-{uuid.uuid4().hex[:12]}"
        print(f"用户：{args.user}；会话：{session}")
        ask(settings, args.message, args.user, session, args.trace)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # 不直接输出 SDK 异常里的完整请求、供应商URL或凭据。
        print(f"操作失败：{type(exc).__name__}。请检查模型服务、网络和本地配置。")
        raise SystemExit(1) from None

"""把 RAG 检索封装成 LangChain Tool。"""

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from customer_service_agent.rag.service import RAGService


class RAGSearchInput(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    top_k: int = Field(default=4, ge=1, le=10, description="返回多少条最相关知识片段")


def build_rag_search_tool(rag_service: RAGService) -> StructuredTool:
    def search_knowledge_base(query: str, top_k: int = 4) -> str:
        """检索知识库，返回适合放进回答里的文本。"""

        hits = rag_service.search(query, top_k=top_k)
        if not hits:
            return "知识库没有检索到相关内容。"

        lines = []
        for index, hit in enumerate(hits, start=1):
            source = hit.metadata.get("filename") or hit.metadata.get("source") or "unknown"
            lines.append(f"[{index}] 相似度 {hit.score:.3f}，来源 {source}\n{hit.content}")
        return "\n\n".join(lines)  # 把line中所有结果join成一行，然后输出一个换一行

    return StructuredTool.from_function(
        func=search_knowledge_base,
        name="search_knowledge_base",
        description="用 RAG 检索企业知识库，适合回答 FAQ、政策细节、产品说明。",
        args_schema=RAGSearchInput,
    )

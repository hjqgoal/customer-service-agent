"""Load and split local RAG source documents."""

from collections import defaultdict
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED_EXTENSIONS = {".md", ".txt", ".rst", ".csv"}


def load_documents(directory: str | Path) -> list[Document]:
    base_dir = Path(directory).resolve()
    if not base_dir.is_dir():
        raise FileNotFoundError(f"知识库目录不存在：{base_dir}")

    documents: list[Document] = []
    for path in sorted(base_dir.rglob("*")):
        if (
            not path.is_file()
            or path.is_symlink()
            or path.name.startswith(".")
            or path.suffix.lower() not in SUPPORTED_EXTENSIONS
        ):
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": path.relative_to(base_dir).as_posix(),
                    "filename": path.name,
                    "extension": path.suffix.lower(),
                },
            )
        )
    return documents


def split_documents(
    documents: list[Document],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Split by character count, not token count."""

    if chunk_overlap >= chunk_size:
        raise ValueError("rag_chunk_overlap 必须小于 rag_chunk_size。")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    counters: defaultdict[str, int] = defaultdict(int)
    for chunk in chunks:
        source = str(chunk.metadata.get("source") or "unknown")
        chunk.metadata = {**chunk.metadata, "chunk_index": counters[source]}
        counters[source] += 1
    return chunks

from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import ChunkingConfig


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, docs: List[Document]) -> List[Document]:
        ...


class RecursiveChunker(BaseChunker):
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def chunk(self, docs: List[Document]) -> List[Document]:
        return self.splitter.split_documents(docs)


class StructuredChunker(BaseChunker):
    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size

    def chunk(self, docs: List[Document]) -> List[Document]:
        from langchain_text_splitters import MarkdownHeaderTextSplitter

        headers = [("#", "title"), ("##", "section"), ("###", "subsection")]
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)

        result = []
        for doc in docs:
            splits = splitter.split_text(doc.page_content)
            for split in splits:
                metadata = {**doc.metadata, **split.metadata}
                # Reconstruct heading context for the chunk
                heading_parts = []
                if "title" in split.metadata:
                    heading_parts.append(f"# {split.metadata['title']}")
                if "section" in split.metadata:
                    heading_parts.append(f"## {split.metadata['section']}")
                if "subsection" in split.metadata:
                    heading_parts.append(f"### {split.metadata['subsection']}")
                heading_prefix = "\n".join(heading_parts) + "\n" if heading_parts else ""
                full_content = heading_prefix + split.page_content

                if len(full_content) > self.chunk_size:
                    sub_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=self.chunk_size, chunk_overlap=64
                    )
                    sub_chunks = sub_splitter.split_text(full_content)
                    for chunk in sub_chunks:
                        result.append(Document(page_content=chunk, metadata=metadata))
                else:
                    result.append(Document(page_content=full_content, metadata=metadata))

        return result if result else docs


class CodeChunker(BaseChunker):
    def chunk(self, docs: List[Document]) -> List[Document]:
        result = []
        for doc in docs:
            if doc.metadata.get("chunk_type") != "code":
                result.append(doc)
            else:
                result.extend(self._split_code(doc))
        return result

    def _split_code(self, doc: Document) -> List[Document]:
        import ast
        try:
            tree = ast.parse(doc.page_content)
        except SyntaxError:
            return [doc]

        lines = doc.page_content.split("\n")
        chunks = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = node.lineno - 1
                end = node.end_lineno if hasattr(node, "end_lineno") and node.end_lineno else start + 1
                chunk_content = "\n".join(lines[start:end])
                metadata = {**doc.metadata, "chunk_type": "code"}
                chunks.append(Document(page_content=chunk_content, metadata=metadata))

        return chunks if chunks else [doc]


class ChunkerFactory:
    @staticmethod
    def get_chunker(doc_type: str, config: ChunkingConfig = None) -> BaseChunker:
        if config is None:
            config = ChunkingConfig()

        if doc_type == "code":
            return CodeChunker()
        elif doc_type in ("technical_doc", "api_ref"):
            return StructuredChunker(chunk_size=config.chunk_size)
        else:
            return RecursiveChunker(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)

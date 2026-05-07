import os
from typing import List, Type
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from .base import DataLoader


class MarkdownLoader(DataLoader):
    def load(self) -> List[Document]:
        with open(self.file_path, "r", encoding="utf-8") as f:
            content = f.read()

        headers_to_split_on = [
            ("#", "title"),
            ("##", "section"),
            ("###", "subsection"),
        ]
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        splits = splitter.split_text(content)

        docs = []
        for i, split in enumerate(splits):
            metadata = {
                "source": self.file_path,
                "doc_type": "technical_doc",
                "chunk_type": "text",
                "title": split.metadata.get("title", ""),
                "section": split.metadata.get("section", ""),
                "has_code": "```" in split.page_content,
            }
            docs.append(Document(page_content=split.page_content, metadata=metadata))

        if not docs:
            docs.append(self._make_doc(content, title=os.path.basename(self.file_path)))

        return docs


class TextLoader(DataLoader):
    def load(self) -> List[Document]:
        with open(self.file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return [self._make_doc(content, doc_type="text", title=os.path.basename(self.file_path))]


class PDFLoader(DataLoader):
    def load(self) -> List[Document]:
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("pypdf is required for PDF loading: pip install pypdf")

        reader = PdfReader(self.file_path)
        docs = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                docs.append(self._make_doc(
                    text,
                    doc_type="technical_doc",
                    title=os.path.basename(self.file_path),
                    page_number=i + 1,
                ))
        return docs


class CodeLoader(DataLoader):
    def load(self) -> List[Document]:
        with open(self.file_path, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = self._split_by_ast(content)
        docs = []
        for chunk_content, chunk_type in chunks:
            docs.append(self._make_doc(
                chunk_content,
                doc_type="code",
                chunk_type="code",
                language="python",
                title=os.path.basename(self.file_path),
                has_code=True,
            ))
        return docs

    def _split_by_ast(self, code: str) -> list:
        import ast
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return [(code, "raw")]

        chunks = []
        lines = code.split("\n")

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = node.lineno - 1
                end = node.end_lineno if hasattr(node, "end_lineno") and node.end_lineno else start + 1
                chunk = "\n".join(lines[start:end])
                chunk_type = "function" if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else "class"
                chunks.append((chunk, chunk_type))

        if not chunks:
            chunks.append((code, "raw"))

        return chunks


class StructuredDataLoader(DataLoader):
    def load(self) -> List[Document]:
        ext = os.path.splitext(self.file_path)[1].lower()
        if ext == ".csv":
            return self._load_csv()
        elif ext in (".db", ".sqlite", ".sqlite3"):
            return self._load_sqlite()
        else:
            raise ValueError(f"Unsupported structured data format: {ext}")

    def _load_csv(self) -> List[Document]:
        import pandas as pd
        df = pd.read_csv(self.file_path)
        table_name = Path(self.file_path).stem

        header_desc = ", ".join([f"{col} ({df[col].dtype})" for col in df.columns])
        summary = f"表名: {table_name}\n字段: {header_desc}\n行数: {len(df)}"

        docs = [self._make_doc(summary, doc_type="structured", chunk_type="table", title=table_name)]

        for _, row in df.head(100).iterrows():
            row_text = " | ".join([f"{col}: {row[col]}" for col in df.columns])
            docs.append(self._make_doc(row_text, doc_type="structured", chunk_type="table_row", title=table_name))

        return docs

    def _load_sqlite(self) -> List[Document]:
        import sqlite3
        conn = sqlite3.connect(self.file_path)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()

        docs = []
        for (table_name,) in tables:
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            col_desc = ", ".join([f"{col[1]} ({col[2]})" for col in columns])

            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]

            summary = f"表名: {table_name}\n字段: {col_desc}\n行数: {row_count}"
            docs.append(self._make_doc(summary, doc_type="structured", chunk_type="table", title=table_name))

        conn.close()
        return docs


class ImageLoader(DataLoader):
    def __init__(self, file_path: str, description: str = "", context: str = ""):
        super().__init__(file_path)
        self.description = description
        self.context = context

    def load(self) -> List[Document]:
        desc = self.description or os.path.basename(self.file_path)
        content = f"[图片: {desc}]"
        if self.context:
            content += f"\n上下文: {self.context}"

        return [self._make_doc(
            content,
            doc_type="image",
            chunk_type="image",
            image_path=self.file_path,
            description=desc,
        )]


class LoaderFactory:
    _mapping = {
        ".md": MarkdownLoader,
        ".markdown": MarkdownLoader,
        ".txt": TextLoader,
        ".pdf": PDFLoader,
        ".py": CodeLoader,
        ".csv": StructuredDataLoader,
        ".db": StructuredDataLoader,
        ".sqlite": StructuredDataLoader,
        ".png": ImageLoader,
        ".jpg": ImageLoader,
        ".jpeg": ImageLoader,
        ".webp": ImageLoader,
    }

    @classmethod
    def get_loader_class(cls, file_path: str) -> Type[DataLoader]:
        ext = os.path.splitext(file_path)[1].lower()
        loader_class = cls._mapping.get(ext)
        if not loader_class:
            raise ValueError(f"Unsupported file format: {ext}")
        return loader_class

    @classmethod
    def load_file(cls, file_path: str, **kwargs) -> List[Document]:
        loader_class = cls.get_loader_class(file_path)
        return loader_class(file_path, **kwargs).load()

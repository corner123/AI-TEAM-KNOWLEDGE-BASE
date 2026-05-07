import pytest
import tempfile
import os


def test_markdown_loader(tmp_path):
    md_content = """# 标题

## 第一节
这是第一节的内容。

## 第二节
这是第二节的内容。
"""
    md_file = tmp_path / "test.md"
    md_file.write_text(md_content, encoding="utf-8")

    from rag_core.data.loaders import MarkdownLoader
    loader = MarkdownLoader(str(md_file))
    docs = loader.load()

    assert len(docs) > 0
    assert any("第一节" in doc.page_content for doc in docs)
    assert any("第二节" in doc.page_content for doc in docs)
    assert all(doc.metadata["source"] == str(md_file) for doc in docs)
    assert all(doc.metadata["doc_type"] == "technical_doc" for doc in docs)


def test_text_loader(tmp_path):
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("这是一个测试文本文件。\n包含多行内容。", encoding="utf-8")

    from rag_core.data.loaders import TextLoader
    loader = TextLoader(str(txt_file))
    docs = loader.load()

    assert len(docs) == 1
    assert "测试文本" in docs[0].page_content


def test_csv_loader(tmp_path):
    csv_content = "name,params,accuracy\nBERT,110M,0.89\nGPT-3,175B,0.92"
    csv_file = tmp_path / "models.csv"
    csv_file.write_text(csv_content, encoding="utf-8")

    from rag_core.data.loaders import StructuredDataLoader
    loader = StructuredDataLoader(str(csv_file))
    docs = loader.load()

    assert len(docs) > 0
    assert any("BERT" in doc.page_content for doc in docs)


def test_python_code_loader(tmp_path):
    py_content = '''def hello():
    """Say hello."""
    print("hello world")

class MyClass:
    def method(self):
        return 42
'''
    py_file = tmp_path / "example.py"
    py_file.write_text(py_content, encoding="utf-8")

    from rag_core.data.loaders import CodeLoader
    loader = CodeLoader(str(py_file))
    docs = loader.load()

    assert len(docs) > 0
    assert any("hello" in doc.page_content for doc in docs)
    assert any("MyClass" in doc.page_content for doc in docs)


def test_image_loader(tmp_path):
    img_file = tmp_path / "arch.png"
    img_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    from rag_core.data.loaders import ImageLoader
    loader = ImageLoader(str(img_file), description="系统架构图")
    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].metadata["chunk_type"] == "image"
    assert "架构图" in docs[0].page_content


def test_loader_factory():
    from rag_core.data.loaders import LoaderFactory

    assert LoaderFactory.get_loader_class("test.md").__name__ == "MarkdownLoader"
    assert LoaderFactory.get_loader_class("test.pdf").__name__ == "PDFLoader"
    assert LoaderFactory.get_loader_class("test.py").__name__ == "CodeLoader"
    assert LoaderFactory.get_loader_class("test.csv").__name__ == "StructuredDataLoader"
    assert LoaderFactory.get_loader_class("test.png").__name__ == "ImageLoader"

    with pytest.raises(ValueError):
        LoaderFactory.get_loader_class("test.xyz")

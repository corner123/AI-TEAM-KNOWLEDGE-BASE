import pytest
from langchain_core.documents import Document


def test_recursive_chunker():
    from rag_core.data.chunkers import RecursiveChunker
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)

    doc = Document(page_content="A" * 250, metadata={"source": "test.md"})
    chunks = chunker.chunk([doc])

    assert len(chunks) > 1
    assert all(len(c.page_content) <= 120 for c in chunks)
    assert all(c.metadata["source"] == "test.md" for c in chunks)


def test_structured_chunker():
    from rag_core.data.chunkers import StructuredChunker
    chunker = StructuredChunker()

    doc = Document(
        page_content="# Title\n\n## Section 1\nContent 1\n\n## Section 2\nContent 2",
        metadata={"source": "test.md", "title": "Title"},
    )
    chunks = chunker.chunk([doc])

    assert len(chunks) >= 2
    assert any("Section 1" in c.page_content for c in chunks)
    assert any("Section 2" in c.page_content for c in chunks)


def test_code_chunker():
    from rag_core.data.chunkers import CodeChunker
    chunker = CodeChunker()

    code = '''def func_a():
    return 1

def func_b():
    return 2

class MyClass:
    def method(self):
        return 3
'''
    doc = Document(page_content=code, metadata={"source": "test.py", "chunk_type": "code"})
    chunks = chunker.chunk([doc])

    assert len(chunks) >= 2
    assert any("func_a" in c.page_content for c in chunks)
    assert any("func_b" in c.page_content or "MyClass" in c.page_content for c in chunks)


def test_chunker_preserves_metadata():
    from rag_core.data.chunkers import RecursiveChunker
    chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)

    doc = Document(
        page_content="Short text " * 20,
        metadata={"source": "test.md", "doc_type": "technical_doc", "title": "Test"},
    )
    chunks = chunker.chunk([doc])

    for chunk in chunks:
        assert chunk.metadata["source"] == "test.md"
        assert chunk.metadata["doc_type"] == "technical_doc"
        assert chunk.metadata["title"] == "Test"

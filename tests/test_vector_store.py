import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


def test_vector_store_init():
    from rag_core.index.vector_store import MilvusVectorStore
    store = MilvusVectorStore.__new__(MilvusVectorStore)
    store.config = MagicMock()
    store.embedding_manager = MagicMock()
    store._collection = None
    assert store is not None


def test_rrf_fusion():
    from rag_core.index.vector_store import MilvusVectorStore
    dense_results = [
        {"id": 1, "score": 0.9, "text": "doc1"},
        {"id": 2, "score": 0.8, "text": "doc2"},
        {"id": 3, "score": 0.7, "text": "doc3"},
    ]
    sparse_results = [
        {"id": 2, "score": 0.85, "text": "doc2"},
        {"id": 1, "score": 0.75, "text": "doc1"},
        {"id": 4, "score": 0.6, "text": "doc4"},
    ]
    fused = MilvusVectorStore.rrf_fusion(dense_results, sparse_results, k=60)
    assert len(fused) == 4
    assert fused[0]["id"] in [1, 2]  # top result should be from overlapping docs


def test_build_filter_expr():
    from rag_core.index.vector_store import MilvusVectorStore
    expr = MilvusVectorStore.build_filter_expr({"doc_type": "api_ref", "has_code": True})
    assert "api_ref" in expr
    assert "true" in expr.lower()

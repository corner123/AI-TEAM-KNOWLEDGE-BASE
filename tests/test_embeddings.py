import pytest
import numpy as np


def test_embedding_manager_init():
    from rag_core.index.embeddings import EmbeddingManager
    manager = EmbeddingManager()
    assert manager is not None


def test_dense_embedding_dimensions():
    from rag_core.index.embeddings import EmbeddingManager
    manager = EmbeddingManager()
    vec = manager.embed_query("测试文本")
    assert isinstance(vec, list)
    assert len(vec) == 1024  # BGE-large 维度


def test_batch_embedding():
    from rag_core.index.embeddings import EmbeddingManager
    manager = EmbeddingManager()
    texts = ["文本1", "文本2", "文本3"]
    vecs = manager.embed_documents(texts)
    assert len(vecs) == 3
    assert all(len(v) == 1024 for v in vecs)

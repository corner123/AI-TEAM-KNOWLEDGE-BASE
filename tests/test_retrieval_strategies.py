import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


def test_retrieval_strategy_interface():
    from rag_core.retrieval.base import RetrievalStrategy
    assert hasattr(RetrievalStrategy, "retrieve")
    assert hasattr(RetrievalStrategy, "get_strategy_name")


def test_hybrid_search_strategy_name():
    from rag_core.retrieval.hybrid_search import HybridSearchStrategy
    vector_store = MagicMock()
    strategy = HybridSearchStrategy(vector_store=vector_store)
    assert strategy.get_strategy_name() == "HybridSearch"


def test_hybrid_search_calls_vector_store():
    from rag_core.retrieval.hybrid_search import HybridSearchStrategy
    vector_store = MagicMock()
    vector_store.hybrid_search.return_value = [
        {"id": 1, "score": 0.9, "text": "test content", "source": "test.md", "doc_type": "technical_doc"},
    ]
    strategy = HybridSearchStrategy(vector_store=vector_store)
    results = strategy.retrieve("test query", top_k=5)

    vector_store.hybrid_search.assert_called_once()
    assert len(results) == 1
    assert isinstance(results[0], Document)

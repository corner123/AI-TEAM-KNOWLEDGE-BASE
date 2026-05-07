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


def test_text2sql_strategy_name():
    from rag_core.retrieval.text2sql import Text2SQLStrategy
    llm = MagicMock()
    strategy = Text2SQLStrategy(llm=llm, db_path=":memory:")
    assert strategy.get_strategy_name() == "Text2SQL"


def test_text2sql_rejects_dangerous_sql():
    from rag_core.retrieval.text2sql import Text2SQLStrategy
    strategy = Text2SQLStrategy.__new__(Text2SQLStrategy)
    assert strategy._is_safe_sql("SELECT * FROM models") is True
    assert strategy._is_safe_sql("DROP TABLE models") is False
    assert strategy._is_safe_sql("DELETE FROM models") is False
    assert strategy._is_safe_sql("INSERT INTO models VALUES (1)") is False


def test_query_rewrite_strategy_name():
    from rag_core.retrieval.query_rewrite import QueryRewriteStrategy
    llm = MagicMock()
    vector_store = MagicMock()
    strategy = QueryRewriteStrategy(llm=llm, vector_store=vector_store)
    assert strategy.get_strategy_name() == "QueryRewrite"


def test_query_rewrite_expands_query():
    from rag_core.retrieval.query_rewrite import QueryRewriteStrategy
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="扩展查询1\n扩展查询2\n扩展查询3")
    vector_store = MagicMock()
    vector_store.search_dense.return_value = []

    strategy = QueryRewriteStrategy(llm=llm, vector_store=vector_store)
    strategy.retrieve("原始查询", top_k=5)

    assert vector_store.search_dense.call_count >= 1

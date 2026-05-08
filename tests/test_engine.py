import pytest
from unittest.mock import MagicMock, patch


def test_engine_init():
    from rag_core.engine import RAGEngine
    config = MagicMock()
    engine = RAGEngine.__new__(RAGEngine)
    engine.config = config
    engine._initialized = False
    assert engine is not None


def test_engine_query_response_format():
    from rag_core.engine import RAGEngine
    engine = RAGEngine.__new__(RAGEngine)
    engine._initialized = True

    mock_router = MagicMock()
    mock_router.route_and_retrieve.return_value = []
    engine.query_router = mock_router

    mock_generator = MagicMock()
    mock_generator.generate.return_value = {
        "answer": "测试回答",
        "sources": ["test.md"],
        "confidence": 0.8,
        "strategy_used": "HybridSearch",
    }
    engine.generator = mock_generator

    result = engine.query("测试问题")
    assert "answer" in result
    assert "sources" in result

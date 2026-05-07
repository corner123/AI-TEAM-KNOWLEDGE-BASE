import pytest
from unittest.mock import MagicMock


def test_query_router_init():
    from rag_core.query_router import QueryRouter
    strategies = {"HybridSearch": MagicMock(), "Text2SQL": MagicMock()}
    router = QueryRouter(strategies=strategies, llm=MagicMock())
    assert router is not None


def test_rule_based_route_text2sql():
    from rag_core.query_router import QueryRouter
    strategies = {"HybridSearch": MagicMock(), "Text2SQL": MagicMock()}
    router = QueryRouter(strategies=strategies, llm=MagicMock())

    assert router._rule_based_route("BERT的参数量是多少") == "Text2SQL"
    assert router._rule_based_route("对比GPT和BERT的准确率") == "Text2SQL"
    assert router._rule_based_route("查询所有模型的排名") == "Text2SQL"


def test_rule_based_route_metadata():
    from rag_core.query_router import QueryRouter
    strategies = {"HybridSearch": MagicMock(), "MetadataFilter": MagicMock()}
    router = QueryRouter(strategies=strategies, llm=MagicMock())

    assert router._rule_based_route("LangChain的API文档") == "MetadataFilter"
    assert router._rule_based_route("如何使用Milvus") == "MetadataFilter"


def test_rule_based_route_default():
    from rag_core.query_router import QueryRouter
    strategies = {"HybridSearch": MagicMock()}
    router = QueryRouter(strategies=strategies, llm=MagicMock())

    assert router._rule_based_route("什么是RAG") is None


def test_fallback_on_empty_results():
    from rag_core.query_router import QueryRouter
    primary = MagicMock()
    primary.retrieve.return_value = []
    primary.get_strategy_name.return_value = "Text2SQL"

    fallback = MagicMock()
    fallback.retrieve.return_value = [MagicMock()]
    fallback.get_strategy_name.return_value = "HybridSearch"

    strategies = {"Text2SQL": primary, "HybridSearch": fallback}
    router = QueryRouter(strategies=strategies, llm=MagicMock())

    result = router.route_and_retrieve("BERT参数量")
    assert len(result) > 0

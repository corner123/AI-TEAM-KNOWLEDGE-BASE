import re
from typing import List, Dict, Optional
from langchain_core.documents import Document
from langchain_core.language_models import BaseLLM
from rag_core.retrieval.base import RetrievalStrategy
from utils.logger import get_logger

logger = get_logger("query_router")

ROUTE_PROMPT = """分析用户问题，选择最合适的检索策略。

可用策略:
- HybridSearch: 通用文档检索
- Text2SQL: 结构化数据查询（参数、对比、统计）
- QueryRewrite: 模糊问题、需要扩展
- MetadataFilter: 限定范围的精确查询
- MultimodalSearch: 图片相关查询

用户问题: {query}

只返回策略名称:"""


class QueryRouter:
    def __init__(self, strategies: Dict[str, RetrievalStrategy], llm: BaseLLM):
        self.strategies = strategies
        self.llm = llm
        self._default_strategy = "HybridSearch"

    def route(self, query: str) -> RetrievalStrategy:
        strategy_name = self._rule_based_route(query)
        if strategy_name is None:
            strategy_name = self._llm_route(query)

        strategy = self.strategies.get(strategy_name)
        if strategy is None:
            strategy = self.strategies.get(self._default_strategy)

        logger.info(f"Routed to: {strategy.get_strategy_name()}")
        return strategy

    def route_and_retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Document]:
        primary = self.route(query)
        results = primary.retrieve(query, top_k=top_k, **kwargs)

        if not results and primary.get_strategy_name() != self._default_strategy:
            logger.info(f"No results from {primary.get_strategy_name()}, falling back to {self._default_strategy}")
            fallback = self.strategies.get(self._default_strategy)
            if fallback:
                results = fallback.retrieve(query, top_k=top_k, **kwargs)

        return results

    def _rule_based_route(self, query: str) -> Optional[str]:
        sql_patterns = ["多少", "对比", "排名", "查询", "统计", "最大", "最小", "平均", "几个"]
        if any(p in query for p in sql_patterns):
            return "Text2SQL"

        meta_patterns = ["文档", "API", "教程", "如何", "怎么", "配置"]
        if any(p in query for p in meta_patterns):
            return "MetadataFilter"

        image_patterns = ["图", "架构", "流程", "截图", "图片"]
        if any(p in query for p in image_patterns):
            return "MultimodalSearch"

        return None

    def _llm_route(self, query: str) -> str:
        try:
            prompt = ROUTE_PROMPT.format(query=query)
            response = self.llm.invoke(prompt)
            strategy_name = response.content.strip()
            if strategy_name in self.strategies:
                return strategy_name
        except Exception as e:
            logger.warning(f"LLM routing failed: {e}")
        return self._default_strategy

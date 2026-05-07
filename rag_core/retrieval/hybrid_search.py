from typing import List, Optional, Dict
from langchain_core.documents import Document
from .base import RetrievalStrategy
from rag_core.index.vector_store import MilvusVectorStore


class HybridSearchStrategy(RetrievalStrategy):
    def __init__(self, vector_store: MilvusVectorStore, use_rerank: bool = False):
        self.vector_store = vector_store
        self.use_rerank = use_rerank

    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Document]:
        filter_expr = kwargs.get("filter_expr", "")
        results = self.vector_store.hybrid_search(query, top_k=top_k, filter_expr=filter_expr)
        docs = self._to_documents(results)

        if self.use_rerank and docs:
            docs = self._rerank(query, docs)

        return docs[:top_k]

    def get_strategy_name(self) -> str:
        return "HybridSearch"

    def _rerank(self, query: str, docs: List[Document]) -> List[Document]:
        try:
            from rag_core.retrieval.reranker import RerankerStrategy
            reranker = RerankerStrategy()
            return reranker.rerank(query, docs)
        except Exception:
            return docs

from typing import List
from langchain_core.documents import Document
from langchain_core.language_models import BaseLLM
from .base import RetrievalStrategy
from rag_core.index.vector_store import MilvusVectorStore
from utils.logger import get_logger

logger = get_logger("query_rewrite")

REWRITE_PROMPT = """你是一个查询优化专家。将用户的原始查询扩展为3个不同角度的子查询，以便更全面地检索相关信息。

原始查询: {query}

请生成3个扩展查询，每行一个:"""


class QueryRewriteStrategy(RetrievalStrategy):
    def __init__(self, llm: BaseLLM, vector_store: MilvusVectorStore):
        self.llm = llm
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Document]:
        expanded_queries = self._expand_query(query)
        all_results = []

        for q in [query] + expanded_queries:
            results = self.vector_store.search_dense(q, top_k=top_k)
            all_results.extend(results)

        return self._deduplicate(all_results)[:top_k]

    def get_strategy_name(self) -> str:
        return "QueryRewrite"

    def _expand_query(self, query: str) -> List[str]:
        try:
            prompt = REWRITE_PROMPT.format(query=query)
            response = self.llm.invoke(prompt)
            queries = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
            return queries[:3]
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return []

    def _deduplicate(self, results: list) -> List[Document]:
        seen_texts = set()
        docs = []
        for r in results:
            text = r.get("text", "")
            if text not in seen_texts:
                seen_texts.add(text)
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "source": r.get("source", ""),
                        "doc_type": r.get("doc_type", ""),
                        "title": r.get("title", ""),
                        "score": r.get("score", 0),
                    },
                ))
        return docs

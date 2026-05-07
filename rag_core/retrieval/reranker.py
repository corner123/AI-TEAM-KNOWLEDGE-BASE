from typing import List, Optional
from langchain_core.documents import Document
from .base import RetrievalStrategy
from utils.logger import get_logger

logger = get_logger("reranker")


class RerankerStrategy(RetrievalStrategy):
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model_name = model_name
        self._model = None

    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Document]:
        docs = kwargs.get("docs", [])
        if not docs:
            logger.warning("No documents provided for reranking")
            return []
        reranked = self.rerank(query, docs)
        return reranked[:top_k]

    def get_strategy_name(self) -> str:
        return "Rerank"

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading CrossEncoder model: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, docs: List[Document]) -> List[Document]:
        try:
            model = self._get_model()
            pairs = [(query, doc.page_content) for doc in docs]
            scores = model.predict(pairs)
            scored_docs = list(zip(scores, docs))
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            result = []
            for score, doc in scored_docs:
                doc.metadata["rerank_score"] = float(score)
                result.append(doc)
            return result
        except Exception as e:
            logger.warning(f"Reranking failed, returning original order: {e}")
            return docs

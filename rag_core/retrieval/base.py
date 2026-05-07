from abc import ABC, abstractmethod
from typing import List, Optional
from langchain_core.documents import Document


class RetrievalStrategy(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Document]:
        ...

    @abstractmethod
    def get_strategy_name(self) -> str:
        ...

    def _to_documents(self, results: list) -> List[Document]:
        docs = []
        for r in results:
            doc = Document(
                page_content=r.get("text", ""),
                metadata={
                    "source": r.get("source", ""),
                    "doc_type": r.get("doc_type", ""),
                    "chunk_type": r.get("chunk_type", ""),
                    "title": r.get("title", ""),
                    "section": r.get("section", ""),
                    "score": r.get("score", 0),
                },
            )
            docs.append(doc)
        return docs

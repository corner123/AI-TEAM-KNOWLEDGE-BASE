from typing import List, Optional
from langchain_core.documents import Document
from .base import RetrievalStrategy
from utils.logger import get_logger

logger = get_logger("multimodal_search")


class MultimodalSearchStrategy(RetrievalStrategy):
    def __init__(self, multimodal_collection=None, embedding_manager=None):
        self.multimodal_collection = multimodal_collection
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Document]:
        logger.info("MultimodalSearch placeholder: returning empty results")
        return []

    def get_strategy_name(self) -> str:
        return "MultimodalSearch"

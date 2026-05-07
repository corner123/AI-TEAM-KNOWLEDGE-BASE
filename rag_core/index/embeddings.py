from typing import List, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from config import EmbeddingConfig


class EmbeddingManager:
    def __init__(self, config: EmbeddingConfig = None):
        if config is None:
            config = EmbeddingConfig()
        self.config = config
        self._dense_model: Optional[HuggingFaceEmbeddings] = None

    @property
    def dense_model(self) -> HuggingFaceEmbeddings:
        if self._dense_model is None:
            self._dense_model = HuggingFaceEmbeddings(
                model_name=self.config.model_path,
                model_kwargs={"device": self.config.device},
                encode_kwargs={"batch_size": self.config.batch_size, "normalize_embeddings": True},
            )
        return self._dense_model

    def embed_query(self, text: str) -> List[float]:
        return self.dense_model.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.dense_model.embed_documents(texts)

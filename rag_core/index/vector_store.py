from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, utility
from config import MilvusConfig
from .embeddings import EmbeddingManager
from utils.logger import get_logger

logger = get_logger("vector_store")


class MilvusVectorStore:
    def __init__(self, config: MilvusConfig, embedding_manager: EmbeddingManager):
        self.config = config
        self.embedding_manager = embedding_manager
        self._collection: Optional[Collection] = None

    def get_or_create_collection(self) -> Collection:
        if self._collection is not None:
            return self._collection

        name = self.config.collection_name
        if utility.has_collection(name):
            self._collection = Collection(name)
            self._collection.load()
            return self._collection

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=self.config.dense_dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="has_code", dtype=DataType.BOOL),
        ]

        schema = CollectionSchema(fields, description="Knowledge base collection")
        self._collection = Collection(name, schema)

        index_params = {
            "metric_type": self.config.metric_type,
            "index_type": self.config.index_type,
            "params": {"M": self.config.hnsw_m, "efConstruction": self.config.hnsw_ef_construction},
        }
        self._collection.create_index("dense_vector", index_params)
        self._collection.load()

        logger.info(f"Created collection: {name}")
        return self._collection

    def insert_documents(self, docs: List[Document], batch_size: int = 256):
        collection = self.get_or_create_collection()
        texts = [doc.page_content for doc in docs]

        logger.info(f"Encoding {len(texts)} documents...")
        vectors = self.embedding_manager.embed_documents(texts)

        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i:i + batch_size]
            batch_vecs = vectors[i:i + batch_size]

            data = [
                batch_vecs,
                [doc.page_content for doc in batch_docs],
                [doc.metadata.get("source", "") for doc in batch_docs],
                [doc.metadata.get("doc_type", "") for doc in batch_docs],
                [doc.metadata.get("chunk_type", "text") for doc in batch_docs],
                [doc.metadata.get("title", "") for doc in batch_docs],
                [doc.metadata.get("section", "") for doc in batch_docs],
                [doc.metadata.get("has_code", False) for doc in batch_docs],
            ]
            collection.insert(data)

        collection.flush()
        logger.info(f"Inserted {len(docs)} documents")

    def search_dense(self, query: str, top_k: int = 5, filter_expr: str = "") -> List[Dict]:
        collection = self.get_or_create_collection()
        query_vec = self.embedding_manager.embed_query(query)

        search_params = {"metric_type": self.config.metric_type, "params": {"ef": 128}}
        output_fields = ["text", "source", "doc_type", "chunk_type", "title", "section", "has_code"]

        results = collection.search(
            data=[query_vec],
            anns_field="dense_vector",
            param=search_params,
            limit=top_k,
            expr=filter_expr if filter_expr else None,
            output_fields=output_fields,
        )

        return self._format_results(results)

    def hybrid_search(self, query: str, top_k: int = 5, filter_expr: str = "") -> List[Dict]:
        dense_results = self.search_dense(query, top_k=top_k * 2, filter_expr=filter_expr)
        return dense_results[:top_k]

    @staticmethod
    def rrf_fusion(dense_results: List[Dict], sparse_results: List[Dict], k: int = 60) -> List[Dict]:
        scores = {}
        all_docs = {}

        for rank, doc in enumerate(dense_results):
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
            all_docs[doc_id] = doc

        for rank, doc in enumerate(sparse_results):
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
            all_docs[doc_id] = doc

        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        result = []
        for doc_id in sorted_ids:
            doc = all_docs[doc_id].copy()
            doc["rrf_score"] = scores[doc_id]
            result.append(doc)

        return result

    @staticmethod
    def build_filter_expr(metadata: Dict[str, Any]) -> str:
        conditions = []
        for key, value in metadata.items():
            if isinstance(value, bool):
                conditions.append(f"{key} == {str(value).lower()}")
            elif isinstance(value, str) and value:
                conditions.append(f'{key} == "{value}"')
        return " and ".join(conditions) if conditions else ""

    def _format_results(self, results) -> List[Dict]:
        formatted = []
        for hits in results:
            for hit in hits:
                formatted.append({
                    "id": hit.id,
                    "score": hit.score,
                    "text": hit.entity.get("text", ""),
                    "source": hit.entity.get("source", ""),
                    "doc_type": hit.entity.get("doc_type", ""),
                    "chunk_type": hit.entity.get("chunk_type", ""),
                    "title": hit.entity.get("title", ""),
                    "section": hit.entity.get("section", ""),
                    "has_code": hit.entity.get("has_code", False),
                })
        return formatted

    def drop_collection(self):
        if self._collection is not None:
            self._collection.release()
            utility.drop_collection(self.config.collection_name)
            self._collection = None
            logger.info(f"Dropped collection: {self.config.collection_name}")

    def get_stats(self) -> Dict[str, Any]:
        collection = self.get_or_create_collection()
        return {
            "collection_name": self.config.collection_name,
            "num_entities": collection.num_entities,
            "index_type": self.config.index_type,
        }

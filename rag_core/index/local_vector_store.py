"""Persistent local FAISS store with stable ids and idempotent updates.

The store intentionally keeps the same small interface as ``MilvusVectorStore``
while preserving all document metadata required by the engineering knowledge
pipeline.  FAISS ``IndexFlatIP`` has no native keyed upsert, so changed records
trigger a deterministic in-memory rebuild; new records can still be appended.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from langchain_core.documents import Document

from .embeddings import EmbeddingManager
from utils.logger import get_logger

logger = get_logger("local_vector_store")


class LocalVectorStore:
    SCHEMA_VERSION = 2

    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        persist_dir: str = "data/vector_index",
        dimension: int = 1024,
    ):
        self.embedding_manager = embedding_manager
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._index = None
        self._documents: List[Dict[str, Any]] = []
        self._dim = dimension

    @property
    def index_path(self) -> Path:
        return self.persist_dir / "index.faiss"

    @property
    def documents_path(self) -> Path:
        return self.persist_dir / "documents.json"

    @property
    def metadata_path(self) -> Path:
        return self.persist_dir / "index_meta.json"

    def _ensure_index(self) -> None:
        if self._index is not None:
            return

        import faiss

        if self.index_path.exists() and self.documents_path.exists():
            if not self.metadata_path.is_file():
                raise FileNotFoundError(
                    f"local vector-store metadata is missing: {self.metadata_path}"
                )
            self._index = faiss.read_index(str(self.index_path))
            self._dim = int(self._index.d)
            with self.documents_path.open("r", encoding="utf-8") as handle:
                raw_documents = json.load(handle)
            with self.metadata_path.open("r", encoding="utf-8") as handle:
                stored_metadata = json.load(handle)
            if int(stored_metadata.get("schema_version", -1)) != self.SCHEMA_VERSION:
                raise ValueError("unsupported local vector-store schema")
            if int(stored_metadata.get("dimension", -1)) != self._index.d:
                raise ValueError("FAISS index/dimension metadata mismatch")
            if int(stored_metadata.get("num_entities", -1)) != self._index.ntotal:
                raise ValueError("FAISS index/entity-count metadata mismatch")
            self._documents = [self._normalise_stored_record(item) for item in raw_documents]
            if self._index.ntotal != len(self._documents):
                raise ValueError(
                    "FAISS index/document metadata mismatch: "
                    f"{self._index.ntotal} vectors != {len(self._documents)} records"
                )
            logger.info(f"Loaded existing index with {self._index.ntotal} vectors")
            return

        self._index = faiss.IndexFlatIP(self._dim)
        self._documents = []
        logger.info("Created new FAISS index")

    def insert_documents(self, docs: List[Document], batch_size: int = 256) -> Dict[str, int]:
        """Idempotently insert or update documents by stable ``chunk_id``."""

        self._ensure_index()
        incoming = self._deduplicate_records(self._record_from_document(doc) for doc in docs)
        existing_by_id = {record["chunk_id"]: record for record in self._documents}

        added: List[Dict[str, Any]] = []
        updated = 0
        skipped = 0
        for record in incoming:
            previous = existing_by_id.get(record["chunk_id"])
            if previous is None:
                existing_by_id[record["chunk_id"]] = record
                added.append(record)
            elif previous == record:
                skipped += 1
            else:
                existing_by_id[record["chunk_id"]] = record
                updated += 1

        if updated:
            ordered_ids = [record["chunk_id"] for record in self._documents]
            for record in added:
                ordered_ids.append(record["chunk_id"])
            merged = [existing_by_id[chunk_id] for chunk_id in ordered_ids]
            self._rebuild(merged, batch_size=batch_size)
        elif added:
            self._append_records(added, batch_size=batch_size)

        if updated or added:
            self._save()

        stats = {"added": len(added), "updated": updated, "skipped": skipped}
        logger.info(f"Local index upsert: {stats}")
        return stats

    def replace_documents(self, docs: List[Document], batch_size: int = 256) -> Dict[str, int]:
        """Replace the full index with a deterministic, duplicate-free build."""

        records = self._deduplicate_records(self._record_from_document(doc) for doc in docs)
        self._rebuild(records, batch_size=batch_size)
        self._save()
        return {"indexed": len(records), "duplicates_removed": len(docs) - len(records)}

    def delete_by_chunk_ids(self, chunk_ids: Iterable[str], batch_size: int = 256) -> int:
        self._ensure_index()
        targets = set(chunk_ids)
        retained = [record for record in self._documents if record["chunk_id"] not in targets]
        deleted = len(self._documents) - len(retained)
        if deleted:
            self._rebuild(retained, batch_size=batch_size)
            self._save()
        return deleted

    def search_dense(self, query: str, top_k: int = 5, filter_expr: str = "") -> List[Dict[str, Any]]:
        self._ensure_index()
        if self._index.ntotal == 0:
            return []

        import faiss

        query_vec = self.embedding_manager.embed_query(query)
        query_np = np.asarray([query_vec], dtype="float32")
        faiss.normalize_L2(query_np)

        search_k = self._index.ntotal if filter_expr else min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_np, search_k)
        results: List[Dict[str, Any]] = []
        for score, index_id in zip(scores[0], indices[0]):
            if index_id < 0 or index_id >= len(self._documents):
                continue
            record = self._documents[int(index_id)]
            result = self._result_from_record(record, int(index_id), float(score))
            if filter_expr and not self._match_filter(result, filter_expr):
                continue
            results.append(result)
            if len(results) >= top_k:
                break
        return results

    def hybrid_search(self, query: str, top_k: int = 5, filter_expr: str = "") -> List[Dict[str, Any]]:
        """Compatibility method; true BM25+dense fusion lives in engineering retrieval."""

        return self.search_dense(query, top_k=top_k, filter_expr=filter_expr)

    def all_documents(self) -> List[Document]:
        self._ensure_index()
        return [
            Document(page_content=record["text"], metadata=dict(record["metadata"]))
            for record in self._documents
        ]

    def drop_collection(self) -> None:
        import faiss

        self._index = faiss.IndexFlatIP(self._dim)
        self._documents = []
        for path in (self.index_path, self.documents_path, self.metadata_path):
            if path.exists():
                path.unlink()
        logger.info("Dropped local vector store")

    def get_stats(self) -> Dict[str, Any]:
        self._ensure_index()
        corpora = sorted(
            {str(record["metadata"].get("corpus", "")) for record in self._documents}
            - {""}
        )
        return {
            "store_type": "local_faiss",
            "schema_version": self.SCHEMA_VERSION,
            "num_entities": int(self._index.ntotal),
            "unique_chunk_ids": len({record["chunk_id"] for record in self._documents}),
            "dimension": int(self._index.d),
            "corpora": corpora,
            "persist_dir": str(self.persist_dir),
        }

    def _append_records(self, records: List[Dict[str, Any]], batch_size: int) -> None:
        if not records:
            return
        vectors = self._embed_texts([record["text"] for record in records], batch_size)
        if self._index.ntotal == 0 and self._index.d != vectors.shape[1]:
            import faiss

            self._dim = int(vectors.shape[1])
            self._index = faiss.IndexFlatIP(self._dim)
        if vectors.shape[1] != self._index.d:
            raise ValueError(f"Embedding dimension {vectors.shape[1]} does not match index {self._index.d}")
        self._index.add(vectors)
        self._documents.extend(records)

    def _rebuild(self, records: List[Dict[str, Any]], batch_size: int) -> None:
        import faiss

        if records:
            vectors = self._embed_texts([record["text"] for record in records], batch_size)
            self._dim = int(vectors.shape[1])
            self._index = faiss.IndexFlatIP(self._dim)
            self._index.add(vectors)
        else:
            self._index = faiss.IndexFlatIP(self._dim)
        self._documents = list(records)

    def _embed_texts(self, texts: List[str], batch_size: int) -> np.ndarray:
        import faiss

        vectors: List[List[float]] = []
        for start in range(0, len(texts), batch_size):
            vectors.extend(self.embedding_manager.embed_documents(texts[start : start + batch_size]))
        array = np.asarray(vectors, dtype="float32")
        if array.ndim != 2:
            raise ValueError("Embedding manager returned an invalid vector matrix")
        faiss.normalize_L2(array)
        return array

    def _save(self) -> None:
        import faiss

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        temp_index = self.index_path.with_suffix(".faiss.tmp")
        temp_documents = self.documents_path.with_suffix(".json.tmp")
        temp_metadata = self.metadata_path.with_suffix(".json.tmp")

        faiss.write_index(self._index, str(temp_index))
        with temp_documents.open("w", encoding="utf-8") as handle:
            json.dump(self._documents, handle, ensure_ascii=False, indent=2)
        with temp_metadata.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "schema_version": self.SCHEMA_VERSION,
                    "dimension": int(self._index.d),
                    "num_entities": int(self._index.ntotal),
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

        temp_index.replace(self.index_path)
        temp_documents.replace(self.documents_path)
        temp_metadata.replace(self.metadata_path)

    def _record_from_document(self, doc: Document) -> Dict[str, Any]:
        metadata = self._json_safe(dict(doc.metadata))
        chunk_id = str(metadata.get("chunk_id") or self._stable_chunk_id(doc.page_content, metadata))
        metadata["chunk_id"] = chunk_id
        return {"chunk_id": chunk_id, "text": doc.page_content, "metadata": metadata}

    def _normalise_stored_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        if "metadata" in record and "chunk_id" in record:
            return {
                "chunk_id": str(record["chunk_id"]),
                "text": str(record.get("text", "")),
                "metadata": self._json_safe(dict(record.get("metadata", {}))),
            }

        # Migration path for the original flat documents.json format.
        text = str(record.get("text", ""))
        metadata = {key: value for key, value in record.items() if key != "text"}
        chunk_id = str(metadata.get("chunk_id") or self._stable_chunk_id(text, metadata))
        metadata["chunk_id"] = chunk_id
        return {"chunk_id": chunk_id, "text": text, "metadata": self._json_safe(metadata)}

    @staticmethod
    def _deduplicate_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        by_id: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []
        for record in records:
            chunk_id = record["chunk_id"]
            if chunk_id not in by_id:
                order.append(chunk_id)
            by_id[chunk_id] = record
        return [by_id[chunk_id] for chunk_id in order]

    @staticmethod
    def _stable_chunk_id(text: str, metadata: Dict[str, Any]) -> str:
        identity = "|".join(
            str(metadata.get(key, ""))
            for key in ("corpus", "source", "relative_path", "symbol", "section", "chunk_index")
        )
        return hashlib.sha256(f"{identity}\n{text}".encode("utf-8")).hexdigest()

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): LocalVectorStore._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [LocalVectorStore._json_safe(item) for item in value]
        return str(value)

    @staticmethod
    def _result_from_record(record: Dict[str, Any], index_id: int, score: float) -> Dict[str, Any]:
        metadata = dict(record["metadata"])
        result = {
            "id": record["chunk_id"],
            "index_id": index_id,
            "chunk_id": record["chunk_id"],
            "text": record["text"],
            "score": score,
            "metadata": metadata,
        }
        result.update(metadata)
        return result

    @staticmethod
    def _match_filter(doc: Dict[str, Any], filter_expr: str) -> bool:
        try:
            for part in filter_expr.split(" and "):
                if "==" not in part:
                    continue
                key, value = part.split("==", 1)
                key = key.strip()
                expected = value.strip().strip('"').strip("'")
                if key not in doc:
                    return False
                actual = doc[key]
                if isinstance(actual, bool):
                    if actual is not (expected.lower() == "true"):
                        return False
                elif str(actual) != expected:
                    return False
            return True
        except Exception:
            return False

    @staticmethod
    def rrf_fusion(dense_results: List[Dict], sparse_results: List[Dict], k: int = 60) -> List[Dict]:
        scores: Dict[Any, float] = {}
        documents: Dict[Any, Dict] = {}
        for results in (dense_results, sparse_results):
            for rank, document in enumerate(results, start=1):
                document_id = document.get("chunk_id", document.get("id", rank))
                scores[document_id] = scores.get(document_id, 0.0) + 1.0 / (k + rank)
                documents[document_id] = document
        fused = []
        for document_id in sorted(scores, key=scores.get, reverse=True):
            document = documents[document_id].copy()
            document["rrf_score"] = scores[document_id]
            fused.append(document)
        return fused

    @staticmethod
    def build_filter_expr(metadata: Dict[str, Any]) -> str:
        conditions = []
        for key, value in metadata.items():
            if isinstance(value, bool):
                conditions.append(f"{key} == {str(value).lower()}")
            elif isinstance(value, str) and value:
                conditions.append(f'{key} == "{value}"')
        return " and ".join(conditions)

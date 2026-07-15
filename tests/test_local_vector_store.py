from __future__ import annotations

import pytest
from langchain_core.documents import Document

pytest.importorskip("faiss")

from rag_core.index.local_vector_store import LocalVectorStore


class FakeEmbeddingManager:
    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.lower()
        return [
            float(len(text) + 1),
            float(sum(lowered.count(vowel) for vowel in "aeiou") + 1),
            float(sum(ord(char) for char in text) % 101 + 1),
        ]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def _doc(chunk_id: str, text: str, corpus: str = "mini_design") -> Document:
    return Document(
        page_content=text,
        metadata={
            "chunk_id": chunk_id,
            "corpus": corpus,
            "source": "docs/example.md",
            "authority": "internal_design",
            "commit_sha": "abc123",
        },
    )


def test_replace_is_duplicate_free_and_preserves_metadata(tmp_path):
    store = LocalVectorStore(FakeEmbeddingManager(), str(tmp_path), dimension=3)
    result = store.replace_documents([_doc("a", "alpha"), _doc("a", "alpha"), _doc("b", "beta")])

    assert result == {"indexed": 2, "duplicates_removed": 1}
    assert store.get_stats()["num_entities"] == 2
    hit = store.search_dense("alpha", top_k=1)[0]
    assert hit["chunk_id"] in {"a", "b"}
    assert hit["authority"] == "internal_design"
    assert hit["commit_sha"] == "abc123"


def test_upsert_is_idempotent_and_updates_changed_chunk(tmp_path):
    store = LocalVectorStore(FakeEmbeddingManager(), str(tmp_path), dimension=3)
    store.replace_documents([_doc("a", "old text")])

    unchanged = store.insert_documents([_doc("a", "old text")])
    assert unchanged == {"added": 0, "updated": 0, "skipped": 1}
    assert store.get_stats()["num_entities"] == 1

    changed = store.insert_documents([_doc("a", "new text"), _doc("b", "second")])
    assert changed["added"] == 1
    assert changed["updated"] == 1
    assert store.get_stats()["num_entities"] == 2
    assert {doc.page_content for doc in store.all_documents()} == {"new text", "second"}


def test_index_reloads_and_supports_metadata_filter(tmp_path):
    first = LocalVectorStore(FakeEmbeddingManager(), str(tmp_path), dimension=3)
    first.replace_documents([
        _doc("a", "internal", corpus="mini_design"),
        _doc("b", "official", corpus="official_agent_docs"),
    ])

    reloaded = LocalVectorStore(FakeEmbeddingManager(), str(tmp_path), dimension=3)
    results = reloaded.search_dense(
        "official",
        top_k=5,
        filter_expr='corpus == "official_agent_docs"',
    )
    assert len(results) == 1
    assert results[0]["chunk_id"] == "b"

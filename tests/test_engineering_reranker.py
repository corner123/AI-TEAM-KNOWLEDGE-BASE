from __future__ import annotations

import pytest

from rag_core.retrieval.engineering import (
    EngineeringCrossEncoderReranker,
    EngineeringSearchResult,
)


def _result(content: str, score: float) -> EngineeringSearchResult:
    return EngineeringSearchResult(content=content, source=content, score=score)


def test_cross_encoder_reranks_bounded_candidates_and_preserves_first_score() -> None:
    reranker = EngineeringCrossEncoderReranker(
        "fake-reranker",
        scorer=lambda pairs: [0.1 if "first" in text else 0.9 for _, text in pairs],
    )
    ranked = reranker.rerank(
        "query", [_result("first", 0.8), _result("second", 0.3)], top_k=1
    )

    assert [item.content for item in ranked] == ["second"]
    assert ranked[0].metadata["first_stage_score"] == 0.3
    assert ranked[0].metadata["rerank_score"] == 0.9
    assert ranked[0].retriever == "cross_encoder_rerank"


def test_cross_encoder_fail_open_is_explicit() -> None:
    candidates = [_result("one", 1.0), _result("two", 0.5)]
    safe = EngineeringCrossEncoderReranker(
        scorer=lambda pairs: (_ for _ in ()).throw(RuntimeError("offline")),
        fail_open=True,
    )
    assert safe.rerank("q", candidates, top_k=1) == candidates[:1]

    strict = EngineeringCrossEncoderReranker(
        scorer=lambda pairs: (_ for _ in ()).throw(RuntimeError("offline")),
        fail_open=False,
    )
    with pytest.raises(RuntimeError, match="offline"):
        strict.rerank("q", candidates)

"""Optional cross-encoder reranking for a small retrieved candidate set."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from .models import EngineeringSearchResult


PairScorer = Callable[[Sequence[tuple[str, str]]], Sequence[float]]


class EngineeringCrossEncoderReranker:
    """Lazy two-stage reranker with an explicit fail-open policy.

    Dense/BM25 retrieval remains the first-stage candidate generator.  The
    cross encoder only scores that bounded set, so its extra latency is both
    measurable and controllable.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        *,
        scorer: PairScorer | None = None,
        fail_open: bool = True,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self._scorer = scorer
        self.fail_open = bool(fail_open)
        self.max_length = int(max_length)
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        self._model: Any = None

    def rerank(
        self,
        query: str,
        results: Sequence[EngineeringSearchResult],
        *,
        top_k: int | None = None,
    ) -> list[EngineeringSearchResult]:
        candidates = list(results)
        if not candidates or (top_k is not None and top_k <= 0):
            return []
        try:
            scores = list(
                self._score([(query, result.content) for result in candidates])
            )
            if len(scores) != len(candidates):
                raise ValueError("reranker returned a score count mismatch")
        except Exception:
            if not self.fail_open:
                raise
            return candidates[:top_k] if top_k is not None else candidates

        ranked: list[EngineeringSearchResult] = []
        for candidate, score in zip(candidates, scores):
            metadata = dict(candidate.metadata)
            metadata.update(
                {
                    "first_stage_score": candidate.score,
                    "rerank_score": float(score),
                    "reranker_model": self.model_name,
                }
            )
            ranked.append(
                candidate.updated(
                    score=float(score),
                    retriever="cross_encoder_rerank",
                    metadata=metadata,
                )
            )
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[:top_k] if top_k is not None else ranked

    def _score(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]:
        if self._scorer is not None:
            return self._scorer(pairs)
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, max_length=self.max_length)
        return self._model.predict(list(pairs), show_progress_bar=False)

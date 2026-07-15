"""Rank fusion utilities."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from .models import EngineeringSearchResult


def rrf_fusion(
    rankings: Iterable[Sequence[EngineeringSearchResult]],
    *,
    k: int = 60,
    top_k: int | None = None,
    weights: Sequence[float] | None = None,
) -> list[EngineeringSearchResult]:
    """Fuse ranked lists with Reciprocal Rank Fusion.

    Scores from the underlying retrievers are deliberately not compared; only
    rank positions contribute. Duplicate results are identified by
    :attr:`EngineeringSearchResult.result_id`.
    """

    if k < 0:
        raise ValueError("k must be non-negative")
    ranking_list = list(rankings)
    if weights is None:
        ranking_weights = [1.0] * len(ranking_list)
    else:
        ranking_weights = [float(weight) for weight in weights]
        if len(ranking_weights) != len(ranking_list):
            raise ValueError("weights must match the number of rankings")
        if any(weight < 0 for weight in ranking_weights):
            raise ValueError("weights cannot be negative")

    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    representatives: dict[str, EngineeringSearchResult] = {}
    contributors: dict[str, list[str]] = {}
    component_scores: dict[str, list[float]] = {}
    sequence_number = 0

    for ranking_index, (ranking, weight) in enumerate(zip(ranking_list, ranking_weights)):
        for rank, result in enumerate(ranking, start=1):
            result_id = result.result_id
            contribution = weight / (k + rank)
            scores[result_id] = scores.get(result_id, 0.0) + contribution
            component_scores.setdefault(result_id, []).append(contribution)
            # Preserve retriever provenance across nested fusion (for example
            # dense+BM25 inside a partition, then federated RRF across
            # partitions) instead of collapsing it to the unhelpful label
            # ``rrf``.
            nested = result.metadata.get("rrf_retrievers")
            if isinstance(nested, (list, tuple, set)) and nested:
                contributor_names = [str(item) for item in nested]
            else:
                contributor_names = [result.retriever or f"ranking_{ranking_index}"]
            for contributor in contributor_names:
                if contributor not in contributors.setdefault(result_id, []):
                    contributors[result_id].append(contributor)

            if result_id not in representatives or result.score > representatives[result_id].score:
                representatives[result_id] = result
            if result_id not in first_seen:
                first_seen[result_id] = sequence_number
                sequence_number += 1

    ordered_ids = sorted(scores, key=lambda item: (-scores[item], first_seen[item]))
    if top_k is not None:
        if top_k <= 0:
            return []
        ordered_ids = ordered_ids[:top_k]

    fused: list[EngineeringSearchResult] = []
    for result_id in ordered_ids:
        representative = representatives[result_id]
        metadata: dict[str, Any] = dict(representative.metadata)
        metadata["rrf_retrievers"] = tuple(contributors[result_id])
        metadata["rrf_component_scores"] = tuple(component_scores[result_id])
        fused.append(
            representative.updated(
                score=scores[result_id],
                retriever="rrf",
                metadata=metadata,
            )
        )
    return fused


reciprocal_rank_fusion = rrf_fusion

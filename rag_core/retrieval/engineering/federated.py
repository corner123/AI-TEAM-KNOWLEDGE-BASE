"""Federated retrieval across corpus and authority partitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Protocol, Sequence, runtime_checkable

from .fusion import rrf_fusion
from .models import EngineeringSearchResult


@runtime_checkable
class EngineeringRetriever(Protocol):
    def search(self, query: str, top_k: int = 5) -> list[EngineeringSearchResult]: ...


@dataclass(frozen=True, slots=True)
class FederatedPartition:
    corpus: str
    authority: str
    retriever: EngineeringRetriever
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.corpus:
            raise ValueError("partition corpus cannot be empty")
        if not self.authority:
            raise ValueError("partition authority cannot be empty")
        if self.weight < 0:
            raise ValueError("partition weight cannot be negative")


def _as_filter(values: str | Iterable[str] | None) -> set[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        return {values}
    return set(values)


class FederatedRetriever:
    """Search selected partitions and combine their ranks with weighted RRF."""

    def __init__(
        self,
        partitions: (
            Mapping[tuple[str, str], EngineeringRetriever]
            | Sequence[FederatedPartition]
            | None
        ) = None,
        *,
        rrf_k: int = 60,
        candidate_multiplier: int = 3,
        fail_open: bool = True,
    ) -> None:
        if rrf_k < 0:
            raise ValueError("rrf_k must be non-negative")
        if candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be at least one")
        self.rrf_k = rrf_k
        self.candidate_multiplier = candidate_multiplier
        self.fail_open = fail_open
        self._partitions: list[FederatedPartition] = []

        if isinstance(partitions, Mapping):
            for (corpus, authority), retriever in partitions.items():
                self.add_partition(corpus, authority, retriever)
        elif partitions:
            self._partitions.extend(partitions)

    @property
    def partitions(self) -> tuple[FederatedPartition, ...]:
        return tuple(self._partitions)

    def add_partition(
        self,
        corpus: str,
        authority: str,
        retriever: EngineeringRetriever,
        *,
        weight: float = 1.0,
    ) -> None:
        key = (corpus, authority)
        if any((part.corpus, part.authority) == key for part in self._partitions):
            raise ValueError(f"partition already exists: {corpus}/{authority}")
        self._partitions.append(FederatedPartition(corpus, authority, retriever, weight))

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        corpora: str | Iterable[str] | None = None,
        authorities: str | Iterable[str] | None = None,
    ) -> list[EngineeringSearchResult]:
        if top_k <= 0:
            return []
        corpus_filter = _as_filter(corpora)
        authority_filter = _as_filter(authorities)
        candidate_k = top_k * self.candidate_multiplier
        rankings: list[list[EngineeringSearchResult]] = []
        weights: list[float] = []

        for partition in self._partitions:
            if corpus_filter is not None and partition.corpus not in corpus_filter:
                continue
            if authority_filter is not None and partition.authority not in authority_filter:
                continue
            try:
                raw_results = partition.retriever.search(query, top_k=candidate_k)
            except Exception:
                if self.fail_open:
                    continue
                raise

            stamped: list[EngineeringSearchResult] = []
            for result in raw_results:
                metadata = dict(result.metadata)
                metadata["federated_partition"] = f"{partition.corpus}/{partition.authority}"
                stamped.append(
                    result.updated(
                        corpus=partition.corpus,
                        authority=partition.authority,
                        metadata=metadata,
                    )
                )
            if stamped:
                rankings.append(stamped)
                weights.append(partition.weight)

        return rrf_fusion(rankings, k=self.rrf_k, top_k=top_k, weights=weights)

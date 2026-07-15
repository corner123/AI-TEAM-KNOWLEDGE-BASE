"""Public response models for engineering retrieval and grounded answers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from rag_core.retrieval.engineering import EngineeringSearchResult, SourceIntent


@dataclass(frozen=True, slots=True)
class EvidenceCitation:
    """A structured, user-visible citation independent of vector-store details."""

    citation_id: str
    source: str
    corpus: str
    authority: str
    evidence_role: str
    line_start: int | None = None
    line_end: int | None = None
    symbol: str | None = None
    revision: str | None = None
    branch: str | None = None
    dirty: bool | None = None
    url: str | None = None
    live_verified: bool = False
    source_version: str | None = None
    fetched_at: str | None = None
    content_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RetrievalOutcome:
    query: str
    intent: SourceIntent
    results: list[EngineeringSearchResult]
    citations: list[EvidenceCitation]
    sufficient_evidence: bool
    refusal_reason: str | None = None
    live_verification_attempted: bool = False
    live_verification_terms: tuple[str, ...] = ()
    live_revision: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    matched_rule: str = ""

    def to_dict(self, *, include_content: bool = True) -> dict[str, Any]:
        return {
            "schema_version": "engineering-retrieval/v1",
            "query": self.query,
            "intent": self.intent.value,
            "sufficient_evidence": self.sufficient_evidence,
            "refusal_reason": self.refusal_reason,
            "live_verification_attempted": self.live_verification_attempted,
            "live_verification_terms": list(self.live_verification_terms),
            "live_revision": dict(self.live_revision),
            "warnings": list(self.warnings),
            "matched_rule": self.matched_rule,
            "citations": [citation.to_dict() for citation in self.citations],
            "results": [
                {
                    "content": result.content if include_content else "",
                    "citation": _public_source(result),
                    "source": _public_source(result),
                    "score": result.score,
                    "corpus": result.corpus,
                    "authority": result.authority,
                    "line_start": result.line_start,
                    "line_end": result.line_end,
                    "symbol": result.symbol,
                    "retriever": result.retriever,
                    "evidence_role": result.metadata.get("evidence_role", ""),
                    "live_verified": bool(result.metadata.get("live_verification")),
                    "metadata": _public_metadata(result.metadata),
                }
                for result in self.results
            ],
        }


_PUBLIC_METADATA_KEYS = frozenset(
    {
        "record_kind",
        "source_id",
        "source_type",
        "source_version",
        "source_fetched_at",
        "source_content_hash",
        "document_content_hash",
        "source_dirty",
        "source_commit_sha",
        "document_path",
        "document_title",
        "parent_path",
        "relative_path",
        "symbol_kind",
        "evidence_role",
        "verification_method",
        "verification_term",
        "git_commit_sha",
        "git_branch",
        "git_dirty",
        "federated_partition",
        "rrf_retrievers",
        "first_stage_score",
        "rerank_score",
        "retrieval_degraded_components",
    }
)


def _public_metadata(metadata: dict[str, Any] | Any) -> dict[str, Any]:
    """Return only stable evidence fields; never expose host or HTTP internals."""

    if not isinstance(metadata, dict):
        return {}
    return {
        key: metadata[key]
        for key in _PUBLIC_METADATA_KEYS
        if key in metadata and metadata[key] is not None
    }


def _public_source(result: EngineeringSearchResult) -> str:
    if result.metadata.get("live_verification"):
        relative = result.metadata.get("relative_path")
        if relative:
            return str(relative).replace("\\", "/")
    return result.source


@dataclass(slots=True)
class AnswerOutcome:
    query: str
    intent: SourceIntent
    answer: str
    refused: bool
    refusal_reason: str | None
    citations: list[EvidenceCitation]
    warnings: list[str] = field(default_factory=list)
    generation_mode: str = "deterministic"
    generation_provider: str = "deterministic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "intent": self.intent.value,
            "answer": self.answer,
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "warnings": list(self.warnings),
            "generation_mode": self.generation_mode,
            "generation_provider": self.generation_provider,
            "citations": [citation.to_dict() for citation in self.citations],
        }

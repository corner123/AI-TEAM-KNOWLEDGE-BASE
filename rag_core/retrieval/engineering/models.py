"""Shared data structures for engineering-oriented retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class EngineeringSearchResult:
    """A retrieval result with enough provenance for an engineering citation.

    ``corpus`` identifies the logical source partition (for example
    ``internal`` or ``official``), while ``authority`` describes why the
    source is trusted (for example ``code``, ``test``, ``design`` or
    ``official``).  File-backed results should populate the line fields.
    """

    content: str
    source: str
    score: float = 0.0
    corpus: str = "internal"
    authority: str = "unknown"
    line_start: int | None = None
    line_end: int | None = None
    symbol: str | None = None
    retriever: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.line_start is not None and self.line_start < 1:
            raise ValueError("line_start must be positive")
        if self.line_end is not None and self.line_end < 1:
            raise ValueError("line_end must be positive")
        if self.line_start is not None and self.line_end is not None and self.line_end < self.line_start:
            raise ValueError("line_end cannot be before line_start")

        # Copy caller-owned mappings so fusion can safely add provenance.
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "source", str(self.source))

    @property
    def result_id(self) -> str:
        """Return a deterministic identity used to deduplicate fused ranks."""

        explicit_id = self.metadata.get("document_id") or self.metadata.get("id")
        if explicit_id is not None:
            # IDs are commonly unique only inside a vector-store partition.
            return f"{self.corpus}|{explicit_id}"

        digest = hashlib.sha1(self.content.encode("utf-8", errors="replace")).hexdigest()[:16]
        return "|".join(
            (
                self.corpus,
                self.source,
                str(self.line_start or ""),
                str(self.line_end or ""),
                self.symbol or "",
                digest,
            )
        )

    @property
    def citation(self) -> str:
        """Render a compact source citation without exposing retrieval internals."""

        if self.line_start is None:
            return self.source
        if self.line_end is None or self.line_end == self.line_start:
            return f"{self.source}:{self.line_start}"
        return f"{self.source}:{self.line_start}-{self.line_end}"

    def updated(self, **changes: Any) -> "EngineeringSearchResult":
        """Return a modified copy, useful for retrievers and rank fusion."""

        return replace(self, **changes)

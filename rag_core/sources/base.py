"""Source adapter protocol and collection result."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .schema import DocumentRecord, SourceRecord


@dataclass(slots=True)
class CollectedSource:
    record: SourceRecord
    documents: list[DocumentRecord]


@runtime_checkable
class SourceAdapter(Protocol):
    source_id: str

    def collect(self) -> CollectedSource:
        """Collect a point-in-time snapshot without mutating the source."""

        ...

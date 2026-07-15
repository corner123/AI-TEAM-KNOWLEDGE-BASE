"""Standalone, versioned ingestion pipeline."""

from .manifest import BuildManifest, ManifestDiff, RecordChanges
from .pipeline import (
    CHUNKER_VERSION,
    PIPELINE_VERSION,
    IngestionPipeline,
    IngestionResult,
)

__all__ = [
    "CHUNKER_VERSION",
    "PIPELINE_VERSION",
    "BuildManifest",
    "IngestionPipeline",
    "IngestionResult",
    "ManifestDiff",
    "RecordChanges",
]

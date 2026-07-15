"""Versioned source adapters for the standalone ingestion pipeline."""

from .base import CollectedSource, SourceAdapter
from .catalog import CatalogEntry, SourceCatalog
from .git_repository import GitRepositoryError, GitRepositorySource
from .official_web import FetchResponse, OfficialWebSource, clean_html_document
from .schema import (
    SCHEMA_VERSION,
    ChunkRecord,
    DocumentRecord,
    SourceRecord,
    canonical_hash,
    content_hash,
)

__all__ = [
    "SCHEMA_VERSION",
    "CatalogEntry",
    "ChunkRecord",
    "CollectedSource",
    "DocumentRecord",
    "FetchResponse",
    "GitRepositoryError",
    "GitRepositorySource",
    "OfficialWebSource",
    "SourceAdapter",
    "SourceCatalog",
    "SourceRecord",
    "canonical_hash",
    "clean_html_document",
    "content_hash",
]

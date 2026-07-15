"""Small workflow functions intended for ``main.py`` and automation scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from rag_core.ingestion import BuildManifest, IngestionPipeline, IngestionResult
from rag_core.sources import SourceCatalog
from rag_core.sources.official_web import Fetcher

from .index import EngineeringIndex
from .service import EngineeringRAGService


DEFAULT_CATALOG = Path("data/sources/catalog.yaml")
DEFAULT_MANIFEST = Path("data/manifests/builds/current.json")
DEFAULT_INDEX = Path("data/indexes/engineering")


def sync_engineering_sources(
    catalog_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    *,
    chunk_size: int = 1_200,
    chunk_overlap: int = 120,
    web_fetcher: Fetcher | None = None,
) -> IngestionResult:
    """Collect catalogued sources and atomically update the build manifest."""

    catalog_file = Path(
        catalog_path or os.getenv("SOURCE_CATALOG_PATH", str(DEFAULT_CATALOG))
    )
    manifest_file = Path(manifest_path or DEFAULT_MANIFEST)
    catalog = SourceCatalog.load(catalog_file)
    previous = manifest_file if manifest_file.is_file() else None
    pipeline = IngestionPipeline(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return pipeline.build(
        catalog.create_sources(web_fetcher=web_fetcher),
        previous=previous,
        output_path=manifest_file,
    )


def build_engineering_index(
    manifest_path: str | Path | None = None,
    index_root: str | Path | None = None,
    *,
    embedding_manager=None,
    batch_size: int = 128,
) -> EngineeringIndex:
    """Build all internal/official partitions from a completed manifest."""

    manifest = BuildManifest.read(manifest_path or DEFAULT_MANIFEST)
    root = index_root or os.getenv("ENGINEERING_INDEX_DIR", str(DEFAULT_INDEX))
    return EngineeringIndex.build(
        manifest,
        root,
        embedding_manager=embedding_manager,
        batch_size=batch_size,
    )


def load_engineering_service(
    index_root: str | Path | None = None,
    *,
    mini_nanobot_repo: str | Path | None = None,
    manifest_path: str | Path | None = None,
    embedding_manager=None,
    answerer=None,
) -> EngineeringRAGService:
    """Load the query service with a fixed, configured live-code root."""

    root = index_root or os.getenv("ENGINEERING_INDEX_DIR", str(DEFAULT_INDEX))
    return EngineeringRAGService.from_index(
        root,
        mini_nanobot_repo=mini_nanobot_repo or os.getenv("MINI_NANOBOT_REPO"),
        manifest_path=manifest_path or os.getenv(
            "ENGINEERING_MANIFEST_PATH", str(DEFAULT_MANIFEST)
        ),
        embedding_manager=embedding_manager,
        answerer=answerer,
    )


def query_engineering_knowledge(
    query: str,
    *,
    top_k: int = 5,
    answer: bool = False,
    service: EngineeringRAGService | None = None,
    **load_options: Any,
) -> dict[str, Any]:
    """Return a JSON-ready retrieval or grounded-answer payload."""

    active = service or load_engineering_service(**load_options)
    if answer:
        return active.answer(query, top_k=top_k).to_dict()
    return active.retrieve(query, top_k=top_k).to_dict()

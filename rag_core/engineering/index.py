"""Partitioned dense + BM25 indexes for engineering knowledge."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
from typing import Any, Iterable, Mapping
import uuid

from langchain_core.documents import Document

from config import EmbeddingConfig
from rag_core.index.embeddings import EmbeddingManager
from rag_core.index.local_vector_store import LocalVectorStore
from rag_core.ingestion import BuildManifest
from rag_core.retrieval.engineering import (
    BM25Retriever,
    EngineeringSearchResult,
    FederatedRetriever,
    rrf_fusion,
)
from rag_core.sources.schema import ChunkRecord, DocumentRecord, SourceRecord


INDEX_SCHEMA_VERSION = 3
_HTTP_RE = re.compile(r"^https?://", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class PartitionSpec:
    corpus: str
    authority: str
    directory: str
    document_count: int
    weight: float = 1.0
    file_sha256: dict[str, str] = field(default_factory=dict)


def infer_partition(
    chunk: ChunkRecord,
    document: DocumentRecord,
    source: SourceRecord,
) -> tuple[str, str]:
    """Infer ``(corpus, authority)`` while honoring explicit source metadata."""

    metadata = {**source.metadata, **document.metadata, **chunk.metadata}
    source_type = str(metadata.get("source_type") or source.source_type).casefold()
    corpus = str(metadata.get("corpus") or "").strip().casefold()
    if not corpus:
        corpus = "official" if source_type == "official_web" else "internal"

    explicit_authority = str(metadata.get("authority") or "").strip().casefold()
    if corpus == "official" or source_type == "official_web":
        return "official", "official"
    if explicit_authority in {"code", "test", "design", "history"}:
        return corpus, explicit_authority

    path = document.relative_path.split("#", 1)[0].replace("\\", "/")
    if str(metadata.get("record_kind") or "").casefold() == "git_commit":
        return corpus, "history"
    normalized = f"/{path.casefold().lstrip('/')}"
    name = PurePosixPath(path).name.casefold()
    suffix = PurePosixPath(path).suffix.casefold()
    if "/tests/" in normalized or name.startswith("test_") or name.endswith("_test.py"):
        authority = "test"
    elif suffix in {".py", ".pyi", ".js", ".ts", ".tsx", ".go", ".rs", ".java"} or "#symbol:" in document.relative_path:
        authority = "code"
    else:
        authority = "design"
    return corpus, authority


def _partition_weight(corpus: str, authority: str) -> float:
    return {
        ("internal", "code"): 1.35,
        ("internal", "test"): 1.20,
        ("internal", "design"): 1.00,
        ("internal", "history"): 0.80,
        ("official", "official"): 0.90,
    }.get((corpus, authority), 1.0)


def _safe_partition_name(corpus: str, authority: str) -> str:
    value = f"{corpus}__{authority}"
    if not re.fullmatch(r"[a-z0-9_-]+", value):
        raise ValueError(f"unsafe partition name: {value}")
    return value


def _chunk_document(
    chunk: ChunkRecord,
    document: DocumentRecord,
    source: SourceRecord,
) -> Document:
    corpus, authority = infer_partition(chunk, document, source)
    metadata: dict[str, Any] = {
        **source.metadata,
        **document.metadata,
        **chunk.metadata,
        "chunk_id": chunk.chunk_id,
        "doc_id": document.doc_id,
        "source_id": source.source_id,
        "source_type": source.source_type,
        "source_uri": source.uri,
        "source_version": source.version,
        "source_license": source.license,
        "source_content_hash": source.content_hash,
        "source_fetched_at": source.fetched_at,
        "source_dirty": source.dirty,
        "document_content_hash": document.content_hash,
        "source_commit_sha": source.commit_sha or document.metadata.get("source_commit_sha"),
        "document_path": document.relative_path,
        "document_title": document.title,
        "corpus": corpus,
        "authority": authority,
    }
    citation_source = document.relative_path
    if corpus == "official":
        citation_source = document.relative_path if _HTTP_RE.match(document.relative_path) else source.uri
    metadata["citation_source"] = citation_source
    return Document(page_content=chunk.content, metadata=metadata)


def _as_search_result(document: Document) -> EngineeringSearchResult:
    metadata = dict(document.metadata)
    source = str(
        metadata.get("citation_source")
        or metadata.get("document_path")
        or metadata.get("source_uri")
        or "unknown-source"
    )
    return EngineeringSearchResult(
        content=document.page_content,
        source=source,
        corpus=str(metadata.get("corpus") or "internal"),
        authority=str(metadata.get("authority") or "unknown"),
        line_start=_optional_int(metadata.get("line_start") or metadata.get("chunk_line_start")),
        line_end=_optional_int(metadata.get("line_end") or metadata.get("chunk_line_end")),
        symbol=_optional_text(
            metadata.get("symbol_qualified_name")
            or metadata.get("symbol")
            or metadata.get("symbol_name")
        ),
        metadata=metadata,
    )


class DenseVectorRetriever:
    """Adapter from :class:`LocalVectorStore` to engineering search results."""

    def __init__(self, store: LocalVectorStore) -> None:
        self.store = store

    def search(self, query: str, top_k: int = 5) -> list[EngineeringSearchResult]:
        results: list[EngineeringSearchResult] = []
        for raw in self.store.search_dense(query, top_k=top_k):
            metadata = dict(raw.get("metadata") or {})
            source = str(
                metadata.get("citation_source")
                or metadata.get("document_path")
                or metadata.get("source_uri")
                or "unknown-source"
            )
            results.append(
                EngineeringSearchResult(
                    content=str(raw.get("text") or ""),
                    source=source,
                    score=float(raw.get("score") or 0.0),
                    corpus=str(metadata.get("corpus") or "internal"),
                    authority=str(metadata.get("authority") or "unknown"),
                    line_start=_optional_int(metadata.get("line_start") or metadata.get("chunk_line_start")),
                    line_end=_optional_int(metadata.get("line_end") or metadata.get("chunk_line_end")),
                    symbol=_optional_text(
                        metadata.get("symbol_qualified_name")
                        or metadata.get("symbol")
                        or metadata.get("symbol_name")
                    ),
                    retriever="dense",
                    metadata=metadata,
                )
            )
        return results


class HybridPartitionRetriever:
    """Fuse dense semantic retrieval and exact-token BM25 inside one partition."""

    def __init__(
        self,
        dense: DenseVectorRetriever,
        bm25: BM25Retriever,
        *,
        rrf_k: int = 60,
        candidate_multiplier: int = 3,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
        fail_open: bool = True,
    ) -> None:
        self.dense = dense
        self.bm25 = bm25
        self.rrf_k = rrf_k
        self.candidate_multiplier = max(1, candidate_multiplier)
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.fail_open = bool(fail_open)

    def search(self, query: str, top_k: int = 5) -> list[EngineeringSearchResult]:
        if top_k <= 0:
            return []
        candidate_k = max(top_k, top_k * self.candidate_multiplier)
        degraded: list[str] = []
        try:
            dense_results = self.dense.search(query, top_k=candidate_k)
        except Exception:
            dense_results = []
            degraded.append("dense")
        try:
            sparse_results = self.bm25.search(query, top_k=candidate_k)
        except Exception:
            sparse_results = []
            degraded.append("bm25")
        if degraded and not self.fail_open:
            raise RuntimeError(
                "hybrid partition retrieval failed closed: " + ", ".join(degraded)
            )
        fused = rrf_fusion(
            [dense_results, sparse_results],
            k=self.rrf_k,
            top_k=top_k,
            weights=[self.dense_weight, self.bm25_weight],
        )
        if not degraded:
            return fused
        return [
            result.updated(
                metadata={
                    **result.metadata,
                    "retrieval_degraded_components": list(degraded),
                }
            )
            for result in fused
        ]


class EngineeringIndex:
    """Build and load physically separated internal/official FAISS partitions."""

    CATALOG_FILENAME = "partitions.json"

    def __init__(
        self,
        root: str | Path,
        embedding_manager: EmbeddingManager,
        specs: Iterable[PartitionSpec],
        *,
        build_id: str = "",
        embedding_model: str = "",
    ) -> None:
        self.root = Path(root).resolve()
        self.embedding_manager = embedding_manager
        self.specs = tuple(specs)
        self.build_id = build_id
        self.embedding_model = embedding_model or _manager_model_name(embedding_manager)
        self.federated = FederatedRetriever()
        self._stores: list[LocalVectorStore] = []
        self._load_partitions()

    @classmethod
    def build(
        cls,
        manifest: BuildManifest,
        root: str | Path,
        *,
        embedding_manager: EmbeddingManager | None = None,
        batch_size: int = 128,
    ) -> "EngineeringIndex":
        manager = embedding_manager or _default_embedding_manager()
        index_root = Path(root).resolve()
        index_root.parent.mkdir(parents=True, exist_ok=True)
        lock_path = index_root.parent / f".{index_root.name}.build.lock"
        try:
            build_lock = _acquire_build_lock(lock_path)
        except FileExistsError as exc:
            raise RuntimeError(
                f"engineering index build is already running: {lock_path}"
            ) from exc
        build_token = uuid.uuid4().hex
        staging_root = index_root.parent / f".{index_root.name}.staging-{build_token}"
        backup_root = index_root.parent / f".{index_root.name}.backup-{build_token}"
        try:
            _recover_interrupted_publish(index_root)
            staging_root.mkdir()
            return cls._build_and_publish(
                manifest,
                index_root,
                staging_root,
                backup_root,
                manager,
                batch_size=batch_size,
            )
        finally:
            if staging_root.exists():
                shutil.rmtree(staging_root, ignore_errors=True)
            # A backup is deleted only by _build_and_publish after the newly
            # published index has loaded successfully. If rollback itself
            # fails, preserving the backup is more important than cleanup.
            _release_build_lock(build_lock)

    @classmethod
    def _build_and_publish(
        cls,
        manifest: BuildManifest,
        index_root: Path,
        staging_root: Path,
        backup_root: Path,
        manager: EmbeddingManager,
        *,
        batch_size: int,
    ) -> "EngineeringIndex":
        sources = {record.source_id: record for record in manifest.sources}
        documents = {record.doc_id: record for record in manifest.documents}
        if not sources or not documents or not manifest.chunks:
            raise ValueError("refusing to publish an empty engineering index")
        if len(sources) != len(manifest.sources):
            raise ValueError("manifest contains duplicate source IDs")
        if len(documents) != len(manifest.documents):
            raise ValueError("manifest contains duplicate document IDs")
        partitions: dict[tuple[str, str], list[Document]] = {}
        seen_chunk_ids: set[str] = set()
        for chunk in manifest.chunks:
            if chunk.chunk_id in seen_chunk_ids:
                raise ValueError(f"manifest contains duplicate chunk ID: {chunk.chunk_id}")
            seen_chunk_ids.add(chunk.chunk_id)
            document = documents.get(chunk.doc_id)
            source = sources.get(chunk.source_id)
            if document is None or source is None:
                raise ValueError(f"orphan chunk in manifest: {chunk.chunk_id}")
            if document.source_id != chunk.source_id:
                raise ValueError(
                    f"chunk/document source mismatch: {chunk.chunk_id}"
                )
            indexed = _chunk_document(chunk, document, source)
            key = (str(indexed.metadata["corpus"]), str(indexed.metadata["authority"]))
            partitions.setdefault(key, []).append(indexed)

        specs: list[PartitionSpec] = []
        for (corpus, authority), indexed_documents in sorted(partitions.items()):
            directory = _safe_partition_name(corpus, authority)
            store = LocalVectorStore(manager, persist_dir=str(staging_root / directory))
            store.replace_documents(indexed_documents, batch_size=batch_size)
            partition_root = staging_root / directory
            file_sha256 = {
                name: _sha256_file(partition_root / name)
                for name in ("index.faiss", "documents.json", "index_meta.json")
            }
            specs.append(
                PartitionSpec(
                    corpus=corpus,
                    authority=authority,
                    directory=directory,
                    document_count=len(indexed_documents),
                    weight=_partition_weight(corpus, authority),
                    file_sha256=file_sha256,
                )
            )

        catalog = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "build_id": manifest.build_id,
            "embedding_model": _manager_model_name(manager),
            "partitions": [asdict(spec) for spec in specs],
        }
        catalog_path = staging_root / cls.CATALOG_FILENAME
        temporary_catalog = staging_root / f".{cls.CATALOG_FILENAME}.tmp"
        temporary_catalog.write_text(
            json.dumps(catalog, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_catalog, catalog_path)
        # Load every staged partition before changing the currently published
        # root. A missing/corrupt FAISS or document file fails here and leaves
        # the old index untouched.
        staged = cls.load(staging_root, embedding_manager=manager)
        if staged.stats()["document_count"] != len(manifest.chunks):
            raise RuntimeError("staged engineering index count does not match manifest")

        if index_root.exists():
            os.replace(index_root, backup_root)
        try:
            os.replace(staging_root, index_root)
        except Exception:
            if backup_root.exists() and not index_root.exists():
                os.replace(backup_root, index_root)
            raise
        try:
            published = cls.load(index_root, embedding_manager=manager)
        except Exception as publish_error:
            if backup_root.exists():
                failed_root = index_root.parent / (
                    f".{index_root.name}.failed-{uuid.uuid4().hex}"
                )
                try:
                    if index_root.exists():
                        os.replace(index_root, failed_root)
                    os.replace(backup_root, index_root)
                except Exception as rollback_error:
                    raise RuntimeError(
                        "published index validation failed and rollback could not "
                        f"complete; preserved backup at {backup_root}"
                    ) from rollback_error
                finally:
                    if failed_root.exists() and index_root.exists():
                        shutil.rmtree(failed_root, ignore_errors=True)
            else:
                shutil.rmtree(index_root, ignore_errors=True)
            raise
        if backup_root.exists():
            # Cleanup is not part of the publish transaction. A transient
            # antivirus/file-handle failure must not turn a successful build
            # into a reported failure.
            shutil.rmtree(backup_root, ignore_errors=True)
        return published

    @classmethod
    def load(
        cls,
        root: str | Path,
        *,
        embedding_manager: EmbeddingManager | None = None,
    ) -> "EngineeringIndex":
        index_root = Path(root).resolve()
        catalog_path = index_root / cls.CATALOG_FILENAME
        if not catalog_path.is_file():
            raise FileNotFoundError(f"engineering index catalog not found: {catalog_path}")
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", -1)) != INDEX_SCHEMA_VERSION:
            raise ValueError("unsupported engineering index schema")
        raw_specs = payload.get("partitions")
        if not isinstance(raw_specs, list) or not raw_specs:
            raise ValueError("engineering index catalog has no partitions")
        specs = [PartitionSpec(**item) for item in raw_specs]
        build_id = str(payload.get("build_id") or "").strip()
        if not build_id:
            raise ValueError("engineering index catalog has no build_id")
        model_name = str(payload.get("embedding_model") or "").strip()
        if not model_name:
            raise ValueError("engineering index catalog has no embedding_model")
        manager = embedding_manager or _default_embedding_manager(model_name)
        if embedding_manager is not None and _manager_model_name(manager) != model_name:
            raise ValueError(
                "engineering index embedding model does not match the provided manager"
            )
        return cls(
            index_root,
            manager,
            specs,
            build_id=build_id,
            embedding_model=model_name,
        )

    def stats(self) -> dict[str, Any]:
        return {
            "build_id": self.build_id,
            "index_root": str(self.root),
            "embedding_model": self.embedding_model,
            "partitions": [asdict(spec) for spec in self.specs],
            "document_count": sum(spec.document_count for spec in self.specs),
        }

    def _load_partitions(self) -> None:
        seen_chunk_ids: set[str] = set()
        for spec in self.specs:
            partition_path = (self.root / spec.directory).resolve()
            try:
                partition_path.relative_to(self.root)
            except ValueError as exc:
                raise ValueError(f"partition escapes index root: {spec.directory}") from exc
            required_files = ("index.faiss", "documents.json", "index_meta.json")
            missing = [
                name for name in required_files if not (partition_path / name).is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    f"engineering partition {spec.directory} is incomplete: {missing}"
                )
            if set(spec.file_sha256) != set(required_files):
                raise ValueError(
                    f"engineering partition {spec.directory} has incomplete checksums"
                )
            for name in required_files:
                actual_hash = _sha256_file(partition_path / name)
                if actual_hash != spec.file_sha256[name]:
                    raise ValueError(
                        f"engineering partition {spec.directory} checksum mismatch: {name}"
                    )
            store = LocalVectorStore(self.embedding_manager, persist_dir=str(partition_path))
            documents = [_as_search_result(document) for document in store.all_documents()]
            if len(documents) != spec.document_count:
                raise ValueError(
                    f"engineering partition {spec.directory} count mismatch: "
                    f"catalog={spec.document_count}, documents={len(documents)}"
                )
            for document in documents:
                if document.corpus != spec.corpus or document.authority != spec.authority:
                    raise ValueError(
                        f"engineering partition metadata mismatch: {spec.directory}"
                    )
                chunk_id = str(document.metadata.get("chunk_id") or "")
                if not chunk_id or chunk_id in seen_chunk_ids:
                    raise ValueError(
                        f"missing or duplicate indexed chunk ID in {spec.directory}: {chunk_id}"
                    )
                seen_chunk_ids.add(chunk_id)
            retriever = HybridPartitionRetriever(
                DenseVectorRetriever(store),
                BM25Retriever(documents),
            )
            self.federated.add_partition(
                spec.corpus,
                spec.authority,
                retriever,
                weight=spec.weight,
            )
            self._stores.append(store)


class _BuildFileLock:
    """Persistent-file advisory lock released automatically on process exit."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.fd: int | None = None

    def acquire(self) -> None:
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            if os.fstat(fd).st_size == 0:
                os.write(fd, b"\0")
                os.fsync(fd)
            os.lseek(fd, 0, os.SEEK_SET)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            raise FileExistsError(self.path) from exc
        self.fd = fd

    def release(self) -> None:
        fd, self.fd = self.fd, None
        if fd is None:
            return
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _acquire_build_lock(lock_path: Path) -> _BuildFileLock:
    """Acquire an OS lock; the stable lock path is never unlinked."""

    lock = _BuildFileLock(lock_path)
    lock.acquire()
    return lock


def _release_build_lock(lock: _BuildFileLock) -> None:
    lock.release()


def _process_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if os.name == "nt":
        return _windows_process_is_running(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _windows_process_is_running(pid: int) -> bool:
    """Query process state without sending a signal on Windows.

    ``os.kill(pid, 0)`` is unsafe on Windows because signal 0 is passed to
    ``TerminateProcess`` rather than acting as the POSIX existence probe.
    """

    process_query_limited_information = 0x1000
    still_active = 259
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    open_process = kernel32.OpenProcess
    open_process.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    open_process.restype = wintypes.HANDLE
    get_exit_code = kernel32.GetExitCodeProcess
    get_exit_code.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    get_exit_code.restype = wintypes.BOOL
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL

    handle = open_process(process_query_limited_information, False, pid)
    if not handle:
        # Access denied means the process exists but is not queryable. Invalid
        # parameter is the normal response for a PID that does not exist.
        return ctypes.get_last_error() == 5
    try:
        exit_code = wintypes.DWORD()
        if not get_exit_code(handle, ctypes.byref(exit_code)):
            return True
        return exit_code.value == still_active
    finally:
        close_handle(handle)


def _recover_interrupted_publish(index_root: Path) -> None:
    parent = index_root.parent
    backups = sorted(parent.glob(f".{index_root.name}.backup-*"))
    stagings = sorted(parent.glob(f".{index_root.name}.staging-*"))
    root_complete = _index_tree_looks_complete(index_root)

    if root_complete:
        # The current root already committed. Compatible backups are cleanup
        # debris; legacy/incomplete backups are preserved under a quarantine
        # name so they cannot block a later current-schema build.
        for backup in backups:
            if _index_tree_looks_complete(backup):
                shutil.rmtree(backup, ignore_errors=True)
            else:
                quarantine = parent / (
                    f".{index_root.name}.quarantine-{uuid.uuid4().hex}"
                )
                os.replace(backup, quarantine)
        for path in stagings:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
        return

    if len(backups) > 1:
        raise RuntimeError(
            "ambiguous interrupted engineering-index publish; multiple backups "
            f"must be inspected manually: {[str(path) for path in backups]}"
        )
    backup = backups[0] if backups else None
    if backup is not None and not _index_tree_looks_complete(backup):
        quarantine = parent / f".{index_root.name}.quarantine-{uuid.uuid4().hex}"
        os.replace(backup, quarantine)
        backup = None

    if backup is not None:
        failed_root = parent / f".{index_root.name}.failed-{uuid.uuid4().hex}"
        if index_root.exists():
            os.replace(index_root, failed_root)
        try:
            os.replace(backup, index_root)
        except Exception:
            if failed_root.exists() and not index_root.exists():
                os.replace(failed_root, index_root)
            raise
        if failed_root.exists():
            shutil.rmtree(failed_root, ignore_errors=True)
        backup = None

    for path in stagings:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)


def _index_tree_looks_complete(root: Path) -> bool:
    catalog_path = root / EngineeringIndex.CATALOG_FILENAME
    if not catalog_path.is_file():
        return False
    try:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", -1)) != INDEX_SCHEMA_VERSION:
            return False
        partitions = payload.get("partitions")
        if not isinstance(partitions, list) or not partitions:
            return False
        for item in partitions:
            if not isinstance(item, dict):
                return False
            directory = str(item.get("directory") or "")
            if not re.fullmatch(r"[a-z0-9_-]+", directory):
                return False
            partition = root / directory
            required = ("index.faiss", "documents.json", "index_meta.json")
            checksums = item.get("file_sha256")
            if not isinstance(checksums, dict) or set(checksums) != set(required):
                return False
            for name in required:
                path = partition / name
                if not path.is_file() or _sha256_file(path) != checksums[name]:
                    return False
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 1 else None


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _default_embedding_manager(model_name: str | None = None) -> EmbeddingManager:
    selected = model_name or os.getenv(
        "ENGINEERING_EMBEDDING_MODEL_PATH", "BAAI/bge-small-zh-v1.5"
    )
    return EmbeddingManager(EmbeddingConfig(model_path=selected))


def _manager_model_name(manager: Any) -> str:
    config = getattr(manager, "config", None)
    value = getattr(config, "model_path", None)
    return str(value or manager.__class__.__name__)

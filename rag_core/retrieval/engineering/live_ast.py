"""Read-only Python AST symbol lookup inside a bounded repository."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import re
import threading
from typing import Iterable

from .models import EngineeringSearchResult


_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{1,255}")
_EXCLUDED = frozenset(
    {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".nanobot"}
)


class LiveASTRetriever:
    """Locate current Python symbols without importing or executing code."""

    def __init__(
        self,
        repo_root: str | Path,
        *,
        excluded_dirs: Iterable[str] = _EXCLUDED,
        max_file_size_bytes: int = 1_000_000,
        max_results: int = 100,
    ) -> None:
        root = Path(repo_root).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"repo_root is not a directory: {root}")
        if max_file_size_bytes <= 0 or max_results <= 0:
            raise ValueError("AST scan limits must be positive")
        self.repo_root = root
        self.excluded_dirs = frozenset(excluded_dirs)
        self.max_file_size_bytes = int(max_file_size_bytes)
        self.max_results = int(max_results)
        self._cache: dict[
            Path, tuple[tuple[int, int], tuple[EngineeringSearchResult, ...]]
        ] = {}
        self._cache_lock = threading.RLock()

    def search(self, query: str, top_k: int = 10) -> list[EngineeringSearchResult]:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if any(char in query for char in ("\x00", "\r", "\n")):
            raise ValueError("query cannot contain control characters")
        # Python identifiers are case-sensitive. This retriever is used as a
        # claim verifier, so partial/case-insensitive discovery is unsafe.
        terms = _IDENTIFIER.findall(query)
        if not terms or top_k <= 0:
            return []
        limit = min(int(top_k), self.max_results)
        matches: list[tuple[int, EngineeringSearchResult]] = []

        paths = self._python_files()
        for path in paths:
            for result in self._cached_symbols(path):
                symbol = result.symbol or ""
                score = _match_score(symbol, terms)
                if score <= 0:
                    continue
                matches.append((score, result.updated(score=float(score))))

        # Do not retain records for files removed since the previous call.
        with self._cache_lock:
            active = set(paths)
            for cached_path in set(self._cache).difference(active):
                del self._cache[cached_path]

        matches.sort(key=lambda item: (-item[0], item[1].source, item[1].line_start or 0))
        return [result for _, result in matches[:limit]]

    def _cached_symbols(self, path: Path) -> tuple[EngineeringSearchResult, ...]:
        try:
            stat = path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            return ()
        with self._cache_lock:
            cached = self._cache.get(path)
            if cached is not None and cached[0] == signature:
                return cached[1]
            try:
                source = path.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(source)
            except (OSError, SyntaxError, UnicodeError):
                self._cache[path] = (signature, ())
                return ()
            lines = source.splitlines()
            relative = path.relative_to(self.repo_root).as_posix()
            records: list[EngineeringSearchResult] = []
            for qualified_name, node, kind in _symbols(tree):
                start = min(
                    [
                        getattr(item, "lineno", node.lineno)
                        for item in getattr(node, "decorator_list", [])
                    ]
                    + [node.lineno]
                )
                end = getattr(node, "end_lineno", None) or node.lineno
                records.append(
                    EngineeringSearchResult(
                        content="\n".join(lines[start - 1 : end]).strip(),
                        source=str(path),
                        corpus="internal",
                        authority="code",
                        line_start=start,
                        line_end=end,
                        symbol=qualified_name,
                        retriever="live_ast",
                        metadata={
                            "repo_root": str(self.repo_root),
                            "relative_path": relative,
                            "symbol_kind": kind,
                            "live_verification": True,
                        },
                    )
                )
            result = tuple(records)
            self._cache[path] = (signature, result)
            return result

    def _python_files(self) -> list[Path]:
        paths: list[Path] = []
        for directory, dir_names, file_names in os.walk(self.repo_root, followlinks=False):
            dir_names[:] = sorted(name for name in dir_names if name not in self.excluded_dirs)
            for file_name in sorted(file_names):
                if not file_name.endswith(".py"):
                    continue
                path = Path(directory, file_name)
                try:
                    if path.is_symlink():
                        continue
                    resolved = path.resolve()
                    resolved.relative_to(self.repo_root)
                    if resolved.stat().st_size > self.max_file_size_bytes:
                        continue
                except (OSError, ValueError):
                    continue
                paths.append(resolved)
        return paths


def _symbols(tree: ast.AST):
    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.parents: list[str] = []
            self.found: list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef, str]] = []

        def _visit_named(self, node, kind: str) -> None:
            qualified = ".".join([*self.parents, node.name])
            self.found.append((qualified, node, kind))
            self.parents.append(node.name)
            self.generic_visit(node)
            self.parents.pop()

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self._visit_named(node, "class")

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._visit_named(node, "function")

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._visit_named(node, "async_function")

    visitor = Visitor()
    visitor.visit(tree)
    return visitor.found


def _match_score(symbol: str, terms: list[str]) -> int:
    leaf = symbol.rsplit(".", 1)[-1]
    best = 0
    for term in terms:
        if term == symbol:
            best = max(best, 4)
        elif "." not in term and term == leaf:
            best = max(best, 3)
    return best

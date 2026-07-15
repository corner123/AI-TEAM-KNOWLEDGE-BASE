"""Read-only, repository-bounded live code search."""

from __future__ import annotations

import os
import fnmatch
from pathlib import Path
import re
import shutil
import subprocess
from typing import Iterable

from .models import EngineeringSearchResult


_RG_LINE = re.compile(r"^(.*?):(\d+):(\d+):(.*)$")
_DEFAULT_EXCLUDED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".idea",
        ".vscode",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".nanobot",
    }
)
_DEFAULT_ALLOWED_FILE_PATTERNS = (
    "*.py",
    "*.pyi",
    "*.toml",
    "*.yaml",
    "*.yml",
    "*.json",
    "Dockerfile",
    "Dockerfile.*",
)


class LiveCodeRetriever:
    """Verify facts against an external repository without modifying it.

    Ripgrep is called with an argument vector, ``shell=False``, a fixed-string
    pattern and ``repo_root`` as its working directory. The search target is
    always ``.`` and every returned path is resolved and checked against the
    configured root. Ripgrep does not follow symlinks by default. If it is not
    installed or exits abnormally, a bounded Python text scan is used.
    """

    def __init__(
        self,
        repo_root: str | Path,
        *,
        prefer_rg: bool = True,
        timeout_seconds: float = 5.0,
        max_file_size_bytes: int = 1_000_000,
        max_results: int = 100,
        excluded_dirs: Iterable[str] = _DEFAULT_EXCLUDED_DIRS,
        allowed_file_patterns: Iterable[str] = _DEFAULT_ALLOWED_FILE_PATTERNS,
        corpus: str = "external_repo",
        authority: str = "code",
        case_sensitive: bool = False,
    ) -> None:
        root = Path(repo_root).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"repo_root is not a directory: {root}")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_file_size_bytes <= 0:
            raise ValueError("max_file_size_bytes must be positive")
        if max_results <= 0:
            raise ValueError("max_results must be positive")

        self.repo_root = root
        self.timeout_seconds = float(timeout_seconds)
        self.max_file_size_bytes = int(max_file_size_bytes)
        self.max_results = int(max_results)
        self.excluded_dirs = frozenset(excluded_dirs)
        self.allowed_file_patterns = tuple(allowed_file_patterns)
        if not self.allowed_file_patterns:
            raise ValueError("allowed_file_patterns cannot be empty")
        self.corpus = corpus
        self.authority = authority
        self.case_sensitive = case_sensitive
        self._rg_path = shutil.which("rg") if prefer_rg else None

    def search(self, query: str, top_k: int = 10) -> list[EngineeringSearchResult]:
        query = self._validate_query(query)
        if top_k <= 0:
            return []
        limit = min(int(top_k), self.max_results)

        if self._rg_path:
            rg_results = self._search_with_rg(query, limit)
            if rg_results is not None:
                return rg_results
        return self._search_with_python(query, limit)

    @staticmethod
    def _validate_query(query: str) -> str:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty")
        if len(query) > 512:
            raise ValueError("query is too long")
        if any(character in query for character in ("\x00", "\r", "\n")):
            raise ValueError("query cannot contain control characters")
        return query

    def _search_with_rg(
        self, query: str, limit: int
    ) -> list[EngineeringSearchResult] | None:
        command = [
            str(self._rg_path),
            "--no-config",
            "--fixed-strings",
            "--line-number",
            "--column",
            "--with-filename",
            "--no-heading",
            "--color=never",
            "--max-columns=4096",
            "--max-columns-preview",
            f"--max-count={limit}",
            # ripgrep interprets a suffix-free value as bytes. Its accepted
            # suffixes are K/M/G; a trailing "B" is not portable.
            f"--max-filesize={self.max_file_size_bytes}",
        ]
        if not self.case_sensitive:
            command.append("--ignore-case")
        for directory in sorted(self.excluded_dirs):
            command.extend(("--glob", f"!{directory}/**"))
        for pattern in self.allowed_file_patterns:
            command.extend(("--glob", pattern))
        # ``--`` ensures a query beginning with '-' is always data, not an arg.
        command.extend(("--", query, "."))

        kwargs: dict[str, object] = {
            "cwd": str(self.repo_root),
            "capture_output": True,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "timeout": self.timeout_seconds,
            "shell": False,
            "check": False,
        }
        if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        try:
            completed = subprocess.run(command, **kwargs)
        except (OSError, subprocess.SubprocessError):
            return None

        # ripgrep uses 1 for a valid search with no matches.
        if completed.returncode == 1:
            return []
        if completed.returncode != 0:
            return None

        results: list[EngineeringSearchResult] = []
        for raw_line in completed.stdout.splitlines():
            match = _RG_LINE.match(raw_line)
            if not match:
                continue
            relative, line_number, column, content = match.groups()
            candidate = self._bounded_file(relative)
            if candidate is None:
                continue
            results.append(
                self._make_result(
                    candidate,
                    int(line_number),
                    int(column),
                    content,
                    rank=len(results) + 1,
                    retriever="live_code_rg",
                )
            )
            if len(results) >= limit:
                break
        return results

    def _search_with_python(self, query: str, limit: int) -> list[EngineeringSearchResult]:
        needle = query if self.case_sensitive else query.casefold()
        results: list[EngineeringSearchResult] = []

        for candidate in self._candidate_files():
            try:
                if candidate.stat().st_size > self.max_file_size_bytes:
                    continue
                raw = candidate.read_bytes()
            except OSError:
                continue
            if b"\x00" in raw[:4096]:
                continue
            text = raw.decode("utf-8", errors="replace")

            for line_number, line in enumerate(text.splitlines(), start=1):
                haystack = line if self.case_sensitive else line.casefold()
                column = haystack.find(needle)
                if column < 0:
                    continue
                results.append(
                    self._make_result(
                        candidate,
                        line_number,
                        column + 1,
                        line,
                        rank=len(results) + 1,
                        retriever="live_code_python",
                    )
                )
                if len(results) >= limit:
                    return results
        return results

    def _candidate_files(self) -> list[Path]:
        """List tracked or non-ignored source files without following links."""

        try:
            completed = subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.repo_root),
                    "ls-files",
                    "--cached",
                    "--others",
                    "--exclude-standard",
                    "-z",
                ],
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
                shell=False,
                env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
            )
        except (OSError, subprocess.SubprocessError):
            completed = None
        paths: list[Path] = []
        if completed is not None and completed.returncode == 0:
            for raw in completed.stdout.split(b"\0"):
                if not raw:
                    continue
                relative = raw.decode("utf-8", errors="replace").replace("\\", "/")
                if any(part in self.excluded_dirs for part in Path(relative).parts):
                    continue
                if not self._allowed_name(Path(relative).name):
                    continue
                candidate = self._bounded_file(relative)
                if candidate is not None and not candidate.is_symlink():
                    paths.append(candidate)
            return sorted(set(paths))

        for directory, dir_names, file_names in os.walk(self.repo_root, followlinks=False):
            dir_names[:] = sorted(
                name
                for name in dir_names
                if name not in self.excluded_dirs and not name.startswith(".")
            )
            for file_name in sorted(file_names):
                if file_name.startswith(".") or not self._allowed_name(file_name):
                    continue
                candidate = self._bounded_file(Path(directory) / file_name)
                if candidate is not None and not candidate.is_symlink():
                    paths.append(candidate)
        return paths

    def _allowed_name(self, file_name: str) -> bool:
        return any(
            fnmatch.fnmatchcase(file_name, pattern)
            for pattern in self.allowed_file_patterns
        )

    def _bounded_file(self, path: str | Path) -> Path | None:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.repo_root / candidate
        try:
            resolved = candidate.resolve()
            resolved.relative_to(self.repo_root)
        except (OSError, ValueError):
            return None
        if not resolved.is_file():
            return None
        return resolved

    def _make_result(
        self,
        path: Path,
        line_number: int,
        column: int,
        content: str,
        *,
        rank: int,
        retriever: str,
    ) -> EngineeringSearchResult:
        relative = path.relative_to(self.repo_root)
        return EngineeringSearchResult(
            content=content.strip(),
            source=str(path),
            score=1.0 / rank,
            corpus=self.corpus,
            authority=self.authority,
            line_start=line_number,
            line_end=line_number,
            retriever=retriever,
            metadata={
                "repo_root": str(self.repo_root),
                "relative_path": str(relative),
                "column": column,
                "live_verification": True,
            },
        )

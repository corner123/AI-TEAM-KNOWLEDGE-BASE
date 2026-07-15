"""Read-only live Git revision and history verification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import subprocess

from .models import EngineeringSearchResult


@dataclass(frozen=True, slots=True)
class GitWorktreeState:
    commit_sha: str
    branch: str
    commit_time: str
    dirty: bool

    def to_dict(self) -> dict[str, str | bool]:
        return asdict(self)


class LiveGitVerifier:
    """Inspect a fixed worktree using only non-mutating Git commands."""

    def __init__(self, repo_root: str | Path, *, timeout_seconds: float = 5.0) -> None:
        root = Path(repo_root).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"repo_root is not a directory: {root}")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.repo_root = root
        self.timeout_seconds = float(timeout_seconds)
        self._git("rev-parse", "--is-inside-work-tree")

    def state(self) -> GitWorktreeState:
        status = self._git("status", "--porcelain", "--untracked-files=normal")
        return GitWorktreeState(
            commit_sha=self._git("rev-parse", "HEAD").strip(),
            branch=self._git("rev-parse", "--abbrev-ref", "HEAD").strip(),
            commit_time=self._git("show", "-s", "--format=%cI", "HEAD").strip(),
            dirty=bool(status.strip()),
        )

    def path_state(
        self,
        relative_path: str | Path,
        *,
        include_worktree_state: bool = True,
    ) -> dict[str, object]:
        path = Path(relative_path)
        if path.is_absolute():
            raise ValueError("path must be relative to repo_root")
        resolved = (self.repo_root / path).resolve()
        try:
            bounded = resolved.relative_to(self.repo_root).as_posix()
        except ValueError as exc:
            raise ValueError("path escapes repo_root") from exc
        status = self._git("status", "--short", "--untracked-files=all", "--", bounded).strip()
        last_commit = self._git(
            "log", "-1", "--format=%H%x09%cI%x09%s", "--", bounded
        ).strip()
        result: dict[str, object] = {
            "relative_path": bounded,
            "exists": resolved.exists(),
            "status": status,
            "last_commit": last_commit,
        }
        if include_worktree_state:
            result.update(self.state().to_dict())
        return result

    def search(self, query: str, top_k: int = 5) -> list[EngineeringSearchResult]:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        query = query.strip()
        if not query or top_k <= 0:
            return []
        if len(query) > 512 or any(char in query for char in ("\x00", "\r", "\n")):
            raise ValueError("invalid Git history query")
        state = self.state()
        raw = self._git(
            "log",
            f"--max-count={min(int(top_k), 50)}",
            "--all",
            "--regexp-ignore-case",
            "--fixed-strings",
            f"--grep={query}",
            "--format=%H%x09%cI%x09%s",
        )
        results: list[EngineeringSearchResult] = []
        for rank, line in enumerate(raw.splitlines(), start=1):
            parts = line.split("\t", 2)
            if len(parts) != 3:
                continue
            sha, committed_at, subject = parts
            results.append(
                EngineeringSearchResult(
                    content=f"Commit {sha}: {subject} ({committed_at})",
                    source=f"git:{sha}",
                    score=1.0 / rank,
                    corpus="internal",
                    authority="history",
                    retriever="live_git",
                    metadata={
                        "commit_sha": sha,
                        "committed_at": committed_at,
                        "current_revision": state.to_dict(),
                        "live_verification": True,
                    },
                )
            )
        return results

    def _git(self, *args: str) -> str:
        kwargs: dict[str, object] = {
            "cwd": str(self.repo_root),
            "capture_output": True,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "timeout": self.timeout_seconds,
            "shell": False,
            "check": False,
            # Git may otherwise refresh stat data in .git/index even for
            # status-like commands.  Live verification must be observational.
            "env": {**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        }
        if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        completed = subprocess.run(["git", *args], **kwargs)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"read-only git command failed: {detail}")
        return completed.stdout

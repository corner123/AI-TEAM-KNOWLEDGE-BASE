from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from rag_core.retrieval.engineering import LiveASTRetriever, LiveCodeRetriever, LiveGitVerifier


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, check=True,
        text=True, encoding="utf-8",
    )
    return completed.stdout.strip()


def _repo(root: Path) -> Path:
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.name", "Verification Test")
    _git(root, "config", "user.email", "verification@example.invalid")
    (root / "agent.py").write_text(
        "class QueryEngine:\n"
        "    async def submit_message(self, message: str):\n"
        "        return message\n",
        encoding="utf-8",
    )
    _git(root, "add", "agent.py")
    _git(root, "commit", "-m", "add query engine")
    return root


def test_live_ast_returns_current_symbol_and_exact_lines(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    result = LiveASTRetriever(repo).search("QueryEngine.submit_message", top_k=3)[0]

    assert result.symbol == "QueryEngine.submit_message"
    assert result.metadata["live_verification"] is True
    assert result.metadata["relative_path"] == "agent.py"
    assert result.line_start == 2
    assert "async def submit_message" in result.content

    (repo / "agent.py").write_text(
        "class QueryEngine:\n"
        "    async def resume(self):\n"
        "        return None\n",
        encoding="utf-8",
    )
    assert LiveASTRetriever(repo).search("QueryEngine.resume")[0].symbol == "QueryEngine.resume"
    # The same retriever instance invalidates its cached parse on mtime/size.
    retriever = LiveASTRetriever(repo)
    assert retriever.search("QueryEngine.resume")
    (repo / "agent.py").write_text("def replacement():\n    return 1\n", encoding="utf-8")
    assert retriever.search("QueryEngine.resume") == []
    assert retriever.search("replacement")[0].symbol == "replacement"


def test_live_ast_does_not_match_parent_of_missing_qualified_symbol(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    retriever = LiveASTRetriever(repo)

    assert retriever.search("QueryEngine.definitely_missing_method") == []


def test_live_code_python_fallback_ignores_gitignored_secret_files(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    (repo / ".gitignore").write_text(".env\n", encoding="utf-8")
    (repo / ".env").write_text("SECRET_MARKER=do-not-return\n", encoding="utf-8")
    (repo / "config.py").write_text("SAFE_MARKER = True\n", encoding="utf-8")

    retriever = LiveCodeRetriever(repo, prefer_rg=False)

    assert retriever.search("do-not-return") == []
    assert retriever.search("SAFE_MARKER")[0].metadata["relative_path"] == "config.py"


def test_live_git_reports_revision_dirty_path_and_history(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    verifier = LiveGitVerifier(repo)
    clean = verifier.state()
    assert len(clean.commit_sha) == 40
    assert clean.dirty is False
    assert verifier.search("query engine", top_k=2)[0].authority == "history"

    (repo / "agent.py").write_text("# changed\n", encoding="utf-8")
    dirty = verifier.path_state("agent.py")
    assert dirty["dirty"] is True
    assert dirty["status"].startswith("M")


def test_live_git_rejects_paths_outside_repository(tmp_path: Path) -> None:
    verifier = LiveGitVerifier(_repo(tmp_path / "repo"))
    with pytest.raises(ValueError, match="escapes"):
        verifier.path_state("../outside.txt")

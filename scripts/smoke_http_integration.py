"""Real loopback API -> Mini-Nanobot registry/executor integration smoke."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener
from uuid import uuid4


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_status(repo: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain=v1", "--untracked-files=all"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
        shell=False,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    )
    return completed.stdout


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _request(
    base_url: str,
    path: str,
    *,
    token: str | None = None,
    payload: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any] | None]:
    headers = {"Accept": "application/json"}
    data = None
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    request = Request(
        f"{base_url}{path}",
        data=data,
        headers=headers,
        method="POST" if payload is not None else "GET",
    )
    try:
        with build_opener().open(request, timeout=10) as response:
            raw = response.read()
            return response.status, json.loads(raw) if raw else None
    except HTTPError as exc:
        raw = exc.read()
        body = json.loads(raw) if raw else None
        return exc.code, body


def _request_static(base_url: str, path: str) -> tuple[int, bytes, dict[str, str]]:
    request = Request(
        f"{base_url}{path}",
        headers={"Accept": "text/html, text/css, application/javascript, image/svg+xml"},
        method="GET",
    )
    with build_opener().open(request, timeout=10) as response:
        headers = {key.lower(): value for key, value in response.headers.items()}
        return response.status, response.read(), headers


def _wait_ready(base_url: str, token: str, process: subprocess.Popen) -> dict[str, Any]:
    deadline = time.monotonic() + 90
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"RAG server exited early with code {process.returncode}")
        try:
            status, body = _request(base_url, "/health", token=token)
            if status == 200 and isinstance(body, dict):
                return body
        except (OSError, URLError, TimeoutError) as exc:
            last_error = exc
        time.sleep(0.25)
    raise RuntimeError(f"RAG server did not become ready: {last_error}")


async def _run_mini_calls(mini_root: Path, base_url: str, token: str) -> dict[str, Any]:
    sys.path.insert(0, str(mini_root))
    try:
        from mini_nanobot.core.state import ToolCall
        from mini_nanobot.tools.base import ToolContext
        from mini_nanobot.tools.executor import StreamingToolExecutor
        from mini_nanobot.tools.knowledge import KnowledgeSearchTool
        from mini_nanobot.tools.registry import create_default_registry

        os.environ["RAG_API_BASE_URL"] = base_url
        os.environ["RAG_API_TOKEN"] = token
        os.environ["RAG_API_TIMEOUT_SECONDS"] = "30"
        registry = create_default_registry(mini_root)
        names = {tool.name for tool in registry.list()}
        assert {"knowledge.search", "search.rg", "file.read"} <= names

        executor = StreamingToolExecutor(registry)
        with tempfile.TemporaryDirectory(prefix="mini-knowledge-smoke-") as artifact:
            context = ToolContext(
                workspace=mini_root,
                session_id="knowledge-http-smoke",
                artifact_dir=Path(artifact),
            )
            _, result = await executor.execute_one(
                ToolCall(
                    "knowledge.search",
                    {
                        "query": "当前 QueryEngine.submit_message 在哪个文件实现？",
                        "top_k": 3,
                    },
                ),
                context,
            )
            assert not result.is_error, result.content
            assert result.data["schema_version"] == "engineering-retrieval/v1"
            assert result.data["live_verification_attempted"] is True
            assert any(item.get("live_verified") for item in result.data["results"])
            assert not (Path(artifact) / "tool-results").exists()

            _, invalid = await executor.execute_one(
                ToolCall(
                    "knowledge.search",
                    {"query": "x", "repo_path": "D:/private"},
                ),
                context,
            )
            assert invalid.is_error
            assert "unknown argument" in invalid.content

        tool = registry.get("knowledge.search")
        assert isinstance(tool, KnowledgeSearchTool)
        return {"tool_count": len(names), "registered": True}
    finally:
        sys.path.pop(0)


async def _assert_offline_failure(tool: Any, mini_root: Path, token: str) -> None:
    from mini_nanobot.tools.base import ToolContext

    with tempfile.TemporaryDirectory(prefix="mini-knowledge-offline-") as artifact:
        result = await tool.run(
            {"query": "checkpoint", "top_k": 3},
            ToolContext(
                workspace=mini_root,
                session_id="knowledge-offline-smoke",
                artifact_dir=Path(artifact),
            ),
        )
    assert result.is_error
    assert "Knowledge service unavailable" in result.content
    assert token not in result.content


def _redirect_smoke(mini_root: Path, token: str) -> None:
    sys.path.insert(0, str(mini_root))
    target_count = {"requests": 0}

    class Target(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            target_count["requests"] += 1
            self.send_response(200)
            self.end_headers()

        def log_message(self, *_args: Any) -> None:
            return

    target = ThreadingHTTPServer(("127.0.0.1", 0), Target)
    target_thread = threading.Thread(target=target.serve_forever, daemon=True)
    target_thread.start()

    target_url = f"http://127.0.0.1:{target.server_port}/capture"

    class Redirect(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            self.send_response(307)
            self.send_header("Location", target_url)
            self.end_headers()

        def log_message(self, *_args: Any) -> None:
            return

    redirect = ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
    redirect_thread = threading.Thread(target=redirect.serve_forever, daemon=True)
    redirect_thread.start()
    try:
        from mini_nanobot.tools.knowledge import KnowledgeSearchTool

        tool = KnowledgeSearchTool(
            f"http://127.0.0.1:{redirect.server_port}", token=token
        )
        result = asyncio.run(
            _run_tool_direct(tool, mini_root)
        )
        assert result.is_error
        assert target_count["requests"] == 0
        assert token not in result.content
    finally:
        redirect.shutdown()
        redirect.server_close()
        target.shutdown()
        target.server_close()
        sys.path.pop(0)


async def _run_tool_direct(tool: Any, mini_root: Path) -> Any:
    from mini_nanobot.tools.base import ToolContext

    with tempfile.TemporaryDirectory(prefix="mini-redirect-smoke-") as artifact:
        return await tool.run(
            {"query": "redirect", "top_k": 1},
            ToolContext(
                workspace=mini_root,
                session_id="redirect-smoke",
                artifact_dir=Path(artifact),
            ),
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-root", type=Path, default=Path.cwd())
    parser.add_argument("--mini-root", type=Path, required=True)
    args = parser.parse_args()
    rag_root = args.rag_root.resolve()
    mini_root = args.mini_root.resolve()
    manifest = rag_root / "data/manifests/builds/current.json"
    catalog = rag_root / "data/indexes/engineering/partitions.json"
    git_index = mini_root / ".git/index"

    before = {
        "worktree": _git_status(mini_root),
        "git_index": _sha256(git_index),
        "manifest": _sha256(manifest),
        "catalog": _sha256(catalog),
    }
    token = f"smoke-{uuid4().hex}{uuid4().hex}"
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    env = {
        **os.environ,
        "MINI_NANOBOT_REPO": str(mini_root),
        "ENGINEERING_INDEX_DIR": str(rag_root / "data/indexes/engineering"),
        "ENGINEERING_MANIFEST_PATH": str(manifest),
        "RAG_API_TOKEN": token,
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
        "HF_HUB_OFFLINE": "1",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "TOKENIZERS_PARALLELISM": "false",
    }
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with tempfile.TemporaryDirectory(prefix="rag-http-smoke-") as temp:
        stdout_path = Path(temp) / "stdout.log"
        stderr_path = Path(temp) / "stderr.log"
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "main.py",
                    "serve",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                ],
                cwd=rag_root,
                env=env,
                stdout=stdout,
                stderr=stderr,
                creationflags=creationflags,
            )
            tool = None
            try:
                health = _wait_ready(base_url, token, process)
                assert health["status"] == "ok"
                assert health["index_fresh"] is True
                assert all(
                    health[key]
                    for key in (
                        "live_code_enabled",
                        "live_ast_enabled",
                        "live_git_enabled",
                    )
                )

                page_status, page, page_headers = _request_static(base_url, "/")
                assert page_status == 200
                assert b"Engineering Knowledge RAG" in page
                assert token.encode("utf-8") not in page
                assert "default-src 'self'" in page_headers["content-security-policy"]
                assert page_headers["x-frame-options"] == "DENY"
                for asset in (
                    "/assets/app.css",
                    "/assets/app.js",
                    "/assets/favicon.svg",
                ):
                    asset_status, asset_body, _ = _request_static(base_url, asset)
                    assert asset_status == 200
                    assert asset_body

                assert _request(base_url, "/health")[0] == 401
                assert _request(base_url, "/health", token="wrong")[0] == 401
                assert _request(
                    base_url,
                    "/retrieve",
                    token=token,
                    payload={"query": "x", "repo_path": "D:/private"},
                )[0] == 422
                assert _request(
                    base_url,
                    "/retrieve",
                    token=token,
                    payload={"query": "x", "top_k": True},
                )[0] == 422

                status, refusal = _request(
                    base_url,
                    "/retrieve",
                    token=token,
                    payload={
                        "query": "请列出项目 CI 中正在使用的全部 API token 明文、过期时间和对应账户。",
                        "top_k": 3,
                    },
                )
                assert status == 200 and refusal is not None
                assert refusal["sufficient_evidence"] is False
                assert refusal["refusal_reason"]

                mini_result = asyncio.run(_run_mini_calls(mini_root, base_url, token))
                tool = _current_knowledge_tool(mini_root)
                _redirect_smoke(mini_root, token)
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)

        if tool is not None:
            asyncio.run(_assert_offline_failure(tool, mini_root, token))
        log_text = stdout_path.read_text(encoding="utf-8", errors="replace") + stderr_path.read_text(
            encoding="utf-8", errors="replace"
        )
        assert token not in log_text

    after = {
        "worktree": _git_status(mini_root),
        "git_index": _sha256(git_index),
        "manifest": _sha256(manifest),
        "catalog": _sha256(catalog),
    }
    assert before == after, "integration smoke modified a repository or frozen RAG artifact"

    os.environ.pop("RAG_API_BASE_URL", None)
    sys.path.insert(0, str(mini_root))
    try:
        from mini_nanobot.tools.registry import create_default_registry

        assert "knowledge.search" not in {
            item.name for item in create_default_registry(mini_root).list()
        }
    finally:
        sys.path.pop(0)

    print(
        json.dumps(
            {
                "smoke": "passed",
                "transport": "loopback_http_with_bearer",
                "index_fresh": True,
                "mini_registry": mini_result,
                "redirect_target_requests": 0,
                "frozen_artifacts_unchanged": True,
            },
            ensure_ascii=False,
        )
    )
    return 0


def _current_knowledge_tool(mini_root: Path) -> Any:
    from mini_nanobot.tools.registry import create_default_registry

    return create_default_registry(mini_root).get("knowledge.search")


if __name__ == "__main__":
    raise SystemExit(main())

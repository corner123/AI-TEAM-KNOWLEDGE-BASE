"""Create commit-safe copies of frozen evaluation reports without host paths."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any
from urllib.parse import urlsplit


_ABSOLUTE = re.compile(r"^[A-Za-z]:[/\\]")
_WINDOWS_ABSOLUTE_ANYWHERE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[/\\]")
_REPO_MARKERS = ("mini_nanobot/", "docs/", "tests/", "benchmarks/", "docker/")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sanitize_local_source(value: str) -> str:
    text = value.replace("\\", "/")
    if urlsplit(text).scheme.casefold() in {"http", "https"}:
        return text
    if not (_ABSOLUTE.match(text) or text.startswith("/")):
        return text
    lower = text.casefold()
    for marker in _REPO_MARKERS:
        needle = "/" + marker.casefold()
        position = lower.rfind(needle)
        if position >= 0:
            return text[position + 1 :]
    name = PurePosixPath(text).name
    if name in {"README.md", "pyproject.toml"}:
        return name
    raise ValueError(f"cannot safely publish an unrecognized absolute source: {name}")


def _sanitize(value: Any, *, key: str = "") -> Any:
    if isinstance(value, dict):
        return {str(name): _sanitize(item, key=str(name)) for name, item in value.items()}
    if isinstance(value, list):
        return [_sanitize(item, key=key) for item in value]
    if isinstance(value, str) and key.endswith("sources"):
        return _sanitize_local_source(value)
    if isinstance(value, str) and (_ABSOLUTE.match(value) or value.startswith("/")):
        raise ValueError(f"absolute host path remains in report field {key!r}")
    return value


def _assert_no_host_path(value: str, *, field: str) -> None:
    """Refuse publication when a rendered artifact still exposes a host path."""

    if _WINDOWS_ABSOLUTE_ANYWHERE.search(value):
        raise ValueError(f"absolute host path remains in {field}")


def _is_frozen_holdout_first_run(payload: dict[str, Any], input_json: Path) -> bool:
    roles = payload.get("run_metadata", {}).get("dataset_roles", ())
    return "holdout" in roles and "first_run" in input_json.stem.casefold()


def publish(input_json: Path, output_dir: Path) -> tuple[Path, Path]:
    payload = json.loads(input_json.read_text(encoding="utf-8"))
    sanitized = _sanitize(payload)
    frozen_holdout_first_run = _is_frozen_holdout_first_run(payload, input_json)
    sanitized["publication_metadata"] = {
        "path_sanitized": True,
        "raw_report_name": input_json.name,
        "raw_report_sha256": _sha256(input_json),
        "metrics_unchanged": True,
        "frozen_holdout_first_run": frozen_holdout_first_run,
    }
    json_text = json.dumps(sanitized, ensure_ascii=False, indent=2) + "\n"
    _assert_no_host_path(json_text, field="public JSON")

    raw_markdown = input_json.with_suffix(".md")
    raw_markdown_text = raw_markdown.read_text(encoding="utf-8")
    _assert_no_host_path(raw_markdown_text, field="raw Markdown")
    banner = (
        "> Public copy: local absolute paths were converted to repository-relative "
        "sources; metrics and per-question decisions are unchanged.\n\n"
    )
    if frozen_holdout_first_run:
        banner += (
            "> Frozen holdout: this is the first formal run and must not be replaced "
            "after tuning against its failures.\n\n"
        )
    markdown_text = banner + raw_markdown_text

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / input_json.name
    json_path.write_text(json_text, encoding="utf-8")

    markdown_path = output_dir / raw_markdown.name
    markdown_path.write_text(markdown_text, encoding="utf-8")
    return json_path, markdown_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/eval/reports_public"))
    args = parser.parse_args()
    for input_path in args.inputs:
        json_path, markdown_path = publish(input_path, args.output_dir)
        print(f"published {json_path} and {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

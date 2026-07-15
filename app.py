"""Convenience launcher for the engineering-knowledge RAG web application."""

from __future__ import annotations

import os

from dotenv import load_dotenv

from main import main


load_dotenv(override=False)


def _port() -> int:
    raw = os.getenv("RAG_UI_PORT", "8000").strip() or "8000"
    try:
        port = int(raw)
    except ValueError as exc:
        raise SystemExit("RAG_UI_PORT must be an integer") from exc
    if not 1 <= port <= 65_535:
        raise SystemExit("RAG_UI_PORT must be between 1 and 65535")
    return port


if __name__ == "__main__":
    raise SystemExit(
        main(["serve", "--host", "127.0.0.1", "--port", str(_port())])
    )

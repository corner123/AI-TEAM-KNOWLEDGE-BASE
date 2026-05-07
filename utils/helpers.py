import time
import uuid
from contextlib import contextmanager
from typing import Optional


class Timer:
    def __init__(self, name: str = ""):
        self.name = name
        self.elapsed: float = 0
        self._start: float = 0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start


def generate_id(prefix: str = "") -> str:
    short_id = uuid.uuid4().hex[:8]
    return f"{prefix}_{short_id}" if prefix else short_id


def truncate_text(text: str, max_length: int = 500) -> str:
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."

"""Append-only JSONL progress stream.

Every line is fsync'd so the backend's tailer sees events immediately. The format
is one JSON object per line; the backend parses by splitting on newlines.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class ProgressWriter:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", buffering=1)  # line-buffered

    def emit(self, event: str, **fields: Any) -> None:
        payload = {"event": event, **fields}
        self._fh.write(json.dumps(payload, default=str) + "\n")
        self._fh.flush()
        try:
            os.fsync(self._fh.fileno())
        except OSError:
            pass  # non-fatal — only an optimisation for the tailer

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self) -> "ProgressWriter":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

"""Append-only JSONL progress stream.

Every line is fsync'd so the backend's tailer sees events immediately. The format
is one JSON object per line; the backend parses by splitting on newlines.
"""
from __future__ import annotations

import json
import os
import sys
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
        if os.environ.get("ML_PROGRESS_ECHO"):
            print(_human(payload), file=sys.stderr, flush=True)

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self) -> "ProgressWriter":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def _num(v: Any) -> str:
    return f"{v:+.4f}" if isinstance(v, (int, float)) else "—"


def _human(p: dict) -> str:
    """One readable terminal line per event (enabled by ML_PROGRESS_ECHO=1)."""
    ev = p.get("event")
    if ev == "trial":
        return f"  · {p.get('symbol'):<12} trial {p.get('trial'):>4}  score {_num(p.get('value'))}  {p.get('state')}"
    if ev == "symbol_done":
        return f"✔ {p.get('symbol'):<12} done   best {_num(p.get('best_value'))}"
    if ev == "run_start":
        return (f"▶ {p.get('model_id')}: {p.get('symbols_total')} symbols × {p.get('trials')} trials "
                f"({p.get('preset')}, {p.get('dataset')}, {p.get('window')})")
    if ev == "run_done":
        return f"■ artifact written → {p.get('artifact')}"
    if ev == "error":
        return f"✘ {p.get('symbol') or ''} {p.get('message')}"
    return json.dumps(p, default=str)

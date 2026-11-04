"""Model artifact builder.

Shape matches `docs/backend/ml-pipeline.md`:

    {
      "version": "1",
      "id": "dev_signal_ml1",
      "label": "Dev Signal ML1",
      "created_at": "2026-...",
      "training": {...},
      "per_symbol": {SYM: {"params": {...}, "cv": {...}, "oos": {...}}, ...},
      "global_fallback": {"params": {...}}
    }

Writes atomically: .tmp → os.replace.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.integrations.kite.session import IST


def build(
    *,
    model_id: str,
    label: str,
    training: dict,
    per_symbol: dict[str, dict],
    global_fallback: dict,
) -> dict[str, Any]:
    return {
        "version": "1",
        "id": model_id,
        "label": label,
        "created_at": datetime.now(IST).isoformat(),
        "training": training,
        "per_symbol": per_symbol,
        "global_fallback": global_fallback,
    }


def _validate(payload: dict) -> None:
    required = ("version", "id", "label", "created_at", "training",
                "per_symbol", "global_fallback")
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"artifact missing required keys: {missing}")
    if not isinstance(payload["per_symbol"], dict):
        raise ValueError("per_symbol must be a dict")
    gf = payload["global_fallback"]
    if "params" not in gf:
        raise ValueError("global_fallback.params is required")


def write_atomic(payload: dict, path: Path) -> Path:
    _validate(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(tmp, path)
    return path


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

"""End-to-end smoke test: 10 trials on 1 symbol, Fast preset, tiny window.

This is the strongest guardrail against backend-import drift — if `app.*` moves,
this breaks loudly. Skips automatically when no offline dataset is on disk.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.config import settings
from app.integrations.offline.store import DEFAULT_DATASET, dataset_dir

from stock_terminal_ml.train import main
from stock_terminal_ml.search_space import TUNED_KEYS


def _pick_symbol() -> str | None:
    try:
        root = dataset_dir(DEFAULT_DATASET)
    except ValueError:
        return None
    if not root.is_dir():
        return None
    for p in sorted(root.glob("*.json")):
        if p.name == "manifest.json":
            continue
        try:
            doc = json.loads(p.read_text())
            if doc.get("symbol") and doc.get("candles"):
                return doc["symbol"]
        except Exception:
            continue
    return None


def test_smoke(tmp_path: Path) -> None:
    sym = _pick_symbol()
    if not sym:
        pytest.skip(f"no offline dataset on disk ({settings.offline_data_dir / DEFAULT_DATASET}); "
                    "smoke test did NOT run")

    out = tmp_path / "dev_signal_ml_smoke.json"
    rc = main([
        "--window", "2026-03-01:2026-05-31",
        "--preset", "fast",
        "--trials", "10",
        "--symbols", sym,
        "--output", str(out),
        "--jobs", "1",
        "--model-id", "dev_signal_ml_smoke",
        "--label", "Smoke",
    ])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["id"] == "dev_signal_ml_smoke"
    assert payload["version"] == "1"
    assert sym in payload["per_symbol"], f"{sym} missing from per_symbol"
    row = payload["per_symbol"][sym]
    assert "cv" in row and "oos" in row and "params" in row
    for k in TUNED_KEYS:
        assert k in payload["global_fallback"]["params"], f"fallback missing {k}"

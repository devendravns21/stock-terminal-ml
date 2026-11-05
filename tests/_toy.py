"""Shared fixtures: a tiny hand-trained LightGBM booster wrapped in a v2 artifact.

The toy model has one feature `x` and learns y == x, so P(profit) is high for x=1
and low for x=0. Tests patch `features.extract_features` to emit `x = bar_index % 2`
so roughly half of the Dev Signal fires get kept at threshold 0.5.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from app.integrations.offline.store import DEFAULT_DATASET, dataset_dir

TOY_FEATURE = "x"


def toy_booster_str() -> str:
    X = np.array([[0.0]] * 20 + [[1.0]] * 20)
    y = np.array([0] * 20 + [1] * 20)
    ds = lgb.Dataset(X, label=y, feature_name=[TOY_FEATURE],
                     params={"min_data_in_bin": 1, "verbose": -1})
    booster = lgb.train({"objective": "binary", "num_leaves": 2, "min_data_in_leaf": 1,
                         "learning_rate": 0.5, "verbose": -1, "seed": 42},
                        ds, num_boost_round=5)
    return booster.model_to_string()


def toy_artifact(model_id: str = "dev_signal_ml_toy", threshold: float = 0.5,
                 feature_names: list[str] | None = None, **overrides) -> dict:
    art = {
        "version": "2",
        "kind": "meta_labeling",
        "id": model_id,
        "label": "Toy ML",
        "created_at": "2026-11-05T10:00:00+05:30",
        "primary": {"params": {"timeframe": "3m"}},
        "model": {
            "framework": "lightgbm",
            "objective": "binary",
            "booster": toy_booster_str(),
            "decision_threshold": threshold,
            "calibration": {"x": [0.0, 0.5, 1.0], "y": [0.0, 0.5, 1.0]},
        },
        "features": {"spec_version": "toy", "names": feature_names or [TOY_FEATURE], "categorical": []},
        "training": {"dataset": DEFAULT_DATASET, "window": {"from": "2026-01-01", "to": "2026-06-30"}},
        "evaluation": {"oos": {"folds": [], "aggregate": {}}},
        "per_symbol": {},
        "global_fallback": {"decision_threshold": threshold},
    }
    art.update(overrides)
    return art


def parity_features(**kw) -> dict:
    return {TOY_FEATURE: float(kw["i"] % 2)}


def pick_symbol() -> str | None:
    """First equity symbol in the default offline dataset, or None if no dataset."""
    try:
        root = dataset_dir(DEFAULT_DATASET)
    except ValueError:
        return None
    if not root.is_dir():
        return None
    for p in sorted(root.glob("*.json")):
        if p.name in ("manifest.json", "NIFTY_50.json"):
            continue
        try:
            doc = json.loads(p.read_text())
        except Exception:
            continue
        if doc.get("symbol") and doc.get("candles"):
            return doc["symbol"]
    return None


def write_model(models_dir: Path, artifact: dict) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    p = models_dir / f"{artifact['id']}.json"
    p.write_text(json.dumps(artifact))
    return p

"""Task 7.3 — v2 artifact build → write → reload → validate; tampering raises."""
import copy
import json

import pytest

from app.integrations.ml_meta.schema import is_v2, validate_v2
from stock_terminal_ml import artifact

from ._toy import toy_artifact


def _build():
    a = toy_artifact()
    return artifact.build_v2(
        model_id=a["id"], label=a["label"], primary=a["primary"], model=a["model"],
        features=a["features"], training=a["training"], evaluation=a["evaluation"],
        per_symbol={"RELIANCE": {"threshold_override": 0.6}},
        global_fallback=a["global_fallback"],
    )


def test_round_trip(tmp_path):
    payload = _build()
    assert payload["version"] == "2" and payload["kind"] == "meta_labeling"
    out = artifact.write_atomic_v2(payload, tmp_path / "m.json")
    loaded = json.loads(out.read_text())
    validate_v2(loaded)
    assert is_v2(loaded)
    assert not (tmp_path / "m.json.tmp").exists()


@pytest.mark.parametrize("path", [
    ("model", "booster"), ("model", "decision_threshold"), ("features", "names"),
    ("global_fallback", "decision_threshold"), ("training",), ("id",), ("kind",),
])
def test_missing_required_key_raises(tmp_path, path):
    payload = copy.deepcopy(_build())
    d = payload
    for k in path[:-1]:
        d = d[k]
    del d[path[-1]]
    with pytest.raises(ValueError):
        validate_v2(payload)
    with pytest.raises(ValueError):
        artifact.write_atomic_v2(payload, tmp_path / "bad.json")
    assert not (tmp_path / "bad.json").exists()


def test_empty_booster_and_bad_calibration_raise():
    p = _build()
    p["model"]["booster"] = ""
    with pytest.raises(ValueError):
        validate_v2(p)
    p = _build()
    p["model"]["calibration"] = {"x": [0, 1], "y": [0.5]}
    with pytest.raises(ValueError):
        validate_v2(p)


def test_v1_is_not_v2():
    assert not is_v2({"version": "1", "id": "x"})

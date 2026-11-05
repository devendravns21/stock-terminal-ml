"""Task 7.4 — POST /api/ml/score with a toy v2 model on one symbol (offline data)."""
import pytest

from app.integrations.ml import registry
from app.integrations.ml_meta import features

from ._toy import parity_features, pick_symbol, toy_artifact, write_model

RANGE = {"from": "2026-10-01", "to": "2026-10-31", "dataset": "nifty50_3m_ytd"}


@pytest.fixture
def toy_model(models_dir):
    write_model(models_dir, toy_artifact())
    registry.notify_models_changed()
    return "dev_signal_ml_toy"


def test_score_returns_markers(client, toy_model, monkeypatch):
    sym = pick_symbol()
    if not sym:
        pytest.skip("no offline dataset on disk")
    monkeypatch.setattr(features, "extract_features", parity_features)
    r = client.post("/api/ml/score", json={"model_id": toy_model, "symbol": sym, **RANGE})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model_id"] == toy_model and body["symbol"] == sym
    assert body["threshold"] == 0.5
    assert 0 < body["baseline_count"]
    assert body["kept_count"] <= body["baseline_count"]
    assert body["kept_count"] == len(body["markers"])
    for m in body["markers"]:
        assert {"time", "price", "direction", "kind", "line_name", "p_profit"} <= m.keys()
        assert m["p_profit"] >= body["threshold"]


def test_score_errors(client, toy_model, v1_model):
    ok = {"symbol": "RELIANCE", **RANGE}
    assert client.post("/api/ml/score", json={"model_id": "bad id!", **ok}).status_code == 422
    assert client.post("/api/ml/score", json={"model_id": "dev_signal_ml404", **ok}).status_code == 404
    r = client.post("/api/ml/score", json={"model_id": v1_model, **ok})
    assert r.status_code == 422 and "not a meta-labeling model" in r.json()["detail"]
    r = client.post("/api/ml/score", json={"model_id": toy_model, **ok, "timeframe": "15m"})
    assert r.status_code == 422


def test_models_lists_v1_and_v2(client, toy_model, v1_model):
    rows = {r["id"]: r for r in client.get("/api/ml/models").json()}
    assert rows[v1_model]["version"] == "1" and rows[v1_model]["kind"] == "params"
    assert rows[v1_model]["tuned_keys"]
    v2 = rows[toy_model]
    assert v2["version"] == "2" and v2["kind"] == "meta_labeling"
    assert v2["features"] == ["x"] and v2["tuned_keys"] == []
    assert "evaluation" in v2

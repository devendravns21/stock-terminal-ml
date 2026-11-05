"""Tasks 7.5 / 4.3 — v1 models still drive backtests; v2 models filter trades."""
import pytest

from app.integrations.backtest.jobs import JobInstrument
from app.integrations.backtest.router import BacktestIn, _build_batch_config, run_one
from app.integrations.ml import registry
from app.integrations.ml_meta import features
from app.integrations.offline.store import DEFAULT_DATASET

from ._toy import parity_features, pick_symbol, toy_artifact, write_model


def _run(sym, ml_model=None):
    body = BacktestIn(instruments=[{"exchange": "NSE", "name": sym}],
                      from_date="2026-09-01", to_date="2026-10-31",
                      offline_dataset=DEFAULT_DATASET, ml_model=ml_model,
                      costs={"enabled": True, "segment": "EQUITY_INTRADAY"})
    config = _build_batch_config(body)
    return run_one(None, JobInstrument(exchange="NSE", name=sym), config)


@pytest.fixture
def sym():
    s = pick_symbol()
    if not s:
        pytest.skip("no offline dataset on disk")
    return s


def test_v1_backtest_still_runs(v1_model, sym):
    res = _run(sym, v1_model)
    assert res.error is None, res.error
    assert res.params_source in ("symbol", "fallback")


def test_v2_backtest_filters_trades(models_dir, sym, monkeypatch):
    monkeypatch.setattr(features, "extract_features", parity_features)
    write_model(models_dir, toy_artifact("dev_signal_ml_half", threshold=0.5))
    write_model(models_dir, toy_artifact("dev_signal_ml_none", threshold=0.999))
    registry.notify_models_changed()

    base = _run(sym)
    half = _run(sym, "dev_signal_ml_half")
    none = _run(sym, "dev_signal_ml_none")
    for r in (base, half, none):
        assert r.error is None, r.error
    assert half.params_source == "meta_labeling" and none.params_source == "meta_labeling"
    assert base.metrics.num_trades > 0
    assert half.metrics.num_trades <= base.metrics.num_trades
    assert none.metrics.num_trades == 0 < base.metrics.num_trades


def test_v2_backtest_requires_offline(models_dir):
    from fastapi import HTTPException
    write_model(models_dir, toy_artifact())
    registry.notify_models_changed()
    body = BacktestIn(instruments=[{"exchange": "NSE", "name": "RELIANCE"}],
                      from_date="2026-09-01", to_date="2026-10-31", ml_model="dev_signal_ml_toy")
    with pytest.raises(HTTPException) as e:
        _build_batch_config(body)
    assert e.value.status_code == 422


def test_runner_aggregate_ignores_new_events():
    from app.integrations.ml.runner import _aggregate
    out = _aggregate([
        {"event": "run_start", "symbols_total": 3},
        {"event": "stage", "name": "features", "pct": 0.2},
        {"event": "fold", "index": 0, "metrics": {"pr_auc": 0.6}},
        {"event": "run_done", "artifact": "/tmp/x.json"},
    ])
    assert out["symbols_total"] == 3 and out["artifact"] == "/tmp/x.json"

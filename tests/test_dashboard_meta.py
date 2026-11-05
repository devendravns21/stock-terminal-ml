"""Task 6.3 — the dashboard writes a meta-mode config.json + run.sh (no terminal opened)."""
import json

import pytest

from stock_terminal_ml import dashboard

from ._toy import pick_symbol


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(dashboard, "_open_terminal", lambda script: "test")
    return tmp_path / "runs"


def _body(sym, **kw):
    return {"from": "2026-01-01", "to": "2026-10-31", "symbols": [sym], "publish": False, **kw}


def test_meta_run_writes_config(runs):
    sym = pick_symbol()
    if not sym:
        pytest.skip("no offline dataset on disk")
    out = dashboard.start_run(_body(sym, mode="meta", train_months=4, test_months=1,
                                    embargo_days=2, calibrate=False))
    cfg = json.loads((runs / out["run_id"] / "config.json").read_text())
    assert cfg["mode"] == "meta"
    assert (cfg["train_months"], cfg["test_months"], cfg["embargo_days"]) == (4, 1, 2)
    assert cfg["calibrate"] is False and cfg["model_family"] == "lightgbm"
    cmd = cfg["command"]
    assert cmd[1:4] == ["-m", "stock_terminal_ml.train_meta", "--mode"] and cmd[4] == "meta"
    assert "--no-calibrate" in cmd and "--preset" not in cmd
    assert "stock_terminal_ml.train_meta" in (runs / out["run_id"] / "run.sh").read_text()
    # The CLI accepts exactly the flags the dashboard generated.
    from stock_terminal_ml.train_meta import build_parser
    args = build_parser().parse_args(cmd[3:])
    assert args.mode == "meta" and args.train_months == 4 and args.calibrate is False


def test_params_run_unchanged(runs):
    sym = pick_symbol()
    if not sym:
        pytest.skip("no offline dataset on disk")
    out = dashboard.start_run(_body(sym, preset="fast"))
    cfg = json.loads((runs / out["run_id"] / "config.json").read_text())
    assert cfg["mode"] == "params" and "train_months" not in cfg
    assert cfg["command"][1:3] == ["-m", "stock_terminal_ml.train"] and "--preset" in cfg["command"]


def test_bad_meta_values_rejected(runs):
    with pytest.raises(dashboard.BadRequest):
        dashboard._validate(_body(pick_symbol() or "RELIANCE", mode="meta", train_months=0))

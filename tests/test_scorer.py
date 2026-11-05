"""Task 7.2 — MetaScorer scoring/thresholds and filtered_fires ⊆ run_strategy."""
import math

import pytest

from app.integrations.backtest.strategy import build_cpr_by_day, run_strategy
from app.integrations.ml_meta import features, offline as meta_offline
from app.integrations.ml_meta.scorer import MetaScorer, filtered_fires, params_from_primary, score_fires
from app.integrations.offline.store import DEFAULT_DATASET
from app.integrations.signals.engine import _ist_date
from datetime import date

from ._toy import parity_features, pick_symbol, toy_artifact


def test_score_fire_in_unit_interval_and_monotone():
    s = MetaScorer.from_artifact(toy_artifact())
    lo, hi = s.score_fire({"x": 0.0}), s.score_fire({"x": 1.0})
    assert 0.0 <= lo < hi <= 1.0
    assert s.decide(hi, "ANY") and not s.decide(lo, "ANY")
    # Missing feature -> NaN input, still a valid probability.
    assert 0.0 <= s.score_fire({}) <= 1.0


def test_feature_order_follows_artifact():
    s = MetaScorer.from_artifact(toy_artifact())
    s.feature_names = ["b", "a", "c"]
    v = s.vector({"a": 1, "b": 2, "junk": 9})
    assert v[:2] == [2.0, 1.0] and math.isnan(v[2])


def test_per_symbol_threshold_override():
    art = toy_artifact(per_symbol={"RELIANCE": {"threshold_override": 0.99}})
    s = MetaScorer.from_artifact(art)
    assert s.threshold_for("RELIANCE") == 0.99
    assert s.threshold_for("TCS") == 0.5
    p = s.score_fire({"x": 1.0})
    assert s.decide(p, "TCS") and not s.decide(p, "RELIANCE")


def test_invalid_artifact_raises():
    art = toy_artifact()
    del art["model"]["booster"]
    with pytest.raises(ValueError):
        MetaScorer.from_artifact(art)


def _load(sym):
    got = meta_offline.load_bars(DEFAULT_DATASET, sym, "3m", date(2026, 9, 1), date(2026, 10, 31))
    bars, _first, daily = got
    idx_bars, idx_daily = meta_offline.load_index(DEFAULT_DATASET, "3m", date(2026, 9, 1), date(2026, 10, 31))
    cpr = build_cpr_by_day(daily, {_ist_date(b["time"]) for b in bars})
    return bars, daily, idx_bars, idx_daily, cpr


def test_filtered_fires_subset_of_run_strategy(monkeypatch):
    """With a patched 1-feature extractor matching the toy booster's single feature,
    `filtered_fires` keeps a well-defined subset of `run_strategy`'s fires."""
    sym = pick_symbol()
    if not sym:
        pytest.skip("no offline dataset on disk")
    monkeypatch.setattr(features, "extract_features", parity_features)
    scorer = MetaScorer.from_artifact(toy_artifact(feature_names=None))
    params = params_from_primary(scorer.primary_params)
    bars, daily, idx_bars, idx_daily, cpr = _load(sym)
    base = run_strategy(bars, cpr, params)
    kept = filtered_fires(bars=bars, cpr_by_day=cpr, params=params, scorer=scorer,
                          index_bars=idx_bars, daily=daily, symbol=sym, index_daily=idx_daily)
    assert base, "expected some Dev Signal fires over two months"
    assert set(kept) <= set(base)
    assert kept == [f for f in base if f.bar_index % 2 == 1]


def test_extract_features_receives_no_future_context(monkeypatch):
    sym = pick_symbol()
    if not sym:
        pytest.skip("no offline dataset on disk")
    seen = []

    def spy(**kw):
        t = kw["bars"][kw["i"]]["time"]
        assert kw["index_bar"] is None or kw["index_bar"]["time"] <= t
        assert all(b["time"] <= t for b in kw["index_hist"])
        assert all(_ist_date(r["time"]) < _ist_date(t) for r in kw["daily_before"])
        assert kw["cpr"], "CPR should exist for fire days"
        seen.append(kw["i"])
        return {"x": 1.0}

    monkeypatch.setattr(features, "extract_features", spy)
    scorer = MetaScorer.from_artifact(toy_artifact())
    bars, daily, idx_bars, idx_daily, cpr = _load(sym)
    scored = score_fires(bars=bars, cpr_by_day=cpr, params=params_from_primary({}), scorer=scorer,
                         index_bars=idx_bars, daily=daily, symbol=sym, index_daily=idx_daily)
    assert len(seen) == len(scored) > 0

"""Raising the decision threshold can only reduce the kept-count — never grow it.

A monotonicity bug here (floating-point comparison direction wrong, off-by-one in
a sweep) would silently invalidate every acceptance-gate result.
"""
from __future__ import annotations

import random

import pytest

from stock_terminal_ml.meta.evaluate import THRESHOLD_GRID, pick_threshold


def _mk_sweep(probs: list[float], labels: list[int]) -> list[dict]:
    rows = []
    for p in THRESHOLD_GRID:
        kept_idx = [i for i, pr in enumerate(probs) if pr >= p]
        n = len(kept_idx)
        if n == 0:
            rows.append({"p": p, "score": 0.0, "ml": {"num_trades": 0, "net_pnl_pct": 0.0,
                                                       "sharpe": 0.0, "max_drawdown_pct": 0.0,
                                                       "win_rate": 0.0}})
            continue
        wins = sum(labels[i] for i in kept_idx)
        score = wins / n  # precision stand-in so pick_threshold has something to maximize
        rows.append({
            "p": p, "score": score,
            "ml": {"num_trades": n, "net_pnl_pct": float(wins * 1.0 - (n - wins) * 0.5),
                   "sharpe": 0.0, "max_drawdown_pct": 0.0, "win_rate": wins / n * 100.0},
        })
    return rows


def test_threshold_sweep_is_monotone_in_kept_count():
    rng = random.Random(0)
    probs = [rng.random() for _ in range(200)]
    labels = [1 if p + rng.gauss(0, 0.1) > 0.6 else 0 for p in probs]
    sweep = _mk_sweep(probs, labels)
    kept = [row["ml"]["num_trades"] for row in sweep]
    # non-increasing as p* rises
    for a, b in zip(kept, kept[1:]):
        assert a >= b, f"kept count jumped up: {a} -> {b}"


def test_pick_threshold_selects_best_mean_across_folds():
    # Two synthetic folds where p*=0.56 has the highest mean score.
    peak_p = 0.56
    folds = []
    for seed in (0, 1):
        rng = random.Random(seed)
        sweep = []
        for p in THRESHOLD_GRID:
            base = 1.0 - abs(p - peak_p) * 2.0 + rng.gauss(0.0, 0.01)
            sweep.append({"p": p, "score": base,
                           "ml": {"num_trades": 50, "net_pnl_pct": 0.0, "sharpe": 0.0,
                                   "max_drawdown_pct": 0.0, "win_rate": 0.0}})
        folds.append({"sweep": sweep})
    chosen, info = pick_threshold(folds)
    assert abs(chosen - peak_p) <= 0.04  # within one grid step
    assert "mean_score" in info

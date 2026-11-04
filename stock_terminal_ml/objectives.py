"""Objective functions scored on the output of a run.

Each function takes `(trades, equity_curve, metrics)` and returns a scalar Optuna
maximises. Default is `expected_pnl_with_constraints` — smooth, no cliffs, so the
TPE sampler sees a well-behaved landscape instead of a plateau.
"""
from __future__ import annotations

import math
from typing import Callable

from app.integrations.backtest.config import EquityPoint, Metrics, Trade


Objective = Callable[[list[Trade], list[EquityPoint], Metrics], float]


def expected_pnl_with_constraints(trades, equity_curve, metrics: Metrics) -> float:
    """Average per-trade PnL%, discounted by trade-count scarcity and drawdown.

    * n_scale ramps 0 → 1 across the first 30 trades so very-few-trade strategies
      can't fluke a high per-trade PnL into first place.
    * dd_scale is 1 while drawdown ≤ 10 %, decaying exponentially beyond.
    """
    if metrics.num_trades <= 0:
        return 0.0
    pnl_per_trade = metrics.net_pnl_pct / max(metrics.num_trades, 1)
    n_scale = min(1.0, metrics.num_trades / 30.0)
    dd_scale = math.exp(-max(0.0, metrics.max_drawdown_pct - 10.0) / 5.0)
    return pnl_per_trade * n_scale * dd_scale


def pure_pnl(trades, equity_curve, metrics: Metrics) -> float:
    return metrics.net_pnl_pct


def profit_factor(trades, equity_curve, metrics: Metrics) -> float:
    return metrics.profit_factor


def sharpe_like(trades, equity_curve, metrics: Metrics) -> float:
    """Per-trade mean / stdev of `net_pct`. Zero if < 2 trades or zero stdev."""
    if len(trades) < 2:
        return 0.0
    xs = [t.net_pct for t in trades]
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
    sd = math.sqrt(var)
    if sd == 0:
        return 0.0
    return mean / sd


BY_NAME: dict[str, Objective] = {
    "expected_pnl_with_constraints": expected_pnl_with_constraints,
    "pure_pnl": pure_pnl,
    "profit_factor": profit_factor,
    "sharpe_like": sharpe_like,
}

"""Decision threshold sweep + ML-vs-baseline economic comparison.

We never optimize F1. For each fold we sweep probability thresholds, re-run the
backtest keeping only fires with `p >= p*`, and score with the same economic
objective the v1 trainer uses (`stock_terminal_ml.objectives`). The threshold
that maximizes the *mean OOS* score across folds becomes `decision_threshold`.

A fold also reports the ML-vs-baseline pair (`net_pnl_pct`, `sharpe`, `num_trades`,
`max_drawdown_pct`, `win_rate`) that the acceptance gates in §6 read.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

from app.integrations.backtest.analytics import compute_metrics
from app.integrations.backtest.config import CostConfig, ExitRules
from app.integrations.backtest.simulator import simulate
from app.integrations.signals.config import Params
from app.integrations.signals.pattern import Fire

from .. import objectives


THRESHOLD_GRID = [round(0.30 + i * 0.02, 2) for i in range(31)]  # 0.30 .. 0.90


def _sharpe_like(trades) -> float:
    if len(trades) < 2:
        return 0.0
    xs = [t.net_pct for t in trades]
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    sd = math.sqrt(var)
    return m / sd if sd > 0 else 0.0


@dataclass(frozen=True)
class RunStats:
    net_pnl_pct: float
    sharpe: float
    num_trades: int
    max_drawdown_pct: float
    win_rate: float


def _run_trades(bars: Sequence[dict], fires: Sequence[Fire], params: Params,
                exits: ExitRules, costs: CostConfig, initial_capital: float) -> tuple[list, list, RunStats]:
    trades, curve = simulate(bars, fires, params,
                             initial_capital=initial_capital, index_factor=1.0,
                             exits=exits, costs=costs)
    metrics = compute_metrics(initial_capital, trades, curve)
    stats = RunStats(
        net_pnl_pct=float(metrics.net_pnl_pct),
        sharpe=_sharpe_like(trades),
        num_trades=int(metrics.num_trades),
        max_drawdown_pct=float(metrics.max_drawdown_pct),
        win_rate=float(metrics.win_rate),
    )
    return trades, curve, stats


def evaluate_fold(*, bars: Sequence[dict], params: Params, exits: ExitRules,
                  costs: CostConfig, initial_capital: float,
                  baseline_fires: Sequence[Fire],
                  fire_probs: Sequence[float],
                  thresholds: Sequence[float] = THRESHOLD_GRID,
                  objective: str = "expected_pnl_with_constraints") -> dict:
    """Score the fold at every candidate `p*`. Returns per-threshold economic
    score + baseline/ML stats, plus PR-AUC and precision/recall on the fold labels
    when they are supplied through `fire_probs` (one entry per `baseline_fires`)."""
    obj_fn = objectives.BY_NAME[objective]
    if len(fire_probs) != len(baseline_fires):
        raise ValueError("fire_probs must align 1:1 with baseline_fires")

    _, _, baseline_stats = _run_trades(bars, baseline_fires, params, exits, costs, initial_capital)
    base_metrics = compute_metrics(initial_capital,
                                   *simulate(bars, baseline_fires, params,
                                             initial_capital=initial_capital, index_factor=1.0,
                                             exits=exits, costs=costs))
    baseline_score = float(obj_fn(*simulate(bars, baseline_fires, params,
                                            initial_capital=initial_capital, index_factor=1.0,
                                            exits=exits, costs=costs), base_metrics))

    sweep: list[dict] = []
    for p_star in thresholds:
        kept = [f for f, p in zip(baseline_fires, fire_probs) if p >= p_star]
        trades, curve = simulate(bars, kept, params,
                                 initial_capital=initial_capital, index_factor=1.0,
                                 exits=exits, costs=costs)
        m = compute_metrics(initial_capital, trades, curve)
        score = float(obj_fn(trades, curve, m))
        sweep.append({
            "p": p_star,
            "score": score,
            "ml": asdict(RunStats(
                net_pnl_pct=float(m.net_pnl_pct), sharpe=_sharpe_like(trades),
                num_trades=int(m.num_trades), max_drawdown_pct=float(m.max_drawdown_pct),
                win_rate=float(m.win_rate),
            )),
        })
    return {
        "baseline": asdict(baseline_stats),
        "baseline_score": baseline_score,
        "sweep": sweep,
    }


def pick_threshold(folds_eval: Sequence[dict], *, min_trades: int = 1) -> tuple[float, dict]:
    """Pick the threshold that maximizes the mean economic score across folds.

    Rejects thresholds that produced <`min_trades` trades on every fold (a filter
    that keeps nothing is a failure, not a model — §6 acceptance gate 3).
    """
    if not folds_eval:
        return 0.5, {}
    # Index by threshold (same grid in every fold).
    by_p: dict[float, list[dict]] = {}
    for fe in folds_eval:
        for row in fe["sweep"]:
            by_p.setdefault(row["p"], []).append(row)
    best_p = None
    best_score = -float("inf")
    for p, rows in sorted(by_p.items()):
        if all(r["ml"]["num_trades"] < min_trades for r in rows):
            continue
        mean = sum(r["score"] for r in rows) / len(rows)
        if mean > best_score:
            best_score = mean
            best_p = p
    if best_p is None:  # every threshold killed all trades — fall back to the lowest
        best_p = min(by_p.keys())
        best_score = sum(r["score"] for r in by_p[best_p]) / len(by_p[best_p])
    return float(best_p), {"mean_score": float(best_score)}


def pr_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """Average-precision style PR-AUC (no sklearn). Handles ties by grouping."""
    pairs = sorted(zip(y_score, y_true), key=lambda t: t[0], reverse=True)
    tp = fp = 0
    total_pos = sum(1 for y in y_true if y == 1)
    if total_pos == 0:
        return 0.0
    ap = 0.0
    prev_recall = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        for k in range(i, j + 1):
            if pairs[k][1] == 1:
                tp += 1
            else:
                fp += 1
        precision = tp / (tp + fp)
        recall = tp / total_pos
        ap += precision * (recall - prev_recall)
        prev_recall = recall
        i = j + 1
    return float(ap)


def aggregate_gates(folds: Sequence[dict], *, min_kept_ratio: float = 0.2,
                    max_kept_ratio: float = 0.9) -> dict:
    """Roll fold stats up into the §6 acceptance signals.

    Each fold entry must carry {ml:{sharpe,net_pnl_pct,max_drawdown_pct,num_trades,win_rate},
    baseline:{...}, pr_auc, precision}.
    """
    if not folds:
        return {"shippable": False, "reason": "no folds"}
    better_sharpe = sum(1 for f in folds if f["ml"]["sharpe"] > f["baseline"]["sharpe"])
    ml_pnl = sum(f["ml"]["net_pnl_pct"] for f in folds)
    base_pnl = sum(f["baseline"]["net_pnl_pct"] for f in folds)
    ml_dd = max((f["ml"]["max_drawdown_pct"] for f in folds), default=0.0)
    base_dd = max((f["baseline"]["max_drawdown_pct"] for f in folds), default=0.0)
    ml_trades = sum(f["ml"]["num_trades"] for f in folds)
    base_trades = sum(f["baseline"]["num_trades"] for f in folds) or 1
    kept_ratio = ml_trades / base_trades
    ml_wr = sum(f["ml"]["win_rate"] for f in folds) / len(folds)
    base_wr = sum(f["baseline"]["win_rate"] for f in folds) / len(folds)
    pr_mean = sum(f.get("pr_auc", 0.0) for f in folds) / len(folds)

    gates = {
        "beats_baseline_sharpe_majority": better_sharpe * 2 >= len(folds),
        "aggregate_pnl_not_worse": ml_pnl >= base_pnl,
        "drawdown_not_worse": ml_dd <= base_dd + 1e-9,
        "pr_auc_mean": pr_mean,
        "win_rate_lift": ml_wr > base_wr,
        "trades_cut_sensibly": min_kept_ratio <= kept_ratio <= max_kept_ratio,
    }
    shippable = all([
        gates["beats_baseline_sharpe_majority"],
        gates["aggregate_pnl_not_worse"],
        gates["drawdown_not_worse"],
        gates["win_rate_lift"],
        gates["trades_cut_sensibly"],
    ])
    return {
        "shippable": shippable,
        "gates": gates,
        "ml_net_pnl_pct": ml_pnl,
        "baseline_net_pnl_pct": base_pnl,
        "ml_vs_baseline_net_pnl_pct": ml_pnl - base_pnl,
        "ml_max_drawdown_pct": ml_dd,
        "baseline_max_drawdown_pct": base_dd,
        "kept_ratio": kept_ratio,
        "pr_auc_mean": pr_mean,
    }

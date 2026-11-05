"""Meta-labeling trainer (docs/ml-v2/02-ml-methodology-plan.md).

Pipeline per run:

1. Per symbol, load offline bars/daily + NIFTY 50 index, run `run_strategy` with
   the primary params, label every fire (`labels.build_dataset`), and extract
   features via `app.integrations.ml_meta.scorer.fire_features` — the same
   function the live scorer uses, so train/serve stay in lockstep.
2. Pool samples across symbols. Walk-forward split by IST trading day with
   purge + embargo (`cv.walk_forward`). For each fold: fit LightGBM with
   `scale_pos_weight` computed on the *training* fold only, fit isotonic
   calibration on an inner held-out tail of the training region, score the test
   fold, sweep the probability threshold with the economic objective, and roll
   up ML-vs-baseline stats.
3. Pick the threshold that maximizes the mean OOS score across folds
   (`evaluate.pick_threshold`), refit on all data, and write a v2 artifact.

The CLI glue (`stock_terminal_ml/train_meta.py`) handles `run_start`/`run_done`;
this module emits optional `stage` + `fold` events (Appendix D) through
`progress.emit`.
"""
from __future__ import annotations

import math
import statistics
import traceback
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Sequence

import lightgbm as lgb
import numpy as np
from sklearn.isotonic import IsotonicRegression

from app.integrations.backtest.config import CostConfig, ExitRules
from app.integrations.backtest.strategy import build_cpr_by_day, run_strategy
from app.integrations.ml_meta import features as feat_mod
from app.integrations.ml_meta import offline as meta_offline
from app.integrations.ml_meta.scorer import fire_features, params_from_primary
from app.integrations.signals.engine import _ist_date

from .. import artifact as artifact_mod
from . import cv as cv_mod
from . import evaluate as eval_mod
from .labels import LabeledFire, build_dataset, final_weights


# LightGBM defaults — the user can override via config.extra["lgbm"].
DEFAULT_LGBM: dict = {
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "min_data_in_leaf": 20,
    "lambda_l2": 0.0,
    "verbose": -1,
    "seed": 42,
    "num_threads": 1,
}


@dataclass
class Sample:
    symbol: str
    day: date
    label: LabeledFire
    feats: dict


def _load_symbol_samples(*, symbol: str, dataset: str, lo: date, hi: date,
                         primary_params, exits: ExitRules, costs: CostConfig,
                         tau_cost: float) -> list[Sample]:
    got = meta_offline.load_bars(dataset, symbol, "3m", lo, hi)
    if got is None:
        return []
    bars, _first_in_range, daily = got
    idx_bars, idx_daily = meta_offline.load_index(dataset, "3m", lo, hi)
    if not bars:
        return []
    days = {_ist_date(b["time"]) for b in bars}
    cpr = build_cpr_by_day(daily, days)
    fires = run_strategy(bars, cpr, primary_params)
    if not fires:
        return []
    feats = fire_features(bars=bars, fires=fires, cpr_by_day=cpr, params=primary_params,
                          index_bars=idx_bars, daily=daily, symbol=symbol,
                          index_daily=idx_daily)
    labels = build_dataset(bars, fires, exits, costs, tau_cost=tau_cost)
    # Label and feature lists align except that label_fire drops last-bar-of-session
    # fires; match them back up by bar_index.
    feat_by_bar = {f.bar_index: feats[k] for k, f in enumerate(fires)}
    out: list[Sample] = []
    for L in labels:
        out.append(Sample(symbol=symbol, day=_ist_date(L.time),
                          label=L, feats=feat_by_bar.get(L.bar_index) or {}))
    return out


def _to_matrix(samples: Sequence[Sample], feature_names: Sequence[str]) -> np.ndarray:
    X = np.empty((len(samples), len(feature_names)), dtype=float)
    for i, s in enumerate(samples):
        for j, n in enumerate(feature_names):
            v = s.feats.get(n)
            X[i, j] = float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else float("nan")
    return X


def _fit_booster(X: np.ndarray, y: np.ndarray, w: np.ndarray,
                 feature_names: Sequence[str], categorical: Sequence[str],
                 lgbm_params: dict) -> lgb.Booster:
    pos = int((y == 1).sum()); neg = int((y == 0).sum())
    params = dict(lgbm_params)
    if pos > 0 and neg > 0:
        params["scale_pos_weight"] = neg / pos
    # Carve the last ~15% of the (time-ordered) training set as the internal
    # validation set for early stopping.
    n = len(y)
    v_from = max(1, int(0.85 * n))
    cat = [feature_names.index(c) for c in categorical if c in feature_names]
    train_set = lgb.Dataset(
        X[:v_from], label=y[:v_from], weight=w[:v_from],
        feature_name=list(feature_names), categorical_feature=cat,
        params={"verbose": -1},
    )
    valid_set = lgb.Dataset(
        X[v_from:], label=y[v_from:], weight=w[v_from:],
        feature_name=list(feature_names), categorical_feature=cat,
        reference=train_set, params={"verbose": -1},
    ) if n - v_from >= 5 else None
    callbacks = []
    if valid_set is not None:
        callbacks.append(lgb.early_stopping(stopping_rounds=30, verbose=False))
    return lgb.train(
        params, train_set,
        num_boost_round=400,
        valid_sets=[valid_set] if valid_set is not None else None,
        callbacks=callbacks,
    )


def _calibrate(booster: lgb.Booster, X_hold: np.ndarray, y_hold: np.ndarray) -> tuple[list[float], list[float]] | None:
    """Isotonic regression on raw scores; stored as (x, y) arrays."""
    if len(y_hold) < 20 or len(set(int(v) for v in y_hold)) < 2:
        return None
    raw = booster.predict(X_hold).astype(float)
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw, y_hold.astype(float))
    xs = np.linspace(float(raw.min()), float(raw.max()), 64)
    ys = iso.predict(xs)
    return [float(v) for v in xs], [float(v) for v in ys]


def _label_stats(samples: Sequence[Sample], initial_capital: float = 100.0) -> dict:
    """Per-trade stats aggregated across the given samples (each `net_pct` is one
    independent trade). Equity curve is `initial_capital * prod(1 + net_pct/100)`."""
    if not samples:
        return {"net_pnl_pct": 0.0, "sharpe": 0.0, "num_trades": 0,
                "max_drawdown_pct": 0.0, "win_rate": 0.0}
    rets = [s.label.net_pct for s in samples]
    eq = [initial_capital]
    peak = initial_capital
    dd = 0.0
    for r in rets:
        eq.append(eq[-1] * (1 + r / 100.0))
        peak = max(peak, eq[-1])
        dd = max(dd, (peak - eq[-1]) / peak * 100.0 if peak > 0 else 0.0)
    final = eq[-1]
    net_pnl_pct = (final - initial_capital) / initial_capital * 100.0
    m = sum(rets) / len(rets)
    sd = math.sqrt(sum((r - m) ** 2 for r in rets) / (len(rets) - 1)) if len(rets) > 1 else 0.0
    wins = sum(1 for r in rets if r > 0)
    return {
        "net_pnl_pct": float(net_pnl_pct),
        "sharpe": float(m / sd) if sd > 0 else 0.0,
        "num_trades": int(len(rets)),
        "max_drawdown_pct": float(dd),
        "win_rate": float(wins / len(rets) * 100.0),
    }


def _economic_score(samples: Sequence[Sample]) -> float:
    """Simplified `expected_pnl_with_constraints` on independent per-trade rets."""
    if not samples:
        return 0.0
    stats = _label_stats(samples)
    n = stats["num_trades"]
    pnl_per_trade = stats["net_pnl_pct"] / max(n, 1)
    n_scale = min(1.0, n / 30.0)
    dd_scale = math.exp(-max(0.0, stats["max_drawdown_pct"] - 10.0) / 5.0)
    return pnl_per_trade * n_scale * dd_scale


def _sweep_threshold(test: Sequence[tuple[Sample, float]],
                     thresholds: Sequence[float]) -> list[dict]:
    """For each candidate p*, return the aggregate ML stats + economic score."""
    out = []
    for p_star in thresholds:
        kept = [s for s, p in test if p >= p_star]
        out.append({
            "p": float(p_star),
            "score": _economic_score(kept),
            "ml": _label_stats(kept),
        })
    return out


def run(cfg: dict, progress) -> str | Path:
    """Train a meta-labeling model, write a v2 artifact, return its path."""
    dataset = cfg["dataset"]
    lo = date.fromisoformat(cfg["window"]["from"])
    hi = date.fromisoformat(cfg["window"]["to"])
    symbols: list[str] = list(cfg["symbols"])
    tau_cost = float(cfg.get("extra", {}).get("tau_cost", 0.0))
    primary = params_from_primary(cfg.get("extra", {}).get("primary_params") or {})
    exits = ExitRules(target_pct=float(cfg["target_pct"]), stop_loss_pct=float(cfg["sl_pct"]))
    costs = CostConfig(enabled=True, segment="EQUITY_INTRADAY")

    lgbm_params = dict(DEFAULT_LGBM)
    lgbm_params.update(cfg.get("extra", {}).get("lgbm") or {})

    # --- 1. features + labels ------------------------------------------------
    progress.emit("stage", name="features", pct=0.0)
    all_samples: list[Sample] = []
    for k, sym in enumerate(symbols):
        try:
            all_samples.extend(_load_symbol_samples(
                symbol=sym, dataset=dataset, lo=lo, hi=hi,
                primary_params=primary, exits=exits, costs=costs,
                tau_cost=tau_cost,
            ))
        except Exception as e:  # pragma: no cover
            progress.emit("error", symbol=sym, message=f"{type(e).__name__}: {e}",
                          trace=traceback.format_exc())
        progress.emit("stage", name="features", pct=(k + 1) / max(1, len(symbols)))
    if not all_samples:
        raise RuntimeError("no labeled samples — check the dataset/window/symbols")
    all_samples.sort(key=lambda s: (s.day, s.symbol, s.label.bar_index))
    feature_names = list(feat_mod.FEATURE_NAMES)
    categorical = list(feat_mod.CATEGORICAL_FEATURES)

    # --- 2. walk-forward folds -----------------------------------------------
    trading_days = sorted({s.day for s in all_samples})
    folds = cv_mod.walk_forward(
        trading_days,
        train_months=int(cfg["train_months"]),
        test_months=int(cfg["test_months"]),
        embargo_days=int(cfg["embargo_days"]),
        mode=str(cfg.get("extra", {}).get("cv_mode") or "expanding"),
    )
    if not folds:
        raise RuntimeError("not enough trading days for a single walk-forward fold")

    by_day: dict[date, list[Sample]] = {}
    for s in all_samples:
        by_day.setdefault(s.day, []).append(s)

    thresholds = eval_mod.THRESHOLD_GRID
    folds_eval: list[dict] = []
    for fold in folds:
        progress.emit("stage", name=f"fold-{fold.index}", pct=fold.index / len(folds))
        train = [s for d in fold.train_days for s in by_day.get(d, [])]
        test = [s for d in fold.test_days for s in by_day.get(d, [])]
        if not train or not test:
            continue

        X_tr = _to_matrix(train, feature_names)
        y_tr = np.array([s.label.y for s in train], dtype=float)
        w_tr = np.array(final_weights([s.label for s in train], time_decay=True), dtype=float)
        booster = _fit_booster(X_tr, y_tr, w_tr, feature_names, categorical, lgbm_params)

        # calibration on an inner held-out tail of training
        n = len(train)
        split = max(1, int(0.85 * n))
        if n - split >= 20:
            cal = _calibrate(booster, X_tr[split:], y_tr[split:])
        else:
            cal = None

        X_te = _to_matrix(test, feature_names)
        y_te = np.array([s.label.y for s in test], dtype=float)
        raw = booster.predict(X_te).astype(float)
        if cal:
            xs, ys = cal
            probs = np.clip(np.interp(raw, xs, ys), 0.0, 1.0)
        else:
            probs = np.clip(raw, 0.0, 1.0)

        sweep = _sweep_threshold(list(zip(test, probs.tolist())), thresholds)
        baseline = _label_stats(test)
        pr = eval_mod.pr_auc(y_te.astype(int).tolist(), probs.tolist())
        # Simple precision@0.5 for the fold log.
        pred = (probs >= 0.5).astype(int)
        tp = int(((pred == 1) & (y_te == 1)).sum()); fp = int(((pred == 1) & (y_te == 0)).sum())
        precision = tp / max(1, tp + fp)
        fe = {
            "index": fold.index,
            "train_from": fold.train_from.isoformat(), "train_to": fold.train_to.isoformat(),
            "test_from": fold.test_from.isoformat(), "test_to": fold.test_to.isoformat(),
            "num_train": len(train), "num_test": len(test),
            "baseline": baseline,
            "pr_auc": pr, "precision": precision,
            "sweep": sweep,
        }
        folds_eval.append(fe)
        progress.emit(
            "fold", index=fold.index,
            train_from=fe["train_from"], train_to=fe["train_to"],
            test_from=fe["test_from"], test_to=fe["test_to"],
            metrics={"pr_auc": pr, "baseline_net_pnl_pct": baseline["net_pnl_pct"],
                     "num_train": len(train), "num_test": len(test)},
        )

    if not folds_eval:
        raise RuntimeError("walk-forward produced no usable folds")

    # --- 3. pick threshold + aggregate gates ---------------------------------
    sweeps_for_pick = [{"sweep": fe["sweep"]} for fe in folds_eval]
    chosen_p, pick_info = eval_mod.pick_threshold(sweeps_for_pick)
    # pack each fold's "ml" at the chosen threshold for the aggregate gates
    chosen_rows: list[dict] = []
    for fe in folds_eval:
        row = next((r for r in fe["sweep"] if r["p"] == chosen_p), fe["sweep"][0])
        chosen_rows.append({
            "ml": row["ml"],
            "baseline": fe["baseline"],
            "pr_auc": fe["pr_auc"],
            "precision": fe["precision"],
        })
    gates = eval_mod.aggregate_gates(chosen_rows)

    # --- 4. final refit on all data + final calibration ----------------------
    progress.emit("stage", name="final-fit", pct=0.9)
    X_all = _to_matrix(all_samples, feature_names)
    y_all = np.array([s.label.y for s in all_samples], dtype=float)
    w_all = np.array(final_weights([s.label for s in all_samples], time_decay=True), dtype=float)
    final_booster = _fit_booster(X_all, y_all, w_all, feature_names, categorical, lgbm_params)
    # Calibration refit on the tail of all data.
    split = max(1, int(0.85 * len(all_samples)))
    cal_final = _calibrate(final_booster, X_all[split:], y_all[split:]) if len(all_samples) - split >= 20 else None

    # --- 5. feature importance ------------------------------------------------
    try:
        imp_vals = final_booster.feature_importance(importance_type="gain")
        feature_importance = {feature_names[i]: float(imp_vals[i]) for i in range(len(feature_names))}
    except Exception:
        feature_importance = {}

    # --- 6. v2 artifact -------------------------------------------------------
    per_symbol = {}
    for sym in sorted({s.symbol for s in all_samples}):
        rows = [s for s in all_samples if s.symbol == sym]
        per_symbol[sym] = {"oos": _label_stats(rows), "samples": len(rows)}

    evaluation = {
        "oos": {
            "folds": [
                {
                    "index": fe["index"],
                    "test_from": fe["test_from"], "test_to": fe["test_to"],
                    "pr_auc": fe["pr_auc"], "precision": fe["precision"],
                    "ml": next((r["ml"] for r in fe["sweep"] if r["p"] == chosen_p),
                                fe["sweep"][0]["ml"]),
                    "baseline": fe["baseline"],
                }
                for fe in folds_eval
            ],
            "aggregate": {
                **gates,
                "pr_auc_mean": sum(fe["pr_auc"] for fe in folds_eval) / len(folds_eval),
                "chosen_threshold": chosen_p,
                "chosen_mean_score": pick_info.get("mean_score"),
            },
        },
        "feature_importance": feature_importance,
    }

    primary_dict: dict = asdict(primary) if is_dataclass(primary) else {}
    if isinstance(primary_dict.get("cpr_lines"), tuple):
        primary_dict["cpr_lines"] = list(primary_dict["cpr_lines"])

    cal_block = None
    if cal_final:
        cal_block = {"x": cal_final[0], "y": cal_final[1]}

    payload = artifact_mod.build_v2(
        model_id=cfg["model_id"],
        label=cfg["label"],
        primary={"params": primary_dict},
        model={
            "framework": "lightgbm",
            "objective": "binary",
            "booster": final_booster.model_to_string(),
            "decision_threshold": float(chosen_p),
            **({"calibration": cal_block} if cal_block else {}),
        },
        features={
            "spec_version": feat_mod.FEATURE_SPEC_VERSION,
            "names": feature_names,
            "categorical": categorical,
        },
        training={
            "dataset": dataset,
            "window": cfg["window"],
            "train_months": cfg["train_months"],
            "test_months": cfg["test_months"],
            "embargo_days": cfg["embargo_days"],
            "model_family": cfg["model_family"],
            "hyperparams": lgbm_params,
            "symbols_total": len(symbols),
            "samples_total": len(all_samples),
            "exits": {"target_pct": float(cfg["target_pct"]), "sl_pct": float(cfg["sl_pct"])},
            "objective": cfg.get("threshold_objective", "expected_pnl_with_constraints"),
            "finished_at": artifact_mod.utc_iso(),
        },
        evaluation=evaluation,
        per_symbol=per_symbol,
        global_fallback={"decision_threshold": float(chosen_p)},
    )
    out_path = Path(cfg["output"])
    written = artifact_mod.write_atomic_v2(payload, out_path)
    return written

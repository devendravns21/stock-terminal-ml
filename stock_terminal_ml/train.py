"""CLI entry: tune Dev Signal params per symbol with Optuna, write a model artifact.

Design notes
============
Each symbol runs an independent Optuna study (TPE sampler, seed 42). The objective
is the mean-across-folds score of (`run_strategy` → `simulate` → `compute_metrics`
→ `objective_fn`). Expanding-window CV lives in `cv.kfold`; the last 30 trading days
after the user window are reserved OOS and scored once with the best params.

The backend subprocess reads `progress.jsonl` to drive the UI. Events:
  - {"event": "run_start", ...}
  - {"event": "trial", "symbol": S, "trial": i, "value": v, "state": "complete"|...}
  - {"event": "symbol_done", "symbol": S, "best_value": v}
  - {"event": "run_done", "artifact": path}
  - {"event": "error", "symbol": S?, "message": str}

Global fallback: median per tuned param across symbols; untuned Params fields stay
at their defaults.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics
import sys
import traceback
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Sequence

import optuna

from app.core.config import settings
from app.integrations.backtest.config import CostConfig, ExitRules
from app.integrations.backtest.analytics import compute_metrics
from app.integrations.backtest.simulator import simulate
from app.integrations.backtest.strategy import build_cpr_by_day, run_strategy
from app.integrations.offline.store import DEFAULT_DATASET, dataset_dir, load_symbol
from app.integrations.signals.config import Params
from app.integrations.signals.engine import _ist_date

from . import artifact
from . import cv as cv_mod
from . import objectives
from .progress import ProgressWriter
from .search_space import TUNED_KEYS, WIDEN_KEYS, suggest, tuned_keys

optuna.logging.set_verbosity(optuna.logging.WARNING)


PRESETS = {
    "fast":     {"trials": 25,  "pruner": False},
    "standard": {"trials": 80,  "pruner": False},
    "thorough": {"trials": 200, "pruner": True},
}


def _slice_to_window(full_bars: Sequence[dict], from_iso: str, to_iso: str) -> list[dict]:
    start = date.fromisoformat(from_iso)
    end = date.fromisoformat(to_iso)
    out = []
    for b in full_bars:
        d = _ist_date(b["time"])
        if start <= d <= end:
            out.append(b)
    return out


def _cpr_for(bars: Sequence[dict], daily_rows: Sequence[dict]) -> dict[date, dict]:
    days = {_ist_date(b["time"]) for b in bars}
    return build_cpr_by_day(daily_rows, days)


def _score_once(
    bars: list[dict],
    cpr: dict[date, dict],
    params: Params,
    exits: ExitRules,
    costs: CostConfig,
    initial_capital: float,
    objective_fn,
) -> tuple[float, dict]:
    """Run one strategy+simulate+metrics pass, return (score, serialisable metrics)."""
    fires = run_strategy(bars, cpr, params)
    trades, curve = simulate(
        bars, fires, params,
        initial_capital=initial_capital,
        index_factor=1.0,
        exits=exits,
        costs=costs,
    )
    metrics = compute_metrics(initial_capital, trades, curve)
    score = float(objective_fn(trades, curve, metrics))
    return score, asdict(metrics)


def _tune_symbol(args: dict) -> dict:
    """Entry point for one symbol — run_* functions are inlined here so the Pool
    worker can pickle the args cleanly.

    Returns `{symbol, params, cv, oos, best_value, error?}`.
    """
    symbol: str = args["symbol"]
    try:
        dataset: str = args["dataset"]
        from_iso: str = args["from"]
        to_iso: str = args["to"]
        target_pct: float = args["target_pct"]
        sl_pct: float = args["sl_pct"]
        trials: int = args["trials"]
        use_pruner: bool = args["pruner"]
        objective_name: str = args["objective"]
        widen: bool = args["widen"]
        initial_capital: float = args.get("initial_capital", 100000.0)
        progress_path: str | None = args.get("progress_path")
        folds: int = args.get("folds", 3)

        objective_fn = objectives.BY_NAME[objective_name]

        data = load_symbol(dataset, symbol)
        if not data:
            return {"symbol": symbol, "error": "symbol not in dataset"}
        full_bars: list[dict] = data["candles"]
        daily_rows: list[dict] = data["daily"]

        window_bars = _slice_to_window(full_bars, from_iso, to_iso)
        warmup = cv_mod.warmup_bars(full_bars, from_iso, warmup_days=2)
        train_full = warmup + window_bars
        if len(window_bars) < 50:
            return {"symbol": symbol, "error": "not enough bars in window"}

        folds_data = cv_mod.kfold(window_bars, folds=folds)
        if not folds_data:
            return {"symbol": symbol, "error": "not enough days for CV folds"}

        exits = ExitRules(target_pct=target_pct, stop_loss_pct=sl_pct)
        costs = CostConfig(enabled=True, segment="EQUITY_INTRADAY")

        # Pre-compute cpr_by_day over the whole symbol so each fold just does a lookup.
        cpr_all = build_cpr_by_day(daily_rows, {_ist_date(b["time"]) for b in full_bars})

        # Per-fold runner: prepend warm-up so day-1 inside the fold has CPR + ST warm.
        def _fold_run(trial_params: Params, train: list[dict], val: list[dict]) -> float:
            # Train slice score (expanding-left prefix); validate on val slice.
            train_bars = warmup + train
            val_bars = (train[-240:] if train else []) + val  # small prefix for ST warm-up
            train_score, _ = _score_once(train_bars, cpr_all, trial_params,
                                         exits, costs, initial_capital, objective_fn)
            val_score, _ = _score_once(val_bars, cpr_all, trial_params,
                                       exits, costs, initial_capital, objective_fn)
            # Average the two (TPE sees both in and out-of-sample signal per fold).
            return (train_score + val_score) / 2.0

        progress = ProgressWriter(Path(progress_path)) if progress_path else None

        def objective(trial: optuna.Trial) -> float:
            p = suggest(trial, widen=widen)
            scores = []
            for train, val in folds_data:
                s = _fold_run(p, train, val)
                scores.append(s)
            # Mean across folds — the smooth default objective handles the rest.
            return sum(scores) / len(scores)

        pruner = optuna.pruners.MedianPruner() if use_pruner else optuna.pruners.NopPruner()
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=pruner,
        )

        def _trial_cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            if progress:
                progress.emit(
                    "trial",
                    symbol=symbol,
                    trial=trial.number,
                    value=float(trial.value) if trial.value is not None else None,
                    state=str(trial.state).split(".")[-1].lower(),
                )

        study.optimize(objective, n_trials=trials, callbacks=[_trial_cb], show_progress_bar=False)

        best_params = Params(**{**{k: v for k, v in study.best_params.items()}})
        best_value = float(study.best_value)

        # CV mean: re-score best params on each fold's val slice individually.
        cv_scores: list[float] = []
        cv_metrics: list[dict] = []
        for train, val in folds_data:
            val_bars = (train[-240:] if train else []) + val
            s, m = _score_once(val_bars, cpr_all, best_params, exits, costs,
                               initial_capital, objective_fn)
            cv_scores.append(s)
            cv_metrics.append(m)
        cv_mean_score = sum(cv_scores) / len(cv_scores)
        cv_summary = {
            "score_mean": cv_mean_score,
            "score_per_fold": cv_scores,
            "net_pnl_pct": statistics.mean(m["net_pnl_pct"] for m in cv_metrics),
            "num_trades": int(statistics.mean(m["num_trades"] for m in cv_metrics)),
            "win_rate":   statistics.mean(m["win_rate"]   for m in cv_metrics),
            "max_drawdown_pct": statistics.mean(m["max_drawdown_pct"] for m in cv_metrics),
        }

        # OOS: next `holdout_trading_days` after the user window.
        oos_slice = cv_mod.holdout(full_bars, to_iso, trading_days=30)
        oos_summary: dict | None = None
        if oos_slice:
            oos_bars = (window_bars[-240:] if window_bars else []) + oos_slice
            s, m = _score_once(oos_bars, cpr_all, best_params, exits, costs,
                               initial_capital, objective_fn)
            oos_summary = {
                "score": s,
                "net_pnl_pct": m["net_pnl_pct"],
                "num_trades": m["num_trades"],
                "win_rate":   m["win_rate"],
                "max_drawdown_pct": m["max_drawdown_pct"],
                "trading_days": len({_ist_date(b["time"]) for b in oos_slice}),
            }

        tuned = {k: getattr(best_params, k) for k in tuned_keys(widen)}
        result = {
            "symbol": symbol,
            "params": tuned,
            "cv": cv_summary,
            "oos": oos_summary,
            "best_value": best_value,
        }

        if progress:
            progress.emit("symbol_done", symbol=symbol, best_value=best_value)
            progress.close()
        return result

    except Exception as e:  # pragma: no cover — surface rather than crash the pool
        return {
            "symbol": symbol,
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc(),
        }


def _default_symbols(dataset: str) -> list[str]:
    try:
        root = dataset_dir(dataset)
    except ValueError:
        return []
    if not root.is_dir():
        return []
    syms = []
    for p in sorted(root.glob("*.json")):
        if p.name == "manifest.json":
            continue
        try:
            doc = json.loads(p.read_text())
            sym = doc.get("symbol")
            if sym:
                syms.append(sym)
        except Exception:
            continue
    return syms


def _parse_window(window: str) -> tuple[str, str]:
    a, b = window.split(":", 1)
    date.fromisoformat(a); date.fromisoformat(b)
    return a, b


def _load_config(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _median_params(per_symbol: dict[str, dict], widen: bool) -> dict:
    keys = tuned_keys(widen)
    acc: dict[str, list[float]] = {k: [] for k in keys}
    for row in per_symbol.values():
        params = row.get("params") or {}
        for k in keys:
            v = params.get(k)
            if v is not None:
                acc[k].append(float(v))
    out: dict[str, Any] = {}
    for k, xs in acc.items():
        if not xs:
            continue
        # ints stay ints (cooldown_bars, max_bars_to_retest, break_lookback_bars)
        med = statistics.median(xs)
        if k in {"cooldown_bars", "max_bars_to_retest", "break_lookback_bars"}:
            out[k] = int(round(med))
        else:
            out[k] = float(med)
    return out


def build_parser(prog: str = "stock-terminal-ml.train") -> argparse.ArgumentParser:
    """The v1 trainer's flags. `train_meta.py` extends this parser for `--mode meta`."""
    ap = argparse.ArgumentParser(prog=prog)
    ap.add_argument("--window", help="FROM:TO IST dates, e.g. 2026-06-01:2026-08-31")
    ap.add_argument("--target-pct", type=float, default=1.0)
    ap.add_argument("--sl-pct", type=float, default=0.5)
    ap.add_argument("--preset", choices=list(PRESETS), default="standard")
    ap.add_argument("--trials", type=int, help="override trials per symbol")
    ap.add_argument("--symbols", help="comma-separated SYM list (default: all in dataset)")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--output", help="path for the artifact JSON")
    ap.add_argument("--config", help="path to a JSON config (CLI flags override)")
    ap.add_argument("--progress", help="path to progress JSONL (default: alongside output)")
    ap.add_argument("--objective", default="expected_pnl_with_constraints",
                    choices=list(objectives.BY_NAME))
    ap.add_argument("--widen", action="store_true",
                    help="unlock the frozen knobs listed in the main plan")
    ap.add_argument("--jobs", type=int, default=0,
                    help="worker count (0 = cpu_count, 1 = serial for debugging)")
    ap.add_argument("--model-id", help="override auto-generated id, e.g. dev_signal_ml3")
    ap.add_argument("--label", help="override artifact label")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--initial-capital", type=float, default=100000.0)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cfg: dict = {}
    if args.config:
        cfg = _load_config(args.config)

    # CLI `--window FROM:TO` overrides; otherwise read `window: {from, to}` from config.
    if args.window:
        from_iso, to_iso = _parse_window(args.window)
    else:
        cfg_window = cfg.get("window") or {}
        from_iso = cfg_window.get("from")
        to_iso = cfg_window.get("to")
    if not (from_iso and to_iso):
        print("--window or config.window.{from,to} is required", file=sys.stderr)
        return 2

    target_pct = args.target_pct if args.target_pct != 1.0 else cfg.get("target_pct", args.target_pct)
    sl_pct = args.sl_pct if args.sl_pct != 0.5 else cfg.get("sl_pct", args.sl_pct)
    preset = (args.preset if args.preset != "standard" else cfg.get("preset", "standard")).lower()
    if preset not in PRESETS:
        print(f"unknown preset {preset!r}", file=sys.stderr)
        return 2
    preset_cfg = PRESETS[preset]
    trials = args.trials if args.trials is not None else preset_cfg["trials"]
    use_pruner = preset_cfg["pruner"]

    dataset = args.dataset or cfg.get("dataset") or DEFAULT_DATASET
    symbols_arg = args.symbols or cfg.get("symbols")
    if symbols_arg:
        symbols = [s.strip() for s in (
            symbols_arg if isinstance(symbols_arg, list)
            else symbols_arg.split(",")
        ) if s.strip()]
    else:
        symbols = _default_symbols(dataset)
    if not symbols:
        print(f"no symbols found in dataset {dataset!r}", file=sys.stderr)
        return 2

    output_path = args.output or cfg.get("output") or os.environ.get("ML_OUTPUT_PATH")
    if not output_path:
        output_path = str(settings.ml_models_dir / "dev_signal_ml1.json")
    output_path = Path(output_path)
    progress_path = args.progress or cfg.get("progress") or str(output_path.with_suffix(".progress.jsonl"))

    model_id = args.model_id or cfg.get("model_id") or output_path.stem
    label = args.label or cfg.get("label") or model_id.replace("dev_signal_ml", "Dev Signal ML")

    widen = args.widen or cfg.get("widen", False)
    objective_name = args.objective or cfg.get("objective", "expected_pnl_with_constraints")
    initial_capital = float(args.initial_capital or cfg.get("initial_capital", 100000.0))

    progress = ProgressWriter(Path(progress_path))
    progress.emit(
        "run_start",
        model_id=model_id,
        dataset=dataset,
        window={"from": from_iso, "to": to_iso},
        preset=preset,
        trials=trials,
        symbols_total=len(symbols),
        objective=objective_name,
        widen=widen,
    )

    per_args = [
        {
            "symbol": sym,
            "dataset": dataset,
            "from": from_iso,
            "to": to_iso,
            "target_pct": target_pct,
            "sl_pct": sl_pct,
            "trials": trials,
            "pruner": use_pruner,
            "objective": objective_name,
            "widen": widen,
            "initial_capital": initial_capital,
            "progress_path": progress_path,
            "folds": args.folds,
        }
        for sym in symbols
    ]

    per_symbol_out: dict[str, dict] = {}

    if args.jobs == 1:
        for a in per_args:
            r = _tune_symbol(a)
            _collect(r, per_symbol_out, progress)
    else:
        n_workers = args.jobs or max(1, mp.cpu_count() - 1)
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            for r in pool.imap_unordered(_tune_symbol, per_args):
                _collect(r, per_symbol_out, progress)

    fallback = _median_params(per_symbol_out, widen=widen)

    payload = artifact.build(
        model_id=model_id,
        label=label,
        training={
            "dataset": dataset,
            "window": {"from": from_iso, "to": to_iso},
            "target_pct": target_pct,
            "sl_pct": sl_pct,
            "preset": preset,
            "trials_per_symbol": trials,
            "folds": args.folds,
            "objective": objective_name,
            "widen": widen,
            "symbols_total": len(symbols),
            "symbols_succeeded": len(per_symbol_out),
            "finished_at": artifact.utc_iso(),
        },
        per_symbol=per_symbol_out,
        global_fallback={"params": fallback},
    )

    written = artifact.write_atomic(payload, output_path)
    progress.emit("run_done", artifact=str(written))
    progress.close()
    return 0


def _collect(result: dict, per_symbol_out: dict[str, dict], progress: ProgressWriter) -> None:
    sym = result.get("symbol")
    if not sym:
        return
    if "error" in result:
        progress.emit("error", symbol=sym, message=result["error"])
        return
    per_symbol_out[sym] = {
        "params": result["params"],
        "cv": result["cv"],
        "oos": result["oos"],
        "best_value": result["best_value"],
    }


if __name__ == "__main__":
    sys.exit(main())

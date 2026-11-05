"""CLI entry with a `--mode` switch: v1 param tuning or v2 meta-labeling.

    python -m stock_terminal_ml.train_meta --mode params ...   # identical to stock_terminal_ml.train
    python -m stock_terminal_ml.train_meta --mode meta --window 2026-01-01:2026-10-31 \
        --train-months 3 --test-months 1 --embargo-days 1 --output /tmp/m.json

`--mode params` (default) delegates to `train.main` with the same argv (minus
`--mode`), so v1 behaviour is unchanged. `--mode meta` resolves the config here,
emits `run_start`, calls `meta.train_meta.run(cfg, progress)` (training logic owned
by docs/ml-v2/02-ml-methodology-plan.md) and emits `run_done` with the artifact path
it returns, or `error` if it raises.

Config precedence matches `train.py`: `--config` JSON values are overridden by a CLI
flag only when that flag is not at its argparse default (see CLAUDE.md).
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.integrations.offline.store import DEFAULT_DATASET

from . import objectives
from . import train as v1
from .progress import ProgressWriter

MODES = ("params", "meta")
MODEL_FAMILIES = ("lightgbm",)

# Meta-only flags -> argparse default. Used for the precedence rule and to reject
# meta flags passed in params mode.
META_DEFAULTS: dict[str, Any] = {
    "train_months": 3,
    "test_months": 1,
    "embargo_days": 1,
    "model_family": "lightgbm",
    "calibrate": True,
    "threshold_objective": "expected_pnl_with_constraints",
}


def build_parser() -> argparse.ArgumentParser:
    ap = v1.build_parser(prog="stock-terminal-ml.train_meta")
    ap.add_argument("--mode", choices=MODES, default="params",
                    help="params = v1 per-symbol param tuning (default); meta = v2 meta-labeling")
    g = ap.add_argument_group("meta-labeling (--mode meta)")
    g.add_argument("--train-months", type=int, default=META_DEFAULTS["train_months"],
                   help="walk-forward training span in months")
    g.add_argument("--test-months", type=int, default=META_DEFAULTS["test_months"],
                   help="walk-forward test span in months (the fold step)")
    g.add_argument("--embargo-days", type=int, default=META_DEFAULTS["embargo_days"],
                   help="trading days purged from training before each test span")
    g.add_argument("--model-family", choices=MODEL_FAMILIES, default=META_DEFAULTS["model_family"])
    g.add_argument("--calibrate", action=argparse.BooleanOptionalAction,
                   default=META_DEFAULTS["calibrate"], help="fit probability calibration")
    g.add_argument("--threshold-objective", choices=list(objectives.BY_NAME),
                   default=META_DEFAULTS["threshold_objective"],
                   help="economic objective used to pick the decision threshold")
    return ap


def _pick(cli_value, default, cfg: dict, key: str):
    """v1 precedence: the CLI value wins only when it differs from its default."""
    return cli_value if cli_value != default else cfg.get(key, default)


def _strip_mode(argv: list[str]) -> list[str]:
    out: list[str] = []
    skip = False
    for a in argv:
        if skip:
            skip = False
            continue
        if a == "--mode":
            skip = True
            continue
        if a.startswith("--mode="):
            continue
        out.append(a)
    return out


def resolve_meta_config(args: argparse.Namespace) -> dict:
    """Merge `--config` JSON and CLI flags into the dict handed to `meta.train_meta.run`."""
    cfg: dict = v1._load_config(args.config) if args.config else {}

    if args.window:
        from_iso, to_iso = v1._parse_window(args.window)
    else:
        w = cfg.get("window") or {}
        from_iso, to_iso = w.get("from"), w.get("to")
    if not (from_iso and to_iso):
        raise ValueError("--window or config.window.{from,to} is required")

    dataset = args.dataset or cfg.get("dataset") or DEFAULT_DATASET
    symbols_arg = args.symbols or cfg.get("symbols")
    if symbols_arg:
        symbols = [s.strip() for s in (
            symbols_arg if isinstance(symbols_arg, list) else symbols_arg.split(",")
        ) if s.strip()]
    else:
        symbols = v1._default_symbols(dataset)
    if not symbols:
        raise ValueError(f"no symbols found in dataset {dataset!r}")

    output_path = args.output or cfg.get("output") or os.environ.get("ML_OUTPUT_PATH")
    if not output_path:
        output_path = str(settings.ml_models_dir / "dev_signal_ml1.json")
    output_path = Path(output_path)
    progress_path = args.progress or cfg.get("progress") or str(output_path.with_suffix(".progress.jsonl"))
    model_id = args.model_id or cfg.get("model_id") or output_path.stem
    label = args.label or cfg.get("label") or model_id.replace("dev_signal_ml", "Dev Signal ML")

    out = {
        "mode": "meta",
        "window": {"from": from_iso, "to": to_iso},
        "dataset": dataset,
        "symbols": symbols,
        "output": str(output_path),
        "progress": str(progress_path),
        "model_id": model_id,
        "label": label,
        "target_pct": _pick(args.target_pct, 1.0, cfg, "target_pct"),
        "sl_pct": _pick(args.sl_pct, 0.5, cfg, "sl_pct"),
        "objective": _pick(args.objective, "expected_pnl_with_constraints", cfg, "objective"),
        "jobs": _pick(args.jobs, 0, cfg, "jobs"),
        "initial_capital": float(_pick(args.initial_capital, 100000.0, cfg, "initial_capital")),
    }
    for key, default in META_DEFAULTS.items():
        out[key] = _pick(getattr(args, key), default, cfg, key)
    if out["model_family"] not in MODEL_FAMILIES:
        raise ValueError(f"unknown model_family {out['model_family']!r}")
    if out["threshold_objective"] not in objectives.BY_NAME:
        raise ValueError(f"unknown threshold_objective {out['threshold_objective']!r}")
    for k in ("train_months", "test_months"):
        if int(out[k]) < 1:
            raise ValueError(f"{k} must be >= 1")
    if int(out["embargo_days"]) < 0:
        raise ValueError("embargo_days must be >= 0")
    # Free-form passthrough: Doc 2 may read extra config keys (e.g. cv mode) from here.
    out["extra"] = {k: v for k, v in cfg.items() if k not in out}
    return out


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    cfg_mode = (v1._load_config(args.config).get("mode") if args.config else None) or "params"
    mode = _pick(args.mode, "params", {"mode": cfg_mode}, "mode")
    if mode not in MODES:
        print(f"unknown mode {mode!r}", file=sys.stderr)
        return 2

    if mode == "params":
        bad = [k for k, d in META_DEFAULTS.items() if getattr(args, k) != d]
        if bad:
            print(f"meta-only flags need --mode meta: {', '.join('--' + k.replace('_', '-') for k in bad)}",
                  file=sys.stderr)
            return 2
        return v1.main(_strip_mode(argv))

    try:
        cfg = resolve_meta_config(args)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2

    progress = ProgressWriter(Path(cfg["progress"]))
    progress.emit(
        "run_start",
        model_id=cfg["model_id"],
        mode="meta",
        dataset=cfg["dataset"],
        window=cfg["window"],
        symbols_total=len(cfg["symbols"]),
        objective=cfg["objective"],
        train_months=cfg["train_months"],
        test_months=cfg["test_months"],
        embargo_days=cfg["embargo_days"],
        model_family=cfg["model_family"],
        calibrate=cfg["calibrate"],
    )
    try:
        from .meta import train_meta as meta_train  # Doc 2
        written = meta_train.run(cfg, progress)
    except Exception as e:
        progress.emit("error", message=f"{type(e).__name__}: {e}", trace=traceback.format_exc())
        progress.close()
        print(f"meta training failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    progress.emit("run_done", artifact=str(written or cfg["output"]))
    progress.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

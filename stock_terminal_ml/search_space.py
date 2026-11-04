"""Optuna search space for Dev Signal.

The v1 pipeline tunes six knobs. Everything else on `Params` stays at the backend's
default — the `--widen` flag unlocks the frozen set listed in the main plan.
"""
from __future__ import annotations

from app.integrations.signals.config import Params

# Six v1 knobs (log-scaled where marked).
TUNED_KEYS = (
    "touch_tolerance_pct",
    "break_tolerance_pct",
    "extension_pct_min",
    "retest_max_dip_pct",
    "cooldown_bars",
    "min_move_pct",
)

# Knobs unlocked by --widen (defaults otherwise).
WIDEN_KEYS = (
    "retest_tolerance_pct",
    "max_bars_to_retest",
    "break_lookback_bars",
    "flip_min_move_pct",
)


# key -> (kind, low, high, log). Single source for `suggest` and the dashboard.
RANGES: dict[str, tuple[str, float, float, bool]] = {
    "touch_tolerance_pct":  ("float", 0.03, 0.3, True),
    "break_tolerance_pct":  ("float", 0.03, 0.3, True),
    "extension_pct_min":    ("float", 0.05, 0.6, True),
    "retest_max_dip_pct":   ("float", 0.05, 0.5, True),
    "cooldown_bars":        ("int",   0,    30,  False),
    "min_move_pct":         ("float", 0.1,  1.0, True),
    # --widen
    "retest_tolerance_pct": ("float", 0.03, 0.4, True),
    "max_bars_to_retest":   ("int",   3,    20,  False),
    "break_lookback_bars":  ("int",   10,   60,  False),
    "flip_min_move_pct":    ("float", 0.1,  1.5, True),
}


def suggest(trial, widen: bool = False) -> Params:
    """Produce a `Params` instance from an Optuna trial."""
    kwargs = {}
    for key in tuned_keys(widen):
        kind, lo, hi, log = RANGES[key]
        if kind == "int":
            kwargs[key] = trial.suggest_int(key, int(lo), int(hi))
        else:
            kwargs[key] = trial.suggest_float(key, lo, hi, log=log)
    return Params(**kwargs)


def tuned_keys(widen: bool = False) -> tuple[str, ...]:
    return TUNED_KEYS + (WIDEN_KEYS if widen else ())

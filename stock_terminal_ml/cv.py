"""Expanding-window cross validation over intraday candles.

The walk-forward pattern used here splits the user window by trading day into
`k+1` sub-slices; each fold trains on the expanding-left prefix and validates on
the next slice (the last slice is reserved for OOS and is not seen during search).

A `holdout` helper returns the `trading_days` days immediately after the user
window — the tuner scores the final best params there exactly once.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Sequence

from app.integrations.kite.session import IST


def _bar_date(ts_utc: int) -> date:
    return datetime.fromtimestamp(ts_utc, IST).date()


def _by_day(bars: Sequence[dict]) -> list[list[dict]]:
    """Group bars into one list per IST trading day, preserving order."""
    out: list[list[dict]] = []
    cur_day: date | None = None
    for b in bars:
        d = _bar_date(b["time"])
        if d != cur_day:
            out.append([])
            cur_day = d
        out[-1].append(b)
    return out


def kfold(bars: Sequence[dict], folds: int = 3) -> list[tuple[list[dict], list[dict]]]:
    """Expanding-window folds inside the user window.

    For a window with D trading days and `folds=k`, the slice size is D // (k+1).
    Fold i (1..k) trains on days [:i*slice] and validates on [i*slice:(i+1)*slice].
    """
    if folds < 1:
        raise ValueError("folds must be ≥ 1")
    days = _by_day(bars)
    n = len(days)
    if n < folds + 1:
        return []
    size = n // (folds + 1)
    out: list[tuple[list[dict], list[dict]]] = []
    for i in range(1, folds + 1):
        train_end = i * size
        val_end = min((i + 1) * size, n)
        train = [b for d in days[:train_end] for b in d]
        val = [b for d in days[train_end:val_end] for b in d]
        if train and val:
            out.append((train, val))
    return out


def holdout(bars: Sequence[dict], window_end_iso: str,
            trading_days: int = 30) -> list[dict]:
    """The next `trading_days` IST days after `window_end_iso` (inclusive-of-day).

    `bars` is the *full* symbol candle list (not just the training window). The
    caller supplies the user window's `to=` date string; this returns the bars
    whose IST date is strictly after that date, capped at `trading_days`.
    """
    end = date.fromisoformat(window_end_iso)
    days = _by_day(bars)
    oos_days = [d for d in days if d and _bar_date(d[0]["time"]) > end]
    sliced = oos_days[:trading_days]
    return [b for d in sliced for b in d]


def warmup_bars(full_bars: Sequence[dict], window_start_iso: str,
                warmup_days: int = 2) -> list[dict]:
    """Return the `warmup_days` trading days immediately before the user window.

    Day-1 CPR needs the previous day's daily H/L/C — the warm-up bars make sure
    those are present when `build_cpr_by_day` runs.
    """
    start = date.fromisoformat(window_start_iso)
    days = _by_day(full_bars)
    before = [d for d in days if d and _bar_date(d[0]["time"]) < start]
    tail = before[-warmup_days:] if warmup_days > 0 else []
    return [b for d in tail for b in d]

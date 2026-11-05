"""Purged, embargoed walk-forward cross-validation over IST trading days.

The v2 pipeline uses month-sized blocks (~21 trading days) built from the user
window. With `train_months=T` and `test_months=V` (defaults T=3, V=1):

    fold 0: train = months [0 .. T),               test = month [T .. T+V)
    fold 1: train = months [0 .. T+V) or [V .. T+V),  test = month [T+V .. T+2V)
    ...roll by V until the data is exhausted.

Expanding (default) grows the train set; sliding keeps it fixed at T months.

Purge/embargo: labels are intraday (resolve same session), so a label can only
cross into the test set at the boundary. We drop the last `embargo_days` IST
trading days *before* each test start from training, which is the full
"purge + embargo" these intraday trades need. See §3 of the methodology plan.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterator, Sequence


DAYS_PER_MONTH = 21  # NSE trading days per calendar month, approx


@dataclass(frozen=True)
class Fold:
    index: int
    train_days: tuple[date, ...]
    test_days: tuple[date, ...]

    @property
    def train_from(self) -> date: return self.train_days[0]
    @property
    def train_to(self) -> date: return self.train_days[-1]
    @property
    def test_from(self) -> date: return self.test_days[0]
    @property
    def test_to(self) -> date: return self.test_days[-1]


def walk_forward(trading_days: Sequence[date], *, train_months: int = 3,
                 test_months: int = 1, embargo_days: int = 1,
                 mode: str = "expanding") -> list[Fold]:
    """Walk-forward splits by trading-day block. Each test window is
    `test_months * DAYS_PER_MONTH` consecutive days; train is the preceding
    `train_months * DAYS_PER_MONTH` days, trimmed by `embargo_days` at the seam.

    `mode`: "expanding" anchors the train start at day 0; "sliding" keeps it
    fixed at `train_months`.
    """
    if train_months < 1 or test_months < 1 or embargo_days < 0:
        raise ValueError("train_months/test_months must be >= 1; embargo_days >= 0")
    if mode not in ("expanding", "sliding"):
        raise ValueError(f"unknown mode {mode!r}")
    days = sorted(set(trading_days))
    train_n = train_months * DAYS_PER_MONTH
    test_n = test_months * DAYS_PER_MONTH
    n = len(days)
    folds: list[Fold] = []
    start = train_n
    idx = 0
    while start + test_n <= n:
        test_lo, test_hi = start, start + test_n
        if mode == "expanding":
            train_lo = 0
        else:
            train_lo = max(0, test_lo - train_n)
        train_hi = max(train_lo, test_lo - embargo_days)
        tr = tuple(days[train_lo:train_hi])
        te = tuple(days[test_lo:test_hi])
        if tr and te:
            folds.append(Fold(index=idx, train_days=tr, test_days=te))
            idx += 1
        start += test_n
    # Trailing partial test block: if there is anything left and at least a
    # week in it, make a final fold (common on ~10-month windows).
    if start < n and (n - start) >= 5:
        test_lo, test_hi = start, n
        if mode == "expanding":
            train_lo = 0
        else:
            train_lo = max(0, test_lo - train_n)
        train_hi = max(train_lo, test_lo - embargo_days)
        tr = tuple(days[train_lo:train_hi])
        te = tuple(days[test_lo:test_hi])
        if tr and te:
            folds.append(Fold(index=idx, train_days=tr, test_days=te))
    return folds


def split_labels_by_day(labels, day_of) -> dict:
    """Index `labels` by their IST date via the `day_of(label) -> date` helper.

    Used by the trainer to materialize `(X_train, y_train)` from a fold without
    walking the full bar list again.
    """
    out: dict = {}
    for L in labels:
        d = day_of(L)
        out.setdefault(d, []).append(L)
    return out

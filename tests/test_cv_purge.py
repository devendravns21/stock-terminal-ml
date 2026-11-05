"""`cv.walk_forward` never lets a train day sit within `embargo_days` of its
test start, and train/test day sets stay disjoint fold-by-fold.
"""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from stock_terminal_ml.meta.cv import DAYS_PER_MONTH, walk_forward


def _trading_days(n: int, start: date = date(2026, 1, 5)) -> list[date]:
    out = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:  # Mon–Fri
            out.append(d)
        d += timedelta(days=1)
    return out


def test_no_train_day_within_embargo_of_test_start():
    days = _trading_days(DAYS_PER_MONTH * 6)
    folds = walk_forward(days, train_months=3, test_months=1, embargo_days=2)
    assert folds, "expected at least one fold with 6 months of days"
    for fold in folds:
        gap = min((fold.test_from - d).days for d in fold.train_days)
        assert gap >= 2, f"fold {fold.index} has train day {fold.train_from} within embargo of {fold.test_from}"


def test_train_and_test_days_disjoint_per_fold():
    days = _trading_days(DAYS_PER_MONTH * 8)
    folds = walk_forward(days, train_months=3, test_months=1, embargo_days=1)
    for fold in folds:
        assert not (set(fold.train_days) & set(fold.test_days))


def test_expanding_grows_sliding_stays_fixed():
    days = _trading_days(DAYS_PER_MONTH * 7)
    exp = walk_forward(days, train_months=3, test_months=1, embargo_days=0, mode="expanding")
    sli = walk_forward(days, train_months=3, test_months=1, embargo_days=0, mode="sliding")
    assert exp and sli
    # expanding: train size non-decreasing
    sizes = [len(f.train_days) for f in exp]
    assert sizes == sorted(sizes)
    # sliding: all fold train sizes identical (= train_months * DAYS_PER_MONTH)
    assert all(len(f.train_days) == 3 * DAYS_PER_MONTH for f in sli)


def test_rejects_bad_params():
    with pytest.raises(ValueError):
        walk_forward(_trading_days(100), train_months=0)
    with pytest.raises(ValueError):
        walk_forward(_trading_days(100), embargo_days=-1)
    with pytest.raises(ValueError):
        walk_forward(_trading_days(100), mode="weird")

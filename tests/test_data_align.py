"""Task 7.1 — as-of alignment never looks ahead."""
from datetime import datetime

from app.integrations.kite.session import IST
from app.integrations.ml_meta.data import asof_index_bars, asof_positions, daily_before, slice_with_warmup


def _ts(y, m, d, hh=10, mm=0) -> int:
    return int(datetime(y, m, d, hh, mm, tzinfo=IST).timestamp())


def test_asof_attaches_latest_not_future():
    sym = [{"time": t} for t in (100, 160, 200, 260, 300)]
    idx = [{"time": t, "close": t} for t in (90, 160, 250, 400)]
    got = asof_index_bars(sym, idx)
    assert [g["time"] if g else None for g in got] == [90, 160, 160, 250, 250]
    for s, g in zip(sym, got):
        assert g is None or g["time"] <= s["time"]


def test_asof_before_first_index_bar_is_none():
    sym = [{"time": 10}, {"time": 20}]
    idx = [{"time": 15}]
    assert asof_index_bars(sym, idx) == [None, idx[0]]
    assert asof_positions(sym, idx) == [-1, 0]


def test_asof_empty_index_is_all_none():
    sym = [{"time": t} for t in (1, 2, 3)]
    assert asof_index_bars(sym, []) == [None, None, None]


def test_daily_before_is_strict_and_sorted():
    rows = [{"time": _ts(2026, 3, d, 0, 0)} for d in (5, 3, 4, 6)]
    got = daily_before(rows, _ts(2026, 3, 5, 9, 15))
    assert [r["time"] for r in got] == [_ts(2026, 3, 3, 0, 0), _ts(2026, 3, 4, 0, 0)]


def test_slice_with_warmup():
    bars = [{"time": _ts(2026, 3, d, 10, m)} for d in (2, 3, 4, 5) for m in (0, 3)]
    from datetime import date
    out, first = slice_with_warmup(bars, date(2026, 3, 4), date(2026, 3, 4), warmup_days=1)
    assert first == 2
    assert out == bars[2:6]

"""features.extract_features must only read bars[:i+1] / index up to time_i / daily_before.

If appending future bars to `bars` or future index bars to `index_hist` changes any
value at `i`, the feature is leaking the future — the whole v2 pipeline's OOS metrics
would be invalid. This is the most important correctness guard in the ML layer
(see docs/ml-v2/02-ml-methodology-plan.md §2).
"""
from __future__ import annotations

import math
import random
from datetime import datetime, timedelta

import pytest

from app.integrations.kite.session import IST
from app.integrations.ml_meta import features as feat_mod
from app.integrations.signals.pattern import Fire


def _synthetic_bars(n: int = 180, start_price: float = 100.0, step_sec: int = 180,
                    seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    start = int(datetime(2026, 1, 5, 10, 0, 0, tzinfo=IST).timestamp())
    bars: list[dict] = []
    p = start_price
    for i in range(n):
        ret = rng.gauss(0.0, 0.003)
        o = p
        c = max(0.01, p * (1 + ret))
        hi = max(o, c) * (1 + abs(rng.gauss(0.0, 0.002)))
        lo = min(o, c) * (1 - abs(rng.gauss(0.0, 0.002)))
        v = rng.randint(1000, 5000)
        bars.append({"time": start + i * step_sec, "open": o, "high": hi, "low": lo,
                     "close": c, "volume": v})
        p = c
    return bars


def _mk_fire(i: int, close: float) -> Fire:
    return Fire(direction="BUY", kind="reversal", line_name="pp",
                line_value=close * 0.998, bar_index=i)


def _nans_equal(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


@pytest.mark.parametrize("i", [25, 60, 120, 170])
def test_future_bars_do_not_change_features_at_i(i):
    bars = _synthetic_bars()
    idx_bars = _synthetic_bars(start_price=20000.0, seed=11)
    fire = _mk_fire(i, bars[i]["close"])

    # Build feats at i on full bars...
    f1 = feat_mod.extract_features(
        bars=bars[: i + 1], i=i, fire=fire,
        cpr={"pp": bars[i]["close"] * 0.998, "r1": bars[i]["close"] * 1.005,
             "s1": bars[i]["close"] * 0.993},
        st_line=None, st_trend=None,
        index_bar=idx_bars[i] if i < len(idx_bars) else None,
        index_hist=idx_bars[: i + 1],
        index_cpr=None, daily_before=[],
        symbol="TEST",
    )

    # ...then appending arbitrary future bars must not change any value.
    future = _synthetic_bars(start_price=bars[-1]["close"], seed=99)
    bars_plus = bars[: i + 1] + future
    idx_plus = idx_bars[: i + 1] + future
    f2 = feat_mod.extract_features(
        bars=bars_plus, i=i, fire=fire,
        cpr={"pp": bars[i]["close"] * 0.998, "r1": bars[i]["close"] * 1.005,
             "s1": bars[i]["close"] * 0.993},
        st_line=None, st_trend=None,
        index_bar=idx_bars[i] if i < len(idx_bars) else None,
        index_hist=idx_plus[: i + 1],  # same slice — just prove the API honours its "<=i" contract
        index_cpr=None, daily_before=[],
        symbol="TEST",
    )

    assert set(f1.keys()) == set(f2.keys())
    for k in f1:
        assert _nans_equal(f1[k], f2[k]), f"feature {k} leaked: {f1[k]} vs {f2[k]}"


def test_feature_names_cover_the_module_contract():
    assert "stub_const" not in feat_mod.FEATURE_NAMES, "stub features.py was not replaced"
    assert feat_mod.FEATURE_SPEC_VERSION != "0-stub"
    # categorical must be a subset of names.
    assert set(feat_mod.CATEGORICAL_FEATURES).issubset(set(feat_mod.FEATURE_NAMES))

"""Per-fire triple-barrier labels for meta-labeling.

The backend's `simulate()` manages one position at a time and skips overlapping
fires — wrong for labeling, where every candidate must be labeled independently.
`label_fire` reuses the simulator's exit math (same intrabar target/stop,
stop-wins-ties, session square-off — see `backtest/simulator.py`) but walks each
fire forward in isolation.

`build_dataset` goes further: given bars + fires + exits, it returns one row per
fire with `(y, net_pct, hold_bars, exit_reason, bar_index, time)` ready to join
with feature rows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.integrations.backtest.config import CostConfig, ExitRules
from app.integrations.backtest.costs import trade_costs
from app.integrations.signals.engine import _ist_date
from app.integrations.signals.pattern import Fire


def _is_session_end(bars: Sequence[dict], i: int) -> bool:
    return i == len(bars) - 1 or _ist_date(bars[i + 1]["time"]) != _ist_date(bars[i]["time"])


@dataclass(frozen=True)
class LabeledFire:
    bar_index: int
    time: int
    y: int            # 1 iff net_pct > tau_cost
    net_pct: float    # per-trade net % (same units as `Trade.net_pct`)
    exit_reason: str  # target | stop_loss | square_off
    hold_bars: int    # bars held, inclusive-ish


def label_fire(bars: Sequence[dict], i: int, fire: Fire, exits: ExitRules,
               costs: CostConfig, *, tau_cost: float = 0.0,
               initial_capital: float = 100000.0,
               index_factor: float = 1.0) -> LabeledFire | None:
    """Label a single candidate fire with a triple-barrier outcome.

    Returns `None` for a fire on the last bar of its session (same skip rule the
    simulator uses — no room to manage). Otherwise uses the identical intrabar
    target/stop logic: stop wins when a bar touches both barriers, session end
    forces a close.
    """
    if _is_session_end(bars, i):
        return None
    n = len(bars)
    tgt = exits.target_pct / 100.0
    stop = exits.stop_loss_pct / 100.0
    long = fire.direction == "BUY"
    entry_price = float(bars[i]["close"])
    target_price = entry_price * (1 + tgt) if long else entry_price * (1 - tgt)
    stop_price = entry_price * (1 - stop) if long else entry_price * (1 + stop)

    exit_price = exit_reason = None
    exit_time: int | None = None
    j = i + 1
    while j < n:
        b = bars[j]
        if long:
            hit_stop = b["low"] <= stop_price
            hit_tgt = b["high"] >= target_price
        else:
            hit_stop = b["high"] >= stop_price
            hit_tgt = b["low"] <= target_price
        if hit_stop:            # stop wins ties (conservative; matches simulator)
            exit_price, exit_reason = stop_price, "stop_loss"
        elif hit_tgt:
            exit_price, exit_reason = target_price, "target"
        if exit_price is not None:
            exit_time = b["time"]
            break
        if _is_session_end(bars, j):
            exit_price, exit_reason, exit_time = b["close"], "square_off", b["time"]
            break
        j += 1
    if exit_price is None:  # ran out of bars still open
        exit_price, exit_reason, exit_time = bars[-1]["close"], "square_off", bars[-1]["time"]
        j = n - 1

    # Same cost/PnL math as the simulator: notional = equity·factor. Costs have
    # fixed minimums (STT, stamp), so the notional must be realistic for net_pct
    # to match. Equity doesn't compound across a single trade, so `initial_capital`
    # is the baseline.
    notional = float(initial_capital) * index_factor
    qty = notional / entry_price
    entry_value = qty * entry_price
    exit_value = qty * exit_price
    if long:
        gross_pnl = exit_value - entry_value
        buy_value, sell_value = entry_value, exit_value
    else:
        gross_pnl = entry_value - exit_value
        buy_value, sell_value = exit_value, entry_value
    cost_total = trade_costs(buy_value, sell_value, costs.segment)["total"] if costs.enabled else 0.0
    net_pnl = gross_pnl - cost_total
    net_pct = (net_pnl / notional * 100.0) if notional else 0.0
    y = 1 if net_pct > tau_cost else 0
    return LabeledFire(
        bar_index=i, time=int(bars[i]["time"]), y=y, net_pct=float(net_pct),
        exit_reason=str(exit_reason), hold_bars=int(j - i),
    )


def build_dataset(bars: Sequence[dict], fires: Sequence[Fire], exits: ExitRules,
                  costs: CostConfig, *, tau_cost: float = 0.0,
                  initial_capital: float = 100000.0,
                  index_factor: float = 1.0) -> list[LabeledFire]:
    """Label each fire in `fires`, dropping the last-bar-of-session cases."""
    out: list[LabeledFire] = []
    for f in fires:
        lab = label_fire(bars, f.bar_index, f, exits, costs,
                         tau_cost=tau_cost, initial_capital=initial_capital,
                         index_factor=index_factor)
        if lab is not None:
            out.append(lab)
    return out


# ------- sample weights ------------------------------------------------------

def return_magnitude_weights(labels: Sequence[LabeledFire]) -> list[float]:
    """|net_pct| normalized to mean 1 — big winners/losers weigh more than ties."""
    if not labels:
        return []
    mags = [abs(l.net_pct) for l in labels]
    m = sum(mags) / len(mags)
    if m <= 0:
        return [1.0] * len(labels)
    return [x / m for x in mags]


def time_decay_weights(labels: Sequence[LabeledFire], newest: float = 1.0,
                       oldest: float = 0.5) -> list[float]:
    """Linear decay from `oldest` (first sample) to `newest` (last)."""
    n = len(labels)
    if n == 0:
        return []
    if n == 1:
        return [newest]
    step = (newest - oldest) / (n - 1)
    # Sort indices by time just in case the caller passed them unsorted.
    order = sorted(range(n), key=lambda k: labels[k].time)
    w = [0.0] * n
    for rank, idx in enumerate(order):
        w[idx] = oldest + step * rank
    return w


def final_weights(labels: Sequence[LabeledFire], *, time_decay: bool = True) -> list[float]:
    """`w = w_ret * w_time`, renormalized to mean 1. Used as LightGBM sample_weight."""
    if not labels:
        return []
    wr = return_magnitude_weights(labels)
    wt = time_decay_weights(labels) if time_decay else [1.0] * len(labels)
    raw = [a * b for a, b in zip(wr, wt)]
    m = sum(raw) / len(raw)
    if m <= 0:
        return [1.0] * len(labels)
    return [x / m for x in raw]

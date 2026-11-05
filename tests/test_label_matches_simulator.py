"""`label_fire` must reproduce the simulator's net_pct / exit_reason for a single trade.

Build a tiny synthetic bars sequence with one planted fire, then compare
`label_fire(...).net_pct` to the `simulate([fire])` output's trade.net_pct and
exit_reason. They must match to the simulator's rounding precision.
"""
from __future__ import annotations

from datetime import datetime

from app.integrations.backtest.config import CostConfig, ExitRules
from app.integrations.backtest.simulator import simulate
from app.integrations.kite.session import IST
from app.integrations.signals.config import Params
from app.integrations.signals.pattern import Fire

from stock_terminal_ml.meta.labels import label_fire


def _bars_hitting_target_up():
    """BUY entry at bar 2's close; bar 3's high hits +1% target."""
    start = int(datetime(2026, 1, 5, 10, 0, 0, tzinfo=IST).timestamp())
    tf = 180  # 3-minute bars
    return [
        {"time": start,          "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1000},
        {"time": start + 1 * tf, "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1000},
        {"time": start + 2 * tf, "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1000},  # entry here
        {"time": start + 3 * tf, "open": 100, "high": 102, "low": 100, "close": 101, "volume": 1000},  # +1% target hit
        {"time": start + 4 * tf, "open": 101, "high": 101, "low": 101, "close": 101, "volume": 1000},
    ]


def _bars_hitting_stop_down():
    """BUY entry at bar 2's close; bar 3's low hits -0.5% stop."""
    start = int(datetime(2026, 1, 5, 10, 0, 0, tzinfo=IST).timestamp())
    tf = 180
    return [
        {"time": start,          "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1000},
        {"time": start + 1 * tf, "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1000},
        {"time": start + 2 * tf, "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1000},  # entry
        {"time": start + 3 * tf, "open": 100, "high": 100, "low": 99,  "close": 99.6, "volume": 1000},  # -0.5% stop
        {"time": start + 4 * tf, "open": 99.6, "high": 99.6, "low": 99.6, "close": 99.6, "volume": 1000},
    ]


def _run_both(bars):
    fire = Fire(direction="BUY", kind="reversal", line_name="pp", line_value=100.0, bar_index=2)
    exits = ExitRules(target_pct=1.0, stop_loss_pct=0.5)
    costs = CostConfig(enabled=True, segment="EQUITY_INTRADAY")
    trades, _ = simulate(bars, [fire], Params(),
                         initial_capital=100000.0, index_factor=1.0,
                         exits=exits, costs=costs)
    assert len(trades) == 1
    sim_trade = trades[0]
    lab = label_fire(bars, 2, fire, exits, costs, tau_cost=0.0, index_factor=1.0)
    assert lab is not None
    return sim_trade, lab


def test_target_hit_matches_simulator():
    sim, lab = _run_both(_bars_hitting_target_up())
    assert sim.exit_reason == "target" and lab.exit_reason == "target"
    assert abs(sim.net_pct - lab.net_pct) < 1e-2  # simulator rounds to 4dp; feat ok within 1 bp


def test_stop_hit_matches_simulator():
    sim, lab = _run_both(_bars_hitting_stop_down())
    assert sim.exit_reason == "stop_loss" and lab.exit_reason == "stop_loss"
    assert abs(sim.net_pct - lab.net_pct) < 1e-2


def test_last_bar_of_session_is_skipped():
    bars = _bars_hitting_target_up()
    fire = Fire(direction="BUY", kind="reversal", line_name="pp", line_value=100.0, bar_index=len(bars) - 1)
    exits = ExitRules(target_pct=1.0, stop_loss_pct=0.5)
    costs = CostConfig(enabled=True, segment="EQUITY_INTRADAY")
    assert label_fire(bars, len(bars) - 1, fire, exits, costs) is None

"""stock-terminal-ml — Optuna tuner for Dev Signal.

Reuses the backend's pattern evaluator, trade simulator and metrics verbatim so
backtests, training and live signals all share one code path. Only parameter
values differ.
"""
__version__ = "0.1.0"

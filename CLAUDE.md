# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An Optuna tuner that searches for per-symbol parameters for the "Dev Signal" intraday strategy and writes a model artifact JSON. The parent repo `../stock-terminal` consumes these artifacts (backend registry → `/api/ml/models` → frontend indicator "Dev Signal MLN").

**Hard dependency on a sibling repo:** almost every module imports `app.*` from `../stock-terminal/backend`, installed as an editable package (`stock-terminal-backend`, see `[tool.uv.sources]` in [pyproject.toml](pyproject.toml)). Strategy, simulator, metrics, `Params`, offline candle store and `settings` all come from there. Nothing runs without that checkout. Training and the live backend share one code path, so don't reimplement strategy or backtest logic here. Only the parameter values differ.

## Commands

```bash
uv sync                                    # install (or: pip install -e ../stock-terminal/backend && pip install -e .)
.venv/bin/python -m stock_terminal_ml.dashboard          # local UI at :8765 (--port, --host, --no-browser)
.venv/bin/python -m stock_terminal_ml.train --window 2026-06-01:2026-08-31 --preset fast --symbols RELIANCE --jobs 1 --output /tmp/x.json
pytest -q                                  # tests
pytest tests/test_smoke.py::test_smoke -q  # single test
```

There is no linter or formatter configured.

The only test is an end-to-end smoke test (10 trials, 1 symbol). It **skips silently** when no offline dataset exists on disk, so a "pass" may mean it didn't run at all. Use `-rs` to see skips. It is the guard against `app.*` import drift.

Use `--jobs 1` when debugging. It runs serially in-process. Otherwise the trainer uses a `spawn` multiprocessing Pool, and exceptions inside `_tune_symbol` are caught and turned into `error` progress events, not tracebacks.

## Architecture

**Training pipeline ([train.py](stock_terminal_ml/train.py)).** Each symbol gets its own Optuna study (TPE, seed 42) running in a Pool worker. One trial runs the backend's `run_strategy → simulate → compute_metrics`, then an objective from [objectives.py](stock_terminal_ml/objectives.py). The trial score averages the train and val scores across the expanding-window folds from [cv.py](stock_terminal_ml/cv.py). After the search:
- the best params are re-scored per fold (`cv`);
- they are scored once on the 30 trading days *after* the user window (`oos`). The data must extend past `--window` for OOS to exist.

The global fallback is the median of each tuned param across symbols (int knobs are rounded).

**Search space ([search_space.py](stock_terminal_ml/search_space.py)).** `RANGES` is the single source of truth for both `suggest()` and the dashboard. The six `TUNED_KEYS` are always tuned. `WIDEN_KEYS` are tuned only with `--widen`. All other `Params` fields keep the backend defaults. If you add an int knob, also add it to the int set in `_median_params`.

**Config precedence quirk.** `--config` JSON is overridden by CLI flags, but the check is "is the flag still at its argparse default?" (e.g. `target_pct != 1.0`). So passing a flag explicitly *at* its default value doesn't override the config. Also, `--dataset` always has a default, so `config.dataset` is effectively ignored.

**Progress protocol ([progress.py](stock_terminal_ml/progress.py)).** Progress is an append-only, fsync'd JSONL stream with events `run_start`, `trial`, `symbol_done`, `run_done` and `error`. Two consumers tail it: the backend's `/api/ml/train` route (it spawns this CLI with `--config` + `--output` from `backend/.ml_jobs/<id>/`) and this repo's dashboard. Changing event names or fields breaks both. Setting `ML_PROGRESS_ECHO=1` also prints a human-readable line per event to stderr.

**Artifact ([artifact.py](stock_terminal_ml/artifact.py)).** Schema version `"1"` with keys `id`, `label`, `created_at`, `training`, `per_symbol{SYM:{params,cv,oos,best_value}}` and `global_fallback.params`. It is written atomically (`.tmp` → `os.replace`). The shape must match the backend's `docs/backend/ml-pipeline.md` and registry loader.

**Dashboard ([dashboard.py](stock_terminal_ml/dashboard.py) + [dashboard/index.html](dashboard/index.html)).** A stdlib `ThreadingHTTPServer` with a JSON API (documented in the module docstring) serves a single static page that polls that API.
- `POST /api/runs` validates input and writes `outputs/runs/<YYYYmmdd-HHMMSS>/{config.json,run.sh}`, then opens a **new Terminal window** (osascript on macOS) that runs the trainer CLI. The dashboard never runs training in-process. It learns run state from files in the run dir: `pid`, `progress.jsonl`, `terminal.log`, `exit_code` and `cancelled`. Stop sends SIGTERM to the process group.
- With publish on, output goes straight into the backend's `settings.ml_models_dir`. With it off, output stays in the run dir.
- `next_model_id()` allocates `dev_signal_mlN` by scanning existing models, dashboard runs and the backend's `.ml_jobs/` so IDs never collide.
- `MAIN_BACKEND_URL` (default `:8000`) and `MAIN_FRONTEND_URL` (default `:5173`) are only used for the pipeline/status view.

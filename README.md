# stock-terminal-ml

Optuna-driven Dev Signal parameter tuner. Reads candles from the main
stock-terminal backend's offline snapshots and writes model JSON into
`backend/app/integrations/signals/ml_models/`.

## Install

Recommended (uv):

```bash
cd stock-terminal-ml
uv sync
```

Plain-pip fallback (the backend's `app` package must be importable):

```bash
cd stock-terminal-ml
python3 -m venv .venv && source .venv/bin/activate
pip install -e ../stock-terminal/backend
pip install -e .
```

## Dashboard (start here)

```bash
cd stock-terminal-ml
.venv/bin/python -m stock_terminal_ml.dashboard      # → http://localhost:8765 (opens the browser)
```

A static page (`dashboard/index.html`) served by a small stdlib server (`stock_terminal_ml/dashboard.py`).
No Kite login is needed: it trains on the offline snapshots.

- **Inputs**: dataset, window (quick 2/3/4/6M buttons end ~45 days before the data's end so the
  30-trading-day OOS hold-out fits), target/stop %, preset, trials override, objective, widen, workers,
  symbols, model name, "Publish to main repo". It shows the search space and the exact command.
- **Start** writes `outputs/runs/<run_id>/{config.json,run.sh}` and opens a **new Terminal window**
  (macOS `osascript`; xterm-style fallbacks elsewhere) running the trainer with `ML_PROGRESS_ECHO=1`,
  so each trial prints there. Output is also tee'd to `terminal.log`.
- **Progress** polls `progress.jsonl` once a second: overall bar, ETA, a per-symbol grid (click one for its
  trial-score chart), terminal output, events, and a Stop button (SIGTERM to the run's process group).
- **Pipeline** shows, live, how the run becomes an indicator: config → terminal → Optuna → artifact in
  `backend/app/integrations/signals/ml_models/` → backend `/api/ml/models` (checked at `MAIN_BACKEND_URL`,
  default `http://localhost:8000`) → frontend `Dev Signal MLN` (reload open app tabs) → chart link.
- **Results**: every model with CV vs OOS per symbol (sortable, overfit flag), tuned params vs defaults,
  global fallback, raw JSON and download.

Flags: `--port`, `--host`, `--no-browser`. Run state lives in `outputs/runs/` (gitignored).
Model ids (`dev_signal_mlN`) are allocated across existing models, dashboard runs and the backend's
`.ml_jobs/`, so runs from the dashboard and from the app's ML Training view never collide.

## CLI

```bash
python -m stock_terminal_ml.train \
  --window 2026-06-01:2026-08-31 \
  --preset fast \
  --symbols RELIANCE,HDFCBANK \
  --output /path/to/dev_signal_ml1.json
```

Flags: `--window FROM:TO`, `--target-pct`, `--sl-pct`, `--preset {fast,standard,thorough}`,
`--symbols`, `--output`, `--config`, `--progress`, `--objective`, `--widen`,
`--trials` (override trials-per-symbol).

With no `--symbols`, all symbols in the configured offline dataset are tuned.

## How it integrates with the backend

The backend's `/api/ml/train` route writes a `config.json` into
`backend/.ml_jobs/<id>/`, spawns this CLI with `--config` + `--output`, and tails the
emitted `progress.jsonl`. When the run finishes the artifact lands in
`ml_models_dir`, where the registry picks it up.

## Tests

```bash
pytest -q
```

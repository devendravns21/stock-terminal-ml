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

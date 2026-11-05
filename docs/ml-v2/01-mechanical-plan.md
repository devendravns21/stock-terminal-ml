# Dev Signal ML v2 — Mechanical Implementation Plan (Doc 1 of 2)

> **Audience:** a low-reasoning coding model (and the user as reviewer). Every task here is
> deterministic: exact files, exact signatures, exact JSON shapes, and an acceptance check you
> can verify without making ML/statistics judgment calls. **Do not invent feature math, label
> logic, cross-validation, or training here** — those live in Doc 2
> ([`02-ml-methodology-plan.md`](02-ml-methodology-plan.md)) and are owned by the high-reasoning
> model. You implement the plumbing *around* them against the frozen interfaces in Appendix A.

## 0. What is being built (one paragraph)

The existing "Dev Signal" is a rule-based intraday pattern detector. We are adding a **meta-labeling
ML layer**: the rule still produces candidate BUY/SELL fires, and a trained **LightGBM** classifier
scores each fire with a probability of being profitable; fires below a stored threshold are dropped.
The model + feature list + threshold are saved in a **v2 model artifact**. A new **server-side scorer**
applies the model for both backtests and a new **live scoring endpoint** the chart calls. v1 artifacts
(plain threshold overrides) must keep working unchanged.

You are building: the data-alignment helpers, the artifact v2 read/write, the inference scorer wrapper,
the FastAPI scoring endpoint, the backtest/registry v2 routing, the frontend indicator that fetches
server-scored markers, the trainer CLI/config/progress glue, and the tests. Doc 2 fills in
`features.py`, `labels.py`, `cv.py`, `train_meta.py`, `evaluate.py`.

**Repos involved** (paths are relative to each repo root):
- ML repo: `stock-terminal-ml/` (this repo).
- Backend: `../stock-terminal/backend/` — importable as `app.*` (editable install).
- Frontend: `../stock-terminal/frontend/`.

---

## 1. Dependencies & config

**Task 1.1 — Add ML dependencies to the ML repo.**
Edit [`pyproject.toml`](../../pyproject.toml), `[project].dependencies`, add:
```toml
"lightgbm>=4.3",
"scikit-learn>=1.4",
"numpy>=1.26",
"pandas>=2.2",
```
`optuna` is already present. Then run `uv sync` (or `pip install -e .`).
**Accept:** `.venv/bin/python -c "import lightgbm, sklearn, numpy, pandas"` exits 0.

**Task 1.2 — Add the same deps to the backend** so the live scorer can import them.
Edit `../stock-terminal/backend/pyproject.toml` (or `requirements`), add `lightgbm>=4.3`,
`numpy>=1.26`. (scikit-learn only if the calibration object needs it at serve time — see Appendix A
`MetaScorer`; if calibration is stored as plain arrays, sklearn is **not** required at serve time.
Prefer that: keep the backend serve-path free of sklearn.)
**Accept:** backend venv imports `lightgbm` and `numpy`.

---

## 2. Backend: shared data + inference modules

Create a new package `../stock-terminal/backend/app/integrations/ml_meta/`. It is imported by **both**
the live backend and the trainer (the trainer already imports `app.*`), guaranteeing train/serve parity.

Files **you** create in this package: `__init__.py`, `data.py`, `scorer.py`, `schema.py`.
File **Doc 2** creates in this package: `features.py`. You may create a **stub** `features.py` (see
Task 2.4) so your code imports and tests run before Doc 2 lands; Doc 2 replaces the stub body.

**Task 2.1 — `ml_meta/schema.py`: artifact v2 constants + validation.**
Implement exactly the schema in Appendix B. Provide:
```python
ARTIFACT_VERSION = "2"
ARTIFACT_KIND = "meta_labeling"

def validate_v2(artifact: dict) -> None:
    """Raise ValueError if any required key is missing or mistyped (see Appendix B)."""

def is_v2(artifact: dict) -> bool:
    return str(artifact.get("version")) == ARTIFACT_VERSION and artifact.get("kind") == ARTIFACT_KIND
```
**Accept:** `validate_v2` passes on the Appendix B example and raises on a dict missing `model.booster`.

**Task 2.2 — `ml_meta/data.py`: time alignment helpers.**
Pure functions, no I/O beyond what callers pass in. Needed so features can see index context.
```python
def asof_index_bars(symbol_bars: list[dict], index_bars: list[dict]) -> list[dict]:
    """For each symbol bar, attach the most recent index bar with time <= symbol bar time.
    Return a list parallel to symbol_bars; entry is the index bar dict or None if none <= time.
    Must be O(n+m) with a single merge pass (both inputs are sorted ascending by 'time')."""

def daily_before(daily_rows: list[dict], day_start_ts: int) -> list[dict]:
    """Return daily rows strictly before the IST date of day_start_ts (sorted)."""
```
Reuse `app.integrations.signals.engine._ist_date` for date math. **No look-ahead:** `asof` uses
`<=` (same-timestamp index bar is allowed; it closed at the same time), `daily_before` uses `<`.
**Accept:** unit test in Task 7.1.

**Task 2.3 — `ml_meta/scorer.py`: inference wrapper.** Implement exactly Appendix A `MetaScorer`
and `filtered_fires`. The scorer loads a LightGBM booster from the artifact's text dump
(`lightgbm.Booster(model_str=...)`), applies optional calibration (a monotonic lookup — see
Appendix A; do **not** pull in sklearn), and thresholds. `filtered_fires` runs the existing
`run_strategy`, then for each `Fire` calls `features.extract_features(...)` (Doc 2), scores, and keeps
fires where `decide()` is true. **Feature order** must follow `artifact["features"]["names"]`; build the
input vector in that exact order, substituting `float('nan')` for any missing key (LightGBM handles NaN).
**Accept:** with the stub features + a tiny hand-built booster (Task 7.2) it returns a subset of
`run_strategy`'s fires.

**Task 2.4 — `ml_meta/features.py` STUB (temporary).** So the above imports resolve before Doc 2:
```python
FEATURE_SPEC_VERSION = "0-stub"
FEATURE_NAMES: list[str] = ["stub_const"]
CATEGORICAL_FEATURES: list[str] = []
def extract_features(**kwargs) -> dict: return {"stub_const": 1.0}
```
Mark with `# STUB — replaced by Doc 2`. **Accept:** `from app.integrations.ml_meta import features` works.

---

## 3. ML repo: artifact v2 read/write

**Task 3.1 — Extend [`stock_terminal_ml/artifact.py`](../../stock_terminal_ml/artifact.py).**
Keep `build`/`write_atomic`/`utc_iso` for v1. Add:
```python
def build_v2(*, model_id, label, primary, model, features, training, evaluation,
             per_symbol, global_fallback) -> dict:
    """Return the Appendix B dict with version='2', kind='meta_labeling'."""
def write_atomic_v2(payload: dict, path: Path) -> Path:
    """validate via ml_meta.schema.validate_v2, then atomic .tmp -> os.replace."""
```
The LightGBM booster goes in as a **string** (`booster.model_to_string()`); you just place whatever
the trainer hands you into `payload["model"]["booster"]`. Do not train anything here.
**Accept:** round-trip test (Task 7.3): `write_atomic_v2` then `json.load` then `validate_v2` passes.

---

## 4. Backend: registry, backtest, and live endpoint

**Task 4.1 — Registry awareness.** In `../stock-terminal/backend/app/integrations/ml/registry.py`,
`list_models()` currently returns v1 descriptors. Add to each descriptor row:
`"version": data.get("version", "1")` and `"kind": data.get("kind", "params")`. Keep `tuned_keys`
working for v1 (guard: only compute it when `version == "1"`). For v2, add
`"features": (data.get("features") or {}).get("names")` and
`"evaluation": data.get("evaluation")` to the row. Do not break the v1 shape.
**Accept:** `/api/ml/models` returns both a v1 and a v2 row without error (Task 7.6 integration).

**Task 4.2 — Live scoring endpoint.** In `../stock-terminal/backend/app/integrations/ml/router.py`
add exactly Appendix C:
```
POST /api/ml/score
```
Load the model via `registry.get_model(model_id)`. If `schema.is_v2(model)` is false → HTTP 422
("model {id} is not a meta-labeling model"). Build a `MetaScorer.from_artifact(model)`. Resolve the
candles: if `body.bars` is provided use them; else load from the offline dataset
(`offline.store.load_symbol`) for `[from,to]`. Fetch the NIFTY 50 index bars for the same span
(offline: `load_symbol(dataset, "NIFTY 50")`; live: the caller passes `index_bars` or the backend
fetches via Kite — for v2 Phase-1 **offline path only is required**; live-Kite path may raise
`NotImplementedError` with a TODO). Build `cpr_by_day` with `build_cpr_by_day`, compute supertrend if
needed, call `filtered_fires`, and return the Appendix C response (markers with `p_profit`,
`baseline_count`, `kept_count`, `threshold`).
**Accept:** `curl` with a v2 model id + `{symbol, from, to, dataset}` returns markers; `kept_count <=
baseline_count`.

**Task 4.3 — Backtest routing for v2.** In
`../stock-terminal/backend/app/integrations/backtest/router.py`:
- `_build_batch_config` (around line 231): when `ml_registry.get_model(ml_id)` is a v2 model, set a new
  `BatchConfig` field `meta_scorer: MetaScorer | None` (add it to `BatchConfig` in
  `backtest/jobs.py` next to `ml_per_symbol`). Do **not** populate `ml_per_symbol` for v2.
- `_simulate_into` (around line 177): if `config.meta_scorer` is set, replace
  `fires = run_strategy(...)` with `fires = config.meta_scorer_filtered(bars, cpr_by_day, params, ...)`
  (thin wrapper calling `ml_meta.scorer.filtered_fires` with the index bars for this symbol+span).
  Everything after (`simulate`, `compute_metrics`) is unchanged. Set
  `result.params_source = "meta_labeling"`.
- `_params_for` stays for v1 only.
**Accept:** a backtest with a v2 `ml_model` runs and produces fewer trades than the same run with no
model (Task 7.6).

---

## 5. Frontend: v2 indicator (fetch server-scored markers)

Today `frontend/src/features/futures/devSignalMlDef.js` computes markers **client-side** from tuned
params. For v2 the markers come from the backend.

**Task 5.1 — `frontend/src/features/ml/api.js`:** add
```js
score({ modelId, symbol, from, to, dataset }) // POST /api/ml/score -> { markers, threshold, kept_count, baseline_count }
```

**Task 5.2 — New `frontend/src/features/futures/devSignalMlV2Def.js`:** export
`makeDevSignalMlV2Def(modelJson)` that returns an indicator def whose `compute(bars, _v, p, ctx)`:
- reads `ctx.name`/`ctx.symbol`, the visible time range from `bars[0].time`/`bars.at(-1).time`, and the
  dataset from ctx (fallback `'nifty50_3m_ytd'`);
- calls `mlApi.score(...)` (async; cache per (id,symbol,from,to) to avoid refetch on every redraw —
  follow the existing async pattern in `main.jsx`/`useMlTraining.js`);
- maps response markers to the same marker shape `devSignalMlDef.js` already uses (reuse its `marker`
  import if convenient), and sets a `legendNote` like `` `${label} · ML p≥${threshold} · ${kept}/${baseline}` ``.
Render zero markers + a legend note if the model has no features/booster.

**Task 5.3 — Registration switch.** In `frontend/src/main.jsx` `bootstrapMlIndicators()`: for each
model, choose the def by version — `model.version === '2' ? makeDevSignalMlV2Def(model) :
makeDevSignalMlDef(model)`. Keep the existing `ml-models-updated` event wiring.
**Accept:** with a v2 model present, the chart shows a "Dev Signal MLn" indicator whose markers match
the `/api/ml/score` response for the visible range.

**Task 5.4 — Backtest results label.** In `frontend/src/features/backtest/BacktestResults.jsx` (lines
~116): add a branch for `params_source === 'meta_labeling'` → `' · ML meta-labeling filter'`.

---

## 6. Trainer CLI / config / progress glue (ML repo)

Doc 2 writes the training *logic* in `stock_terminal_ml/meta/train_meta.py`. You wire the **entry point,
flags, config precedence, progress events, and dashboard fields** so it matches the existing v1 ergonomics.

**Task 6.1 — New subcommand / module `stock_terminal_ml/train_meta.py`** with a `main(argv)` mirroring
[`train.py`](../../stock_terminal_ml/train.py)'s argparse + config-precedence pattern. New/changed flags:
`--mode {params,meta}` (default `params` keeps v1 behavior; `meta` runs the new pipeline),
`--train-months`, `--test-months`, `--embargo-days`, `--model-family lightgbm`, `--calibrate/--no-calibrate`,
`--threshold-objective` (reuse names from [`objectives.py`](../../stock_terminal_ml/objectives.py)).
When `--mode meta`, delegate to `meta.train_meta.run(cfg)` (Doc 2). Keep `--config` JSON precedence
identical to v1 (see CLAUDE.md "Config precedence quirk").
**Accept:** `--mode meta --help` lists the new flags; `--mode params` reproduces old behavior.

**Task 6.2 — Progress events.** Reuse [`progress.py`](../../stock_terminal_ml/progress.py). Keep the v1
event names (`run_start`, `trial`, `symbol_done`, `run_done`, `error`) so both consumers keep working.
Add **two optional** events for the meta pipeline, documented in Appendix D: `fold` (one per
walk-forward fold: `{event:"fold", index, train_from, train_to, test_from, test_to, metrics:{...}}`) and
`stage` (`{event:"stage", name:"features|labels|train|calibrate|evaluate", pct}`). Consumers must treat
unknown events as ignorable — verify the backend runner `_aggregate` (in `ml/runner.py`) does not crash
on unknown event names (it switches on known names and ignores others — confirm, don't change).
**Accept:** running `--mode meta` emits `run_start` … `run_done`; `GET /api/ml/train/{id}` still rolls up.

**Task 6.3 — Dashboard fields.** In [`dashboard/index.html`](../../dashboard/index.html) +
[`dashboard.py`](../../stock_terminal_ml/dashboard.py): add a "Mode" selector (params | meta) to the run
form; when meta, show the new flags (train/test months, embargo). In the results view, when a model is v2,
show `evaluation.oos` (baseline vs ML: net PnL, Sharpe, #trades, precision) instead of the per-symbol
param table. The dashboard still just writes `config.json` + `run.sh` and launches the CLI — no in-process
training.
**Accept:** selecting "meta" in the form writes a `config.json` with `mode:"meta"` and the new keys.

**Task 6.4 — Update [`ml-pipeline.md`](../../../stock-terminal/docs/backend/ml-pipeline.md)** backend doc
with the v2 artifact shape (Appendix B), the `/api/ml/score` route (Appendix C), and the new events
(Appendix D). Mechanical: copy the appendices.

---

## 7. Tests (ML repo `tests/`, backend `tests/` if present)

Keep the existing smoke test working. Add:

- **7.1** `test_data_align.py`: `asof_index_bars` attaches correct bars, never a future one; empty index → all None.
- **7.2** `test_scorer.py`: build a 2-leaf LightGBM booster by hand (or train a 10-sample toy), wrap in an
  Appendix B artifact, assert `MetaScorer.score_fire` returns a float in [0,1] and `filtered_fires` ⊆
  `run_strategy` fires.
- **7.3** `test_artifact_v2.py`: `build_v2` → `write_atomic_v2` → reload → `validate_v2` passes; tampering
  removes a required key → raises.
- **7.4** `test_endpoint_score.py` (backend): POST `/api/ml/score` with a toy v2 model on 1 symbol over a
  small offline range → `kept_count <= baseline_count`, markers carry `p_profit`.
- **7.5** `test_v1_still_works.py`: a v1 artifact still lists in `/api/ml/models` and still drives a backtest.
- **7.6** extend the existing smoke test with `-rs` awareness (it silently skips with no dataset — keep that).

**Accept:** `pytest -q` green (or skipped-with-reason where no dataset), no import errors.

---

## 8. Order of work & hand-off points

1. Tasks 1, 2 (with stub features), 3 — unblock everything; tests 7.1–7.3 green.
2. Tasks 4, 5 — backend routing + frontend; tests 7.4–7.6 green (using the toy model).
3. Task 6 — CLI/dashboard glue.
4. **Hand-off:** Doc 2 replaces `features.py` (Task 2.4 stub) and delivers `meta/{labels,cv,train_meta,evaluate}.py`.
   The interfaces in Appendix A are frozen — if Doc 2 needs a new field, it is added to the artifact
   `training`/`evaluation` blob (free-form dicts), never to the frozen signatures without updating both docs.

Everything in this doc is verifiable against a spec or a test. If a task seems to require an ML/stats
decision, **stop** — it belongs in Doc 2.

---

## Appendix A — Frozen interfaces (identical in both docs)

```python
# app/integrations/ml_meta/features.py  — BODY owned by Doc 2; SIGNATURE frozen here.
FEATURE_SPEC_VERSION: str
FEATURE_NAMES: list[str]            # ordered, stable; the model's input contract
CATEGORICAL_FEATURES: list[str]     # subset of FEATURE_NAMES fed to LightGBM as categorical

def extract_features(*,
    bars: list[dict], i: int, fire,                 # fire: app.integrations.signals.pattern.Fire
    cpr: dict[str, float],
    st_line, st_trend,                              # supertrend arrays aligned to bars, or None
    index_bar: dict | None,                         # asof index bar for bars[i] (from data.asof_index_bars)
    index_hist: list[dict],                         # index bars up to and including bars[i]'s time
    index_cpr: dict[str, float] | None,
    daily_before: list[dict],                       # symbol daily rows strictly before today
    symbol: str,
) -> dict[str, float]:
    """Return {feature_name: value} using ONLY data at or before bars[i]. Missing -> omit (scorer fills NaN)."""

# app/integrations/ml_meta/scorer.py  — owned by Doc 1.
class MetaScorer:
    @classmethod
    def from_artifact(cls, artifact: dict) -> "MetaScorer": ...
    def score_fire(self, feats: dict[str, float]) -> float:   # calibrated P(profit) in [0,1]
        ...
    def decide(self, p: float, symbol: str) -> bool:          # threshold; per-symbol override else global
        ...

def filtered_fires(*, bars, cpr_by_day, params, scorer: MetaScorer,
                   index_bars: list[dict], daily: list[dict], symbol: str) -> list:  # -> list[Fire]
    """run_strategy(bars,cpr_by_day,params) then keep fires where scorer.decide(score_fire(features))."""
```

Calibration storage (no sklearn at serve time): store as two equal-length float arrays
`calibration.x` (raw scores, ascending) and `calibration.y` (calibrated probs); `score_fire` does a
`numpy.interp(raw, x, y)`. If `calibration` is absent, use the raw booster probability.

## Appendix B — Artifact schema v2

```json
{
  "version": "2",
  "kind": "meta_labeling",
  "id": "dev_signal_ml5",
  "label": "Dev Signal ML5",
  "created_at": "2026-...+05:30",
  "primary": { "params": { "...Dev Signal Params used to generate candidates...": 0 } },
  "model": {
    "framework": "lightgbm",
    "objective": "binary",
    "booster": "<lightgbm Booster.model_to_string() text>",
    "decision_threshold": 0.57,
    "calibration": { "x": [0.0, 0.5, 1.0], "y": [0.0, 0.48, 1.0] }
  },
  "features": { "spec_version": "1", "names": ["ret_5", "atr_pct", "..."], "categorical": ["fire_kind", "line_name"] },
  "training": {
    "dataset": "nifty50_3m_ytd",
    "window": { "from": "2026-01-01", "to": "2026-10-31" },
    "train_months": 3, "test_months": 1, "embargo_days": 1,
    "model_family": "lightgbm",
    "hyperparams": { "num_leaves": 63, "learning_rate": 0.03, "...": 0 },
    "symbols_total": 51, "samples_total": 0,
    "finished_at": "2026-...Z"
  },
  "evaluation": {
    "oos": { "folds": [ { "test_from": "", "test_to": "", "pr_auc": 0, "precision": 0, "recall": 0,
                          "ml": {"net_pnl_pct":0,"sharpe":0,"num_trades":0,"max_drawdown_pct":0},
                          "baseline": {"net_pnl_pct":0,"sharpe":0,"num_trades":0,"max_drawdown_pct":0} } ],
             "aggregate": { "pr_auc_mean": 0, "ml_vs_baseline_net_pnl_pct": 0, "deflated_sharpe": 0 } },
    "feature_importance": { "ret_5": 0.0 }
  },
  "per_symbol": { "RELIANCE": { "threshold_override": 0.6, "oos": { "...": 0 } } },
  "global_fallback": { "decision_threshold": 0.57 }
}
```
`validate_v2` required keys: `version=="2"`, `kind=="meta_labeling"`, `id`, `label`, `created_at`,
`model.framework`, `model.booster` (non-empty str), `model.decision_threshold` (float),
`features.names` (non-empty list), `training`, `evaluation`, `global_fallback.decision_threshold`.
`primary.params`, `per_symbol`, `calibration`, `feature_importance` are optional.

## Appendix C — `POST /api/ml/score`

Request:
```json
{ "model_id": "dev_signal_ml5", "symbol": "RELIANCE", "timeframe": "3m",
  "from": "2026-10-01", "to": "2026-10-31", "dataset": "nifty50_3m_ytd",
  "bars": null, "index_bars": null }
```
`bars`/`index_bars` optional; when null, load from `dataset` for `[from,to]` (+ warmup/daily as the
trainer does). Response:
```json
{ "model_id": "dev_signal_ml5", "symbol": "RELIANCE", "threshold": 0.57,
  "baseline_count": 42, "kept_count": 18,
  "markers": [ { "time": 1730000000, "price": 1575.4, "direction": "BUY",
                 "kind": "confirmation", "line_name": "pp", "p_profit": 0.63 } ] }
```
Validation: `model_id` matches `[A-Za-z0-9_-]{1,64}`; 404 if unknown; 422 if not v2.

## Appendix D — New progress events (additive; v1 events unchanged)

```
{"event":"stage","name":"features","pct":0.2}
{"event":"fold","index":0,"train_from":"2026-01-01","train_to":"2026-03-31",
 "test_from":"2026-05-01","test_to":"2026-05-31","metrics":{"pr_auc":0.61,"ml_net_pnl_pct":2.3,"baseline_net_pnl_pct":1.1}}
```
Consumers ignore unknown events. `run_start`/`run_done`/`error` keep their v1 fields; `run_done` still
carries `{"artifact": path}`.

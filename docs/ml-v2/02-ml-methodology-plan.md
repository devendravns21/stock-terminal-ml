# Dev Signal ML v2 — ML Methodology Plan (Doc 2 of 2)

> **Audience:** a high-reasoning coding model (and the user as reviewer). This doc owns every
> decision where correctness requires statistical judgment: feature design & leakage, label
> construction, cross-validation, model training/tuning, calibration, decision thresholds, and
> the acceptance gates that decide whether a model ships. The mechanical plumbing (artifact I/O,
> scorer wrapper, endpoints, UI, CLI glue) is specified in Doc 1
> ([`01-mechanical-plan.md`](01-mechanical-plan.md)) and consumes the **frozen interfaces** in
> Appendix A (shared, identical in both docs). You write the *bodies* of `features.py`,
> `labels.py`, `cv.py`, `train_meta.py`, `evaluate.py`; you must not change the frozen signatures
> without updating Doc 1.

## 0. Thesis & why this design

The problem is not "predict the market." It is: **the Dev Signal rule fires too often and too many
fires lose money.** We keep the rule as the *primary* model (it sets direction and entry timing) and
train a *secondary* model to answer one narrow, learnable question: **given this fire and its context,
what is the probability the resulting trade is profitable under the configured target/stop?** This is
**meta-labeling** (López de Prado, *Advances in Financial Machine Learning*, ch. 3). It is the right
tool because:
- The label is well-defined and already computable from the existing simulator (triple-barrier outcome).
- It is a **binary classification on tabular features** → gradient-boosted trees (LightGBM) are the
  pragmatic state of the art; small data, non-linear, robust to unscaled/irregular features, NaN-native,
  fast, interpretable via SHAP, and exportable to a portable text model for serving.
- It *enhances* the indicator instead of replacing it, so the chart stays interpretable and the baseline
  is always available as a fallback and as the comparison benchmark.

**Primary-model tuning for recall.** Meta-labeling works when the primary model is run at **high recall**
(many candidates, tolerant thresholds) and the secondary model supplies **precision**. So generate
candidates with permissive Dev Signal params (default, or loosen `touch/break_tolerance`, `min_move`,
`cooldown`) — do **not** pre-filter hard. Record the exact `primary.params` in the artifact. The ML's job
is to cut the false positives.

---

## 1. Sample & label construction (`stock_terminal_ml/meta/labels.py`)

**Unit of observation.** One candidate fire = one sample. For a `(symbol, window)` run:
1. Build `cpr_by_day` and supertrend as the backtest does.
2. `fires = run_strategy(bars, cpr_by_day, primary_params)` — reuse the backend verbatim.
3. Each `Fire` has `bar_index=i`. Features come from data **at or before** `bars[i]` (Section 2). The
   label comes from the trade's **future** outcome (that is the target, not a feature — allowed).

**Label = per-fire triple-barrier outcome.** The existing `simulate()` manages one position at a time and
skips overlapping fires, which is wrong for labeling (we need every candidate labeled). Implement a
**single-fire evaluator** that reuses the simulator's exit math exactly (same intrabar target/stop,
stop-wins-ties, session square-off — see `backtest/simulator.py` lines 42–104) but evaluates each fire
*independently*:
```python
def label_fire(bars, i, fire, exits: ExitRules, costs: CostConfig) -> dict:
    # enter at bars[i].close (long if BUY else short); walk forward applying target/stop/square-off;
    # return {"net_pct": float, "exit_reason": str, "hold_bars": int, "y": int}
```
- `y = 1` if `net_pct > tau_cost` else `0`, where `tau_cost` is a small positive buffer covering expected
  costs/slippage beyond what `trade_costs` already books (default `tau_cost = 0.0`; make it a config knob).
- Keep the continuous `net_pct` and `hold_bars` on the sample — needed for sample weights and an optional
  future sizing head.
- **Skip** fires on the last bar of a session (no room to manage) exactly as `simulate()` does — those are
  not valid samples.

**Sample weights** (train-time; López de Prado ch. 4 spirit, simplified for intraday non-overlapping
trades):
- **Return-magnitude weight:** `w_ret = |net_pct|` normalized to mean 1 per fold — larger PnL swings
  matter more than marginal ones.
- **Time-decay weight:** `w_time = linear or exp decay`, newest sample weight 1.0, oldest `~0.5` — recent
  regime matters more. Config: `--time-decay`.
- Final weight `w = w_ret * w_time`, renormalized. Pass as LightGBM `sample_weight`.
- Intraday trades rarely overlap across days, so full uniqueness weighting is optional; note it and skip
  in Phase 1 unless folds show heavy same-day overlap.

**Class balance.** Expect imbalance (most fires lose). Do **not** oversample/SMOTE (distorts calibration).
Use LightGBM `scale_pos_weight = n_neg/n_pos` **computed per training fold** (never on the whole dataset —
that leaks test-fold base rates). Keep raw base rate in `training` for reference.

---

## 2. Feature engineering (`app/integrations/ml_meta/features.py`)

**Hard rule — no look-ahead.** Every feature uses only `bars[:i+1]`, `index_hist` (index bars with
`time <= bars[i].time`), and `daily_before` (strictly before today). This is the single most important
correctness property in the whole system; a leak here invalidates every metric. Review each feature
against it and add an assertion test (Section 6) that shifting future bars never changes a feature at `i`.

Implement `FEATURE_NAMES` as the ordered contract and compute all of them in `extract_features`. Groups
(≈35–45 features — "as many as reproducibly possible"; all derivable from the offline dataset):

**A. Fire descriptors (categorical / structural)**
- `fire_direction` (BUY=1/SELL=0), `fire_kind` (reversal/confirmation — categorical),
  `line_name` (pp/cr/cs/r1..s3/st — categorical), `line_is_support`/`line_is_resistance`.
- `dist_to_line_pct = (close_i - L)/L * 100`; `ext_at_fire_pct`, `retest_depth_pct` (for confirmation; NaN
  for reversal — fine).

**B. Trend / momentum (symbol, intraday)**
- `ret_k` for k∈{1,2,5,10,20} bars: `close_i/close_{i-k} - 1`.
- `st_trend` (+1/-1), `dist_to_st_pct`, `bars_since_st_flip`.
- `ema_fast/ema_slow` slope signs; `close_vs_vwap_pct` (session VWAP up to i — cumulative Σ(price·vol)/Σvol).

**C. Volatility / range**
- `atr14_pct = ATR(14)/close_i * 100` (reuse the backend's Wilder `_atr` from `signals/indicators.py`).
- `realized_vol_20` (stdev of 1-bar returns, last 20), `range_expansion = TR_i / ATR14`.
- `day_range_pos = (close_i - day_low_so_far)/(day_high_so_far - day_low_so_far)`.

**D. Volume / liquidity (the user's explicit ask)**
- `rvol = volume_i / mean(volume last N)`; `rvol_tod = volume_i / avg volume at this minute-of-day`
  (intraday volume profile over the training window — compute the profile from **past days only**).
- `signed_vol_k = Σ sign(ret)·volume` over last k; `obv_slope`; `dollar_vol = close_i·volume_i` (log).

**E. Index / market context (NIFTY 50 — the user's explicit ask; already in the dataset)**
- `idx_ret_k` for k∈{1,5,20} from `index_hist`.
- `rel_strength_k = ret_k(symbol) - idx_ret_k` (does the stock lead or lag the market?).
- `beta_60` rolling beta of symbol to index (last 60 bars); `corr_60`.
- `idx_dist_to_pp_pct` (is the whole market sitting on a pivot?); `idx_st_trend`.
- `aligned_with_index` = sign(fire direction) == sign(idx_ret_5).

**F. Time / calendar**
- `mins_since_open`, `session_bucket` (open/mid/close — categorical), `day_of_week`.
- `is_first_hour`, `is_last_hour` (square-off risk).

**G. Prior-day / gap (from `daily_before`)**
- `prev_day_ret`, `prev_day_range_pct`, `gap_open_pct = (today_open - prev_close)/prev_close`,
  `close_vs_prev_close_pct`.

**Cross-sectional peer features — DEFER to Phase 1b.** Features like "momentum rank of this symbol among
all 50 constituents at time i" are powerful but require aligning all 50 symbols at serve time, which the
single-symbol live endpoint does not load. Keep the **serving feature set computable from {symbol bars +
index bars + daily}** only. If you add peer-rank features, gate them behind a `features.spec_version` and a
flag, compute them in a separate batch pass for training/backtest, and do **not** include them in a model
meant for the live endpoint until Doc 1's endpoint is extended to fetch peers. Document this boundary.

**Missing values:** return the key omitted (scorer fills NaN); LightGBM splits on NaN natively. Never
impute with 0 for ratios (0 is meaningful). Keep `FEATURE_SPEC_VERSION="1"`.

---

## 3. Cross-validation — purged, embargoed walk-forward (`stock_terminal_ml/meta/cv.py`)

This operationalizes the user's "6-month range → ~half train, one month test, then feedback loop." It
replaces the v1 `kfold`/`holdout` for the meta pipeline (keep the v1 module for v1 runs).

**Walk-forward with retraining ("feedback loop").** Split the window by IST **trading day** into months
(≈21 trading days). With `train_months=T` and `test_months=V` (defaults T=3, V=1):
```
fold 0: train = months [0 .. T),        test = month [T .. T+V)
fold 1: train = months [0 .. T+V) OR sliding [V .. T+V),  test = month [T+V .. T+2V)
...roll by V until data is exhausted.
```
- **Expanding** train (default, anchored) vs **sliding** (fixed T-month window) — expose
  `--cv-mode {expanding,sliding}`; default expanding (more data) but report both if time permits. This
  retrain-and-roll *is* the feedback loop.
- **Embargo (`embargo_days`, default 1):** drop the first `embargo_days` trading days of each test month
  from… no — embargo removes **training** samples whose label horizon could overlap the test set, plus a
  buffer. Because trades are **intraday** (every label resolves by session close the same day), a leak can
  only happen across the train/test seam. So: remove from **training** any fire on the last `embargo_days`
  trading day(s) before the test start, and never let a sample's label window cross into the test region.
  This is the purge+embargo, made trivial by intraday labels — state this reasoning in a comment.
- On the full ~10-month dataset (Jan–Oct) with T=3,V=1 expanding you get ≈6 OOS months. Report each.

**CPCV for a distribution (recommended, `--cv cpcv`).** A single walk-forward path is one OOS estimate;
its variance is unmeasured. Add Combinatorial Purged CV (López de Prado ch. 7 & 12): split into N day-blocks
(e.g. N=8), train on all-but-k, test on k combinations, purge+embargo each, producing many OOS paths →
report mean and dispersion of the economic metric, and a **Deflated Sharpe Ratio** (ch. 8) to correct for
the selection bias of having tried many configs. Use CPCV for the final acceptance decision; use plain
walk-forward for fast iteration.

**Leakage checklist (enforce in code + tests):**
1. `scale_pos_weight`, feature normalizations (rvol profile, beta windows), and the decision threshold are
   all fit on **training folds only**.
2. The index asof-join uses `<=`, daily uses `<`.
3. Calibration is fit on a held-out slice *inside* the training region (nested), never on test.
4. Hyperparameters are selected by **nested** CV (Section 4), not on the OOS test months.

---

## 4. Model training & tuning (`stock_terminal_ml/meta/train_meta.py`)

**Pooled cross-sectional model (default).** Train **one** LightGBM model across all symbols, not 51 tiny
ones. ~51 symbols × ~hundreds of fires ≫ per-symbol data; pooling generalizes and is the only way the
market/volume/index features pay off. Do **not** one-hot 51 symbols; rely on the symbol-descriptive
features (volatility, liquidity, beta). Optionally add `symbol` as a LightGBM categorical **only** if CV
shows it helps without overfitting. Expose `--per-symbol` to fall back to per-symbol models for comparison,
but pooled is the recommendation.

**Estimator:** `lightgbm.LGBMClassifier` / `Booster`, `objective="binary"`, `metric="binary_logloss"`.
Use `early_stopping` on an inner validation slice. Pass `categorical_feature=CATEGORICAL_FEATURES`.

**Hyperparameter search (nested, reuse Optuna — already a dependency).** Outer loop = walk-forward folds
(OOS). Inner loop = purged CV *within* each training region to pick hyperparameters; never touch the OOS
test month during tuning. Search space (log where sensible):
`num_leaves 15–255`, `learning_rate 0.005–0.1`, `min_child_samples 10–200`, `feature_fraction 0.5–1.0`,
`bagging_fraction 0.5–1.0`, `bagging_freq 0–5`, `lambda_l1 0–5`, `lambda_l2 0–5`, `n_estimators` via early
stopping. Objective for Optuna = the **economic** score (Section 5), not logloss, evaluated on inner
validation; break ties by logloss. Seed everything (seed 42) for reproducibility.

**Calibration.** Tree probabilities are not well-calibrated; sizing/thresholds need real probabilities. Fit
**isotonic regression** (or Platt if folds are tiny) on an inner held-out slice, store it as the
`calibration.x/y` arrays in the artifact (Appendix A — no sklearn needed at serve time). Verify calibration
with a reliability curve in `evaluation`.

**Final fit.** After acceptance, refit on the **entire** window with the chosen hyperparameters (and
refit calibration on a tail slice) to produce the shipped booster. The OOS metrics in the artifact come
from the walk-forward/CPCV evaluation, **not** from this final all-data fit.

---

## 5. Decision threshold & the economic objective (`evaluate.py`)

**Do not optimize F1.** Optimize the thing the user cares about: money. For each fold, sweep the
probability threshold `p* ∈ [0.3, 0.9]` and, for each, re-run the trade simulation keeping only fires with
`p ≥ p*`, then score with the existing objective machinery
([`objectives.py`](../../stock_terminal_ml/objectives.py)) — default `expected_pnl_with_constraints`
(per-trade PnL discounted by trade-count scarcity and drawdown). Pick the `p*` that maximizes the **mean
OOS** economic score across folds (not per-fold cherry-picking). Store it as `model.decision_threshold`;
allow `per_symbol.threshold_override` when a symbol's OOS strongly justifies it (guard against overfitting
— require a minimum sample count).

**The comparison that decides everything.** For every OOS fold, compute metrics for **two** strategies on
the identical bars/exits/costs:
- **baseline:** raw `run_strategy` fires (no ML filter).
- **ml:** fires kept by the meta-labeler at `p*`.
Report net PnL %, Sharpe, #trades, win rate, max drawdown for both, per fold and aggregate.

---

## 6. Acceptance gates (ship / no-ship) & tests

A v2 model is **shippable** only if, on OOS (walk-forward or CPCV):
1. **Beats baseline on risk-adjusted return:** `ml.sharpe > baseline.sharpe` in a majority of folds **and**
   aggregate `ml.net_pnl_pct ≥ baseline.net_pnl_pct` while `ml.max_drawdown_pct ≤ baseline.max_drawdown_pct`.
2. **Precision lift is real:** OOS PR-AUC > base rate, and ML win rate > baseline win rate.
3. **Trades are cut, not zeroed:** `0.2 ≤ kept/baseline ≤ 0.9` aggregate (a filter that keeps everything or
   almost nothing is a failure, not a model).
4. **Not overfit:** CPCV Deflated Sharpe > 0; the spread of OOS economic score across CPCV paths is reported
   and not dominated by one lucky path; train-vs-OOS economic gap is within a stated tolerance.
If a gate fails, the artifact is still written (for inspection) but `evaluation.aggregate.shippable=false`
and the dashboard/README flags it. Never let a model that fails gate 1 become the chart default.

**Tests Doc 2 owns** (in addition to Doc 1's plumbing tests):
- `test_no_lookahead.py`: for a fixed `i`, appending arbitrary future bars to `bars` does not change any
  value in `extract_features(...)`. This is the critical guard.
- `test_label_matches_simulator.py`: a window with exactly one fire labeled by `label_fire` reproduces the
  `net_pct`/`exit_reason` that `simulate()` gives for that single trade.
- `test_cv_purge.py`: no training day in any fold is `> train_to - embargo` relative to its `test_from`;
  train and test day-sets are disjoint.
- `test_threshold_monotone.py`: raising `p*` is monotone non-increasing in `kept_count`.

---

## 7. Risks & failure modes (own these explicitly)

- **Regime shift / tiny data.** ~10 months, one market regime. The model may learn 2026's regime, not a law.
  Mitigation: CPCV + Deflated Sharpe, time-decay weights, expanding window, and honest reporting of OOS
  dispersion. Set expectations: the win is "fewer, better trades," not a magic Sharpe.
- **Label noise from the exit model.** Labels depend on the chosen `target_pct`/`sl_pct`. A model is only
  valid for the exit rule it was trained on — record `exits` in `training` and refuse to serve a model
  against mismatched exits (warn at minimum).
- **Serving/training skew.** The single guarantee against it is that `features.py` lives in the backend and
  is imported by the trainer. Never reimplement feature math in the trainer or the frontend. The frontend
  shows server-scored markers (Doc 1 §5) precisely to avoid a third implementation.
- **Threshold overfitting.** Choosing `p*` on the same data you report is leakage; choose on inner/CV only.
- **Base-rate leakage** via `scale_pos_weight`/normalization fit on all data — enforced by Section 3.

---

## 8. Phased roadmap

- **Phase 1 (this plan):** meta-labeling LightGBM on offline-derivable features (A–G), pooled, purged
  walk-forward + CPCV, server-side scoring. Ship when the gates pass.
- **Phase 1b:** cross-sectional peer-rank features (needs the endpoint to load peers); SHAP-based feature
  pruning; per-symbol threshold overrides where justified.
- **Phase 2 (external data — deferred by the user's choice):** India VIX, FII/DII, US overnight
  (S&P/Nasdaq prior close), GIFT Nifty, news sentiment. **Blocking concern:** historical point-in-time
  integrity. Free sources (nsepy/Kaggle VIX, nsefin FII/DII, yfinance US) rarely provide as-reported
  vintages, NSE endpoints drift, and GIFT Nifty redistribution is restricted/paid. Rule: a feature enters
  Phase 2 only with a point-in-time source (self-snapshotted daily going forward, or a vendor archive);
  until then it is live-only and excluded from historical OOS to avoid look-ahead. Treat Phase 2 as a
  research spike with its own validation, not a drop-in.
- **Phase 3 (optional, the "Google models" idea):** add a **TimesFM 2.5** short-horizon forecast of the
  symbol/index as an *extra feature* feeding the same LightGBM meta-labeler (not as the core model) — only
  if Phase 1 SHAP shows the trend features are saturating. Keep it a feature, not a second serving runtime.

---

## Appendix A — Frozen interfaces (identical in both docs)

```python
# app/integrations/ml_meta/features.py  — BODY owned by THIS doc; SIGNATURE frozen.
FEATURE_SPEC_VERSION: str
FEATURE_NAMES: list[str]            # ordered, stable; the model's input contract
CATEGORICAL_FEATURES: list[str]     # subset of FEATURE_NAMES fed to LightGBM as categorical

def extract_features(*,
    bars: list[dict], i: int, fire,                 # fire: app.integrations.signals.pattern.Fire
    cpr: dict[str, float],
    st_line, st_trend,                              # supertrend arrays aligned to bars, or None
    index_bar: dict | None,                         # asof index bar for bars[i]
    index_hist: list[dict],                         # index bars up to and including bars[i]'s time
    index_cpr: dict[str, float] | None,
    daily_before: list[dict],                       # symbol daily rows strictly before today
    symbol: str,
) -> dict[str, float]:
    """Return {feature_name: value} using ONLY data at or before bars[i]."""

# app/integrations/ml_meta/scorer.py  — BODY owned by Doc 1.
class MetaScorer:
    @classmethod
    def from_artifact(cls, artifact: dict) -> "MetaScorer": ...
    def score_fire(self, feats: dict[str, float]) -> float: ...   # calibrated P(profit) in [0,1]
    def decide(self, p: float, symbol: str) -> bool: ...          # per-symbol override else global threshold

def filtered_fires(*, bars, cpr_by_day, params, scorer: MetaScorer,
                   index_bars: list[dict], daily: list[dict], symbol: str) -> list:  # -> list[Fire]
    ...
```
Artifact schema v2, `POST /api/ml/score`, and progress events are specified in Doc 1 Appendices B–D; the
`training` and `evaluation` blobs are free-form dicts this doc populates (Section 4–6).

## Appendix B — References

- Meta-labeling (primary + secondary binary classifier to filter false positives):
  [Wikipedia](https://en.wikipedia.org/wiki/Meta-Labeling),
  [Hudson & Thames toy example](https://hudsonthames.org/meta-labeling-a-toy-example/),
  [triple-barrier labeling](https://hudsonthames.org/?p=3502).
- Purged / embargoed CV and CPCV + Deflated Sharpe:
  [Purged cross-validation (Wikipedia)](https://en.wikipedia.org/wiki/Purged_cross-validation),
  [`purgedcv` (PyPI)](https://pypi.org/project/purgedcv/0.0.3/),
  [temporalcv financial CV](https://temporalcv.readthedocs.io/en/latest/api/cv_financial.html).
- Gradient-boosted trees for tabular finance (pick via walk-forward, metric PR-AUC/logloss; libraries land
  within a few points — benchmark LightGBM vs XGBoost vs CatBoost):
  [GBDT+LSTM investment prediction (arXiv 2505.23084)](https://arxiv.org/abs/2505.23084).
- Google time-series foundation model (Phase 3 feature idea, not the core model):
  [TimesFM (google-research)](https://github.com/google-research/timesfm).

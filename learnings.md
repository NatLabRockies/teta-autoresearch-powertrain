# RouteE autoresearch — accumulated learnings

## Cross-cutting insights

Pipeline/optimization/data-representation truths that apply across powertrains.

### What works

- **LGBM >> RF, HGBR, XGBoost on this tabular regression.** Directly observed on BEV; HGBR is close but LGBM with `num_leaves=255` wins.
- **Seed-averaged LGBM ensemble is a clean variance-reduction win** (~-0.3% RMSE from a tuned single model). Start with 3 seeds; 5–10 gives diminishing returns. Per-member `random_state` is enough to get decorrelation if combined with subsampling.
- **In-ensemble bagging + column subsampling flips the single-model antipattern.** `bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8` hurt single-model LGBM (+0.17%) but help ensemble members by increasing decorrelation, together worth ~-0.15% beyond bare seed averaging.
- **Previous-link features are the dominant signal beyond raw speed/grade/miles.** Adding `prev_speed_mph` alone was a -31% breakthrough on BEV; `prev2/3/4_speed`, `prev_miles`, `prev2_miles`, `prev_speed_delta`, `prev_grade_percent`, `prev2_grade_percent` each contribute marginal-to-moderate gains.
- **Time/dwell features are informative on top of miles+speed.** `time_seconds` and `prev_time_seconds` each delivered ~-0.23% on BEV even though miles/speed express the same quantity — dwell helps LGBM partition stop-and-go vs cruise regions directly.
- **Geometry-derived features help.** `abs_bearing_delta` (cornering from WKB) was -3% on BEV; `sinuosity` and `prev_abs_bearing_delta` added marginal follow-ups.
- **Ensemble shifts HP optima vs single model.** `n_estimators=1500` (vs 1000 single), `min_child_samples=100` (vs 50 single), `learning_rate=0.04` (vs 0.05 single) all win in the ensemble because averaging absorbs per-member over-fit; use the ensemble setting, not the single-model setting, as the HP baseline once you're averaging. Explicit `reg_lambda`/`reg_alpha` are not material on top of these.
- **Delta features ARE load-bearing (do not drop).** `grade_delta`, `prev_speed_delta` each worth +0.2–0.34% of RMSE when removed. The sign-crossing structure of deltas is what LGBM can't recover axis-aligned from base features.

### What doesn't work

- **Deeper prev-speed lookback saturates at ~prev4.** `prev5_speed_mph` was tied.
- **Explicit product-interaction features are redundant (and slightly harmful).** `speed_times_grade`, `grade_times_miles`, `speed_accel`, `speed_sq_delta`, `prev_speed*prev_grade`, `sinuous_miles`: all tied or worse when added. Dropping `speed_times_grade` and `grade_times_miles` from a seed config each gave a ~-0.015% simplification win. Axis-aligned trees recover monotone products on their own.
- **Data filtering is banned by the domain and is a methodological antipattern here** — the model must predict the heavy regen/traction tails.
- **`extra_trees=True`** (ET-style random split thresholds) is too coarse at `n_estimators=1000+` with ~20 features; +0.88% on BEV ensemble.
- **Bagging without ensembling hurts LGBM.** Single-model LGBM with `bagging_fraction=0.8` is +0.17% worse; only useful when averaging multiple bagged members.
- **Cross-family ensembling with XGBoost didn't pay off** on this feature set: adding one XGB member to the 10-LGBM ensemble was exactly tied at +14 lines. XGB residuals are too correlated with LGBM's here.
- **Heterogeneous num_leaves across ensemble members was tied** (tried 127/255/511 cycle) — diversification ceiling is set by bagging+feature_fraction+seed.
- **Runtime budget is real.** `num_leaves=511` with 10-member ensemble ran 752s, over the 600s per-run cap. `n_estimators=2000` is also near the ceiling.

### Evaluation notes

- `fixed_utils.train_test_split` is row-level (80/20 mask at `random_seed=42`) and is the ground truth — do not change.
- `fixed_utils.rmse` on `energy_rate_gge` is the only score. Huber loss / target transforms don't help because we're scored on squared error.
- Runs are deterministic once seeds and data sort order are fixed.

## BEV (2017 Chevy Bolt)

Battery electric. Link energy is **signed** (regen when negative); asymmetric heavy-tailed target.

### What works

- **Current best feature set (19):** speed_mph, grade_percent, miles, prev_speed_mph, prev_grade_percent, speed_delta, grade_delta, prev_miles, prev2_speed_mph, prev3_speed_mph, prev4_speed_mph, prev_speed_delta, prev2_grade_percent, prev2_miles, time_seconds, prev_time_seconds, abs_bearing_delta, prev_abs_bearing_delta, sinuosity.
- **Current best model:** 10-seed LGBM ensemble, per-member `num_leaves=255, n_estimators=1500, min_child_samples=100, learning_rate=0.04, bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8`. Predictions averaged across seeds.

### What doesn't work

- Features to NOT add: `prev5_speed_mph`, `prev2_time_seconds`, `speed_accel`, `speed_sq_delta`, `sinuous_miles`, `prev_speed*prev_grade` — all tied (axis-aligned trees already recover them).
- Features to NOT drop: `grade_delta` (+0.21% if removed), `prev_speed_delta` (+0.34%) — sign-crossing deltas are irreplaceable.
- HP antipatterns: single-model bagging (+0.17%), `extra_trees=True` (+0.88%), `num_leaves=127` underfits (+0.24%), `num_leaves=511` overfits + over-budget, `min_child_samples=30` slight overfit, `bagging_freq=5` tied, `bagging_fraction=0.7` +0.05%, `feature_fraction=0.7` tied.
- Cross-family XGB member in ensemble: tied (0.0%, +14 lines). Heterogeneous num_leaves across members: tied.

### Best known config

- **RMSE:** 0.006562 (bev-apr24b/exp30, commit 7975cb0, 10-seed ensemble, 19 features).
- **Prior single-model best:** 0.006644 (bev-apr24/exp38, commit a076168).
- Session `bev-apr24b` total: **-1.23%** (0.006644 → 0.006562) over 33 experiments.

### Open hypotheses

- Target transforms for signed heavy tail (e.g. `log1p(|y|)*sign(y)` with inverse at predict) — unclear net gain at RMSE loss, complex to wire.
- Sequence models (GRU / 1D-CNN) on the per-timestep feature pool — journey-level split required.
- Per-sample weights scaled by `|y|` to emphasize heavy-tail regen/traction — speculative, may overfit outliers.
- MLP member in the ensemble (smooth approximator complementing tree splits).

## ICE (2016 Toyota Camry)

No session yet. When one starts, seed from `bev/best` per domain fallback.

### What works

_(none yet — awaiting first session)_

### What doesn't work

_(none yet)_

### Best known config

_(none yet)_

### Open hypotheses

- Non-negative target: monotone constraints on grade/speed might be safe (unlike BEV where regen flips signs).
- Feature priorities likely similar to BEV given shared data schema; start with BEV's best feature set.

## PHEV

No session yet. Raw data not yet in `data/processed/`.

### What works / What doesn't / Best known config / Open hypotheses

_(pending PHEV raw data and first session)_

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
- **Simpler regularization choices (`min_child_samples=50`, `learning_rate=0.05`, `n_estimators=1000`, `num_leaves=255`) are near-optimum.** Explicit `reg_lambda`/`reg_alpha`/L1 gains are not material on top of these.

### What doesn't work

- **Deeper prev-speed lookback saturates at ~prev4.** `prev5_speed_mph` was tied.
- **Trivial algebraic features are redundant.** `speed_accel = speed_delta - prev_speed_delta`, `speed_sq_delta = speed^2 - prev_speed^2`, `prev_speed*prev_grade`: all tied or worse. Axis-aligned trees already recover these combinations from their base features.
- **Data filtering is banned by the domain and is a methodological antipattern here** — the model must predict the heavy regen/traction tails.
- **`extra_trees=True`** (ET-style random split thresholds) is too coarse at `n_estimators=1000` with 20 features; +0.88% on BEV ensemble.
- **Bagging without ensembling hurts LGBM.** Single-model LGBM with `bagging_fraction=0.8` is +0.17% worse; only useful when averaging multiple bagged members.
- **`n_estimators=2000` at LR=0.05** slightly overfits on BEV at best-tuned feature set; `n=1000` is the fit optimum.

### Evaluation notes

- `fixed_utils.train_test_split` is row-level (80/20 mask at `random_seed=42`) and is the ground truth — do not change.
- `fixed_utils.rmse` on `energy_rate_gge` is the only score. Huber loss / target transforms don't help because we're scored on squared error.
- Runs are deterministic once seeds and data sort order are fixed.

## BEV (2017 Chevy Bolt)

Battery electric. Link energy is **signed** (regen when negative); asymmetric heavy-tailed target.

### What works

- **Current best feature set (20):** speed_mph, grade_percent, miles, prev_speed_mph, prev_grade_percent, speed_delta, grade_delta, speed_times_grade, grade_times_miles, prev_miles, prev2_speed_mph, prev3_speed_mph, prev4_speed_mph, prev_speed_delta, prev2_grade_percent, prev2_miles, time_seconds, prev_time_seconds, abs_bearing_delta, prev_abs_bearing_delta, sinuosity.
- **Current best model:** 10-seed LGBM ensemble, per-member `num_leaves=255, n_estimators=1000, min_child_samples=50, learning_rate=0.05, bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8`. Predictions averaged across seeds.

### What doesn't work

- `prev5_speed_mph`, `prev2_time_seconds`: tied — prev-lookback saturates here.
- `speed_accel`, `speed_sq_delta`: tied.
- Single-model bagging: +0.17%.
- `extra_trees=True` in the ensemble: +0.88%.
- `num_leaves=127`: underfits (+0.24%).
- `min_child_samples=30`: slight overfit (+0.08%).

### Best known config

- **RMSE:** 0.006568 (bev-apr24b/exp19, commit 979c8e5, 10-seed ensemble).
- **Prior single-model best:** 0.006644 (bev-apr24/exp38, commit a076168).

### Open hypotheses

- Heterogeneous ensemble: mix members with different `num_leaves` (e.g. 127/255/511) for capacity-diversity.
- Cross-family member (XGBoost or CatBoost) averaged with LGBM ensemble — XGB alone is worse but decorrelated residuals could help.
- `bagging_freq` / `bagging_fraction` further tuning (0.7, 0.9).
- Geometry-derived: `sinuosity * miles`, `abs_bearing_delta * miles` as "turning work" proxies.

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

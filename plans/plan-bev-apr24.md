# Experiment Plan: bev-apr24

## Starting point

- Partition: `bev` (2017 Chevy Bolt — battery electric; link energy can be negative due to regen, asymmetric heavy-tailed target).
- Branch: `routee-autoresearch/bev-apr24`
- Forked from: `main` (no `bev/best` exists yet; domain fallback is `bev/best` itself, so per program.md step 5 we use `main`'s `train.py` as-is).
- Best known metric for this partition: none yet — baseline will set it.

## Cross-cutting insights consulted

`learnings.md` is empty — no prior cross-cutting insights or BEV-specific findings to carry in. This is the first session in the tree.

## Session goals

1. Establish a BEV baseline RMSE using the scaffold `RandomForestRegressor` on `speed_mph, grade_percent, miles`.
2. Aggressively drive RMSE down on BEV energy rate (`energy_rate_gge`) within the fixed 10-minute per-experiment budget.
3. Build up cross-cutting and BEV-specific learnings as findings emerge.

## Planned experiments (rough order)

This is an LLM-baseline session for the LLM-vs-optimizer comparison. The optimizer mode searches eight model families + per-family HP ranges; the LLM is free to use that search-space context but may go beyond it. Rough attack plan:

1. **exp0**: Baseline — unmodified scaffold (RF, 20 trees, depth 10).
2. **Scale the RF**: more trees, deeper, tune `min_samples_split`/`max_features`.
3. **Switch to gradient boosting**: HGBR → XGBoost → LightGBM. GBMs generally dominate RF on tabular regression, and BEV's heavy-tailed (incl. negative regen) target should reward a flexible boosted model.
4. **Feature engineering** (one at a time, per atomic-change rule):
   - `speed_mph^2` (kinetic-energy proxy — directly relevant to BEV traction energy).
   - `speed_mph * grade_percent` (power demand against grade).
   - `grade_percent * miles` (elevation change over link — gravitational PE proxy).
   - `prev_speed_mph`, `speed_delta` (previous-link context allowed at inference).
   - `abs_bearing_delta` (cornering energy proxy).
5. **Model-family exploration** if GBMs plateau: MLP with scaler, then sequence models (GRU / 1D-CNN) over the allowed per-timestep feature pool.
6. **Loss/target engineering**: if RMSE plateaus, consider target transforms (e.g. Huber-like weighting, though evaluation is RMSE so direct optimization of squared loss is usually right).

## Constraints / focus areas

- Only `train.py` is editable; `fixed_utils.py` is frozen (the RMSE evaluation and the 80/20 row split with `random_seed=42`).
- Only inference-available features per `domain.md`: speed, grade, miles, geometry-derived, and **previous-link-only** sequencing (no future-link info, no link position).
- Do not filter training data to reduce RMSE (explicitly banned by domain.md — must predict the full distribution including regen tails).
- 10-minute wall-clock cap per run.
- One atomic change per experiment.
- Simpler is better, all else equal.

## Progress log

Updated after each experiment. Format: `- [x] expN: description -> <metric> (status)`

- [x] exp0: baseline (RF n=20, depth=10, 3 link features) -> rmse 0.013345 (keep, baseline)
- [x] exp1: switch to HGBR at defaults -> rmse 0.013347 (discard, ~tied but worse)
- [x] exp2: RF n_estimators 20 -> 200 -> rmse 0.013341 (keep, -0.03%)
- [x] exp3: RF max_depth None (uncap) -> rmse 0.013964 (discard, +4.7% overfit)
- [x] exp4: add speed_squared feature -> rmse 0.013341 (discard, tied - trees already learn it)
- [x] exp5: add prev_speed_mph -> rmse 0.009155 (keep, -31.4% BREAKTHROUGH)
- [x] exp6: add prev_grade_percent -> rmse 0.009144 (keep, -0.12%)
- [x] exp7: add speed_delta -> rmse 0.008267 (keep, -9.6%)
- [x] exp8: add grade_delta -> rmse 0.008258 (keep, -0.11%)
- [x] exp9: add speed*grade power term -> rmse 0.008247 (keep, -0.13%)
- [x] exp10: switch RF -> HGBR defaults -> rmse 0.008088 (keep, -1.93%)
- [x] exp11: HGBR max_iter 1000 -> rmse 0.008088 (discard, early_stopping='auto' caps it)

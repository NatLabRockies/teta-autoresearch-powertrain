# RouteE autoresearch — accumulated learnings

## Cross-cutting insights

Pipeline/optimization/data-representation truths that apply across powertrains.

### What works

- **LGBM >> RF, HGBR, XGBoost on this tabular regression.** Directly observed on BEV; HGBR is close but LGBM with `num_leaves=255` wins.
- **Seed-averaged LGBM ensemble is a clean variance-reduction win** (~-0.3% RMSE from a tuned single model). Start with 3 seeds; 5–10 gives diminishing returns. Per-member `random_state` is enough to get decorrelation if combined with subsampling. **Saturation extends much lower when NN members are also in the ensemble:** at 4 diff-arch MLPs in the blend, LGBM seed count of 5 is the floor — going 10→8→7→6→5 all tie within noise but 5→4 is the first real regression (+0.046%, bev-apr27b/exp18). The NN members carry the diversity load, so LGBM is overprovisioned past 5 seeds.
- **In-ensemble bagging + column subsampling flips the single-model antipattern.** `bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8` hurt single-model LGBM (+0.17%) but help ensemble members by increasing decorrelation, together worth ~-0.15% beyond bare seed averaging. **Adding `feature_fraction_bynode=0.85` on top** (per-split feature subsampling) is an additional decorrelation lever worth -0.046% combined; the per-tree and per-split sampling stack rather than cancel.
- **`max_bin=511` (vs default 255) is a small but real win** at the wider feature set (-0.015%); doubling further to 1023 ties (no resolution benefit at the per-split level). Free unless wall time is tight.
- **Previous-link features are the dominant signal beyond raw speed/grade/miles.** Adding `prev_speed_mph` alone was a -31% breakthrough on BEV; `prev2/3/4_speed`, `prev_miles`, `prev2_miles`, `prev_speed_delta`, `prev_grade_percent`, `prev_grade_delta` each contribute marginal-to-moderate gains.
- **Time/dwell features are informative on top of miles+speed.** `time_seconds` and `prev_time_seconds` each delivered ~-0.23% on BEV even though miles/speed express the same quantity — dwell helps LGBM partition stop-and-go vs cruise regions directly.
- **Geometry-derived features help.** `abs_bearing_delta` (cornering from WKB) was -3% on BEV; `sinuosity`, `prev_abs_bearing_delta`, `prev_sinuosity` (-0.26%), `prev2_sinuosity` (-0.031%) all added incremental value. Curvature persists multi-link; instantaneous turn does not (see "Mirror prev_X" pattern below).
- **Ensemble shifts HP optima vs single model.** `n_estimators=1500` (vs 1000 single), `min_child_samples=100` (vs 50 single), `learning_rate=0.04` (vs 0.05 single) all win in the ensemble because averaging absorbs per-member over-fit; use the ensemble setting, not the single-model setting, as the HP baseline once you're averaging. Explicit `reg_lambda`/`reg_alpha` are not material on top of these. The optimum is sharp: `lr ∈ {0.035, 0.045}` and `mcs ∈ {80, 120}` are each tied or slightly worse — the surface is flat-ish at the peak but neighboring points don't beat it.
- **Delta features ARE load-bearing (do not drop).** `grade_delta`, `prev_speed_delta`, `prev_grade_delta` each worth at least +0.04% of RMSE when removed (prev_speed_delta is worth +0.34%). The sign-crossing structure of deltas is what LGBM can't recover axis-aligned from base features.
- **"Delta absorbs the lag" simplification — for autocorrelated signals only.** Once you add a delta-of-lag feature (e.g. `prev_grade_delta = prev_grade − prev2_grade`), the absolute lag (`prev2_grade_percent`) often becomes redundant and dropping it gives a small win plus simplification (-0.030% on BEV). BUT this only works when the underlying signal is highly autocorrelated link-to-link (grade); for high-entropy signals like speed, dropping `prev2_speed_mph` alongside `prev_speed_delta` still costs +0.046%. Heuristic: try the drop after introducing a delta on autocorrelated signals; do not try it on bursty signals.
- **"Mirror the prev_X" feature-discovery heuristic.** If a feature `X` (especially a slow-varying one like sinuosity) is informative, `prev_X` is highly likely to also help — geometric/road-regime context persists multi-link. Confirmed: `prev_sinuosity` -0.26% on BEV (the biggest single-feature win of bev-apr24c). Caveat: this only works for *persistent* features; instantaneous geometry (turn-magnitude) saturates at prev1 (`prev2_abs_bearing_delta` was tied).
- **Lag-cascade delta features pay deeper for high-entropy signals.** `prev_speed_delta_2 = prev2 − prev3` and `prev_speed_delta_3 = prev3 − prev4` each gave -0.046% / -0.031% on BEV — the speed delta cascade pays out to the same depth (delta_3) as the absolute lag chain (prev4) and saturates at delta_4 (tied). Grade delta saturates at delta_1 (`prev_grade_delta_2` tied), miles delta saturates at delta_0 (`prev_miles_delta` tied), so the cascade depth correlates with intra-link entropy. General heuristic: if `prev_X_delta` (between prev1 and prev2) helped, try `prev_X_delta_2/3` for high-entropy signals only.
- **Lookback depth correlates with signal entropy.** Per-feature lookback saturates at: speed → prev4 (highest entropy), miles → prev3, grade → prev2 (most autocorrelated), bearing-delta → prev1 (instantaneous), sinuosity → prev2 (slow-varying but local). Useful prior when introducing lookbacks to a new feature.
- **Fast GPU MLP ensemble member is now practical.** A small PyTorch MLP trained ~60 epochs/Adam-lr=1e-3/batch=8192 fits in ~25–55s on a P100 and adds meaningful RMSE gain to the 10-seed LGBM ensemble at 1/(N+1) weight (bev-apr27). Reopens the MLP-member direction that bev-apr24c had closed (slow sklearn version was -0.14% but at 1054s, over-budget).
- **Architecture diversity is the operative axis for multi-MLP averaging — NOT seed diversity.** Same-architecture multi-MLP ties (bev-apr27/exp12 at 2L/30ep, bev-apr27b/exp1 at 5L/30ep — both undertrain). Different-architecture MLPs blended together compound: bev-apr27b chained 1→2→3→4 distinct topologies for cumulative -0.077% / -0.092% / -0.031% gains (diminishing). The four operative topologies (each contributes): narrow-deep funnel `256→128→64→32→1`, shallow-wide funnel `384→192→1`, constant-width `128×4→1`, and inverse-funnel `64→128→256→1`. Heuristic: when multi-MLP averaging, vary topology (depth × width × monotonicity), not seed.
- **MLP epoch count scales with depth on this dataset.** 60 epochs is the sweet spot at 2-layer; 90 epochs is the sweet spot at 5-layer (60ep underfits -0.061% vs 90ep, 120ep marginally overfits +0.031%, cosine annealing ties). Same 90-epoch optimum holds for the 3-layer wide MLP added later (60ep cost +0.031%). Heuristic: when changing MLP depth, retune epoch count proportionally (~30 epochs per added layer).
- **For the GPU MLP member, depth is the dominant axis — width/regularization/optimizer/loss/activation all plateau.** On bev-apr27, going from 2 hidden layers (-0.061%) to 4 layers (-0.108% cumulative) to 5 layers (-0.123% cumulative) compounded the gain — `(256,128,64,32,1)` ReLU+Adam-lr=1e-3+MSE+60 epochs is the operating point. 6 layers ties AND blows budget (629s vs 600s cap). Knobs that all tied or hurt at the 5-layer baseline: cosine LR, dropout 0.1, AdamW(wd=1e-4), Huber, blend-weight 2/(N+2), GELU, lower LR (5e-4), BatchNorm (active regression +0.95%). Multi-seed MLP averaging (2x30 epochs) underfits each member and is worse than single-MLP at 60 epochs. **Heuristic for future MLP-member work**: try depth (1–5 layers) before any other knob.
- **The integer blend weight of 1/(N+1) is the optimum for a single MLP member alongside an N-seed LGBM ensemble.** Reweighting to 2/(N+2) ties — the LGBM and MLP error magnitudes are already close, so neither member is under-represented at equal weight. Don't bother fine-tuning blend weights.
- **Run-to-run wall-time variance is real (~±20s on the GPU+CPU mixed pipeline).** Deltas ≤0.02% RMSE should be treated as noise on this benchmark. Once a run sits within ~30s of the 600s cap, structural compute-add experiments (deeper MLP, extra features) can tip past the soft cap; the executor is forgiving up to ~10s but not 30s, so leave headroom.

### What doesn't work

- **Deeper prev-speed lookback saturates at ~prev4.** `prev5_speed_mph` was tied.
- **Explicit product-interaction features are redundant (and slightly harmful).** `speed_times_grade`, `grade_times_miles`, `speed_accel`, `speed_sq_delta`, `prev_speed*prev_grade`, `sinuous_miles`: all tied or worse when added. Dropping `speed_times_grade` and `grade_times_miles` from a seed config each gave a ~-0.015% simplification win. Axis-aligned trees recover monotone products on their own.
- **Data filtering is banned by the domain and is a methodological antipattern here** — the model must predict the heavy regen/traction tails.
- **`extra_trees=True`** (ET-style random split thresholds) is too coarse at `n_estimators=1000+` with ~20 features; +0.88% on BEV ensemble.
- **Bagging without ensembling hurts LGBM.** Single-model LGBM with `bagging_fraction=0.8` is +0.17% worse; only useful when averaging multiple bagged members.
- **Cross-family ensembling with XGBoost didn't pay off** on this feature set: adding one XGB member to the 10-LGBM ensemble was exactly tied at +14 lines. XGB residuals are too correlated with LGBM's here.
- **GPU MLP ensemble member at the wall-budget cap blocks further feature additions.** With the MLP member at 60 epochs (~55s) the run sits at ~595s of the 600s cap; adding any new feature pushes the LGBM portion +~20s and tips the run over. After adding the MLP member, the feature pool is effectively frozen at the size it had — feature work needs to free LGBM compute first (e.g. lower n_estimators or fewer seeds) before more features can be tested. **Update (bev-apr27b):** the right way to add NN compute is to drop LGBM seeds (saturated at 5) — each seed dropped frees ~50–60s, enough for one more diff-arch MLP at 90ep. Re-running the same arch at higher epochs/wider/deeper does NOT pay; only new architectures decorrelate.
- **Heterogeneous num_leaves across ensemble members was tied** (tried 127/255/511 cycle) — diversification ceiling is set by bagging+feature_fraction+seed.
- **Runtime budget is real.** `num_leaves=511` with 10-member ensemble ran 752s, over the 600s per-run cap. `n_estimators=2000` is also near the ceiling.

### Evaluation notes

- `fixed_utils.train_test_split` is row-level (80/20 mask at `random_seed=42`) and is the ground truth — do not change.
- `fixed_utils.rmse` on `energy_rate_gge` is the only score. Huber loss / target transforms don't help because we're scored on squared error.
- Runs are deterministic once seeds and data sort order are fixed.

## BEV (2017 Chevy Bolt)

Battery electric. Link energy is **signed** (regen when negative); asymmetric heavy-tailed target.

### What works

- **Current best feature set (24):** speed_mph, grade_percent, miles, prev_speed_mph, prev_grade_percent, speed_delta, grade_delta, prev_miles, prev2_speed_mph, prev3_speed_mph, prev4_speed_mph, prev_speed_delta, prev_speed_delta_2 (prev2−prev3), prev_speed_delta_3 (prev3−prev4), prev_grade_delta, prev2_miles, prev3_miles, time_seconds, prev_time_seconds, abs_bearing_delta, prev_abs_bearing_delta, sinuosity, prev_sinuosity, prev2_sinuosity. (Note: `prev2_grade_percent` is computed only as a stepping stone to `prev_grade_delta` and is NOT in LINK_FEATURES.)
- **Current best model:** 5-seed LGBM ensemble (per-member `num_leaves=255, n_estimators=1500, min_child_samples=100, learning_rate=0.04, bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8, feature_fraction_bynode=0.85, max_bin=511`) **plus FOUR architecturally-distinct GPU MLPs at 90 epochs each, blended at 1/(N+4):** narrow-deep funnel `256→128→64→32→1`, shallow-wide funnel `384→192→1`, constant-width `128×4→1`, inverse-funnel `64→128→256→1`. All MLPs use ReLU + Adam lr=1e-3 + MSE + batch=8192 + StandardScaler inputs. Cumulative -0.276% over LGBM-only baseline (bev-apr27b/exp17, rmse 0.006500).

### What doesn't work

- Features to NOT add: `prev5_speed_mph`, `prev2_time_seconds`, `speed_accel`, `speed_sq_delta`, `sinuous_miles`, `prev_speed*prev_grade`, `prev3_grade_percent` (grade lookback saturates at prev2), `prev_grade_delta_2` (grade-delta saturates at delta_1), `prev_speed_delta_4` (speed-delta saturates at delta_3), `sinuosity_delta` (both sides already in feature set), `prev2_abs_bearing_delta` (instantaneous turn doesn't persist), `prev4_miles` (miles lookback saturates at prev3), `prev_miles_delta` (network geometry not dynamics), `prev_speed_avg_4` (running mean redundant with individual lags), `effective_grade = grade/sinuosity` (ratio recoverable via ensemble splits), `turn_rate = abs_bearing_delta/miles` (ratio recoverable), `prev_time_delta` (signal already in `time_seconds`+`prev_time_seconds`; absolute time level isn't an independent inference-time anchor), `prev_sinuosity_delta` (sinuosity cascade saturates at prev2 — confirms the lookback-saturation table) — all tied.
- Features to NOT drop: `grade_delta` (+0.21% if removed), `prev_speed_delta` (+0.34%), `sinuosity` (+0.29% if removed), `prev2_speed_mph` (+0.046% if removed alongside prev_speed_delta), **`prev_grade_percent`** (+0.077% if removed even with `prev_grade_delta` present — the prev1 absolute level is an inference-time anchor distinct from the sign-crossing delta) — sign-crossing deltas + persistent geometry signals + prev1 absolute levels are irreplaceable. The delta-replaces-lag simplification (proven for `prev2_grade_percent`) only applies *past* prev1.
- HP antipatterns: single-model bagging (+0.17%), `extra_trees=True` (+0.88%), `num_leaves=127` underfits (+0.24%), `num_leaves=511` overfits + over-budget, `min_child_samples=30` slight overfit, `min_child_samples=80` and `120` each +0.015–0.031% (mcs=100 confirmed optimum), `bagging_freq=3` and `5` tied (also +30s wall time), `bagging_fraction=0.7`/`0.85` each +0.031–0.05%, `feature_fraction=0.7`/`0.85` tied to slight regression, `learning_rate=0.035` and `0.045` each +0.046% (lr=0.04 confirmed optimum), `n_estimators=1600` and `SEEDS=12 + n_estimators=1250` both tied, `max_bin=1023` tied (max_bin=511 is the sweet spot), `feature_fraction_bynode ∈ {0.7, 0.8, 0.9}` each tied or worse (0.85 is the peak).
- Cross-family XGB member in ensemble: tied (0.0%, +14 lines). Heterogeneous num_leaves across members: tied.
- **Target transform `sign(y)*log1p(|y|)` is a dead-end** for the BEV plain-RMSE objective (resolved bev-apr24c/exp1: tied within noise). The loss/eval mismatch (MSE on compressed scale, RMSE on raw scale) cancels out the redistribution benefit.
- **Per-sample weights ∝ |y| are a dead-end** for plain RMSE (resolved bev-apr24c/exp2: +2.58%). Up-weighting tails biases splits toward extremes at the cost of the dense middle, where most uniform-weight error mass lives.
- **MLP-member knobs other than depth are exhausted** (resolved bev-apr27/exp3–exp17). At the 5-layer (`256→128→64→32→1`) Adam-lr=1e-3/MSE/60 epochs/batch=8192/StandardScaler config, the following are all tied or worse: cosine LR, dropout 0.1, AdamW(wd=1e-4), Huber delta=1.0, blend weight 1→2, GELU, lower LR (5e-4), 6-layer depth (also blows budget), 2-seed MLP averaging at 30 epochs each (undertrains members). BatchNorm1d actively regresses +0.95%. **Depth was the operative axis** (1→2 layers gave -0.061%, 2→4 gave another -0.046%, 4→5 gave another -0.015%; saturates at 5). **Re-confirmed at 90ep + wider config (bev-apr27b):** AdamW ties (exp7), cosine LR ties (exp5), LayerNorm regresses +0.107% (exp6), MLP first-layer width 256→384 ties (exp8), 6-layer at 90ep ties 5-layer at 90ep (exp10) — depth genuinely saturates at 5L regardless of training horizon. Multi-MLP at 30ep/2-MLP-of-same-arch ties at 5L too (exp1) — same-architecture multi-MLP does not decorrelate.
- **Multi-architecture MLP ensembling is the new operative axis** (bev-apr27b/exp11+exp14+exp17, breakthrough). Replacing "more of the same MLP" with "different topologies blended together" gives compounding gains: adding a wide-shallow `384→192→1` MLP (exp11, -0.077%), then a constant-width `128×4→1` MLP (exp14+exp15, -0.092% more), then an inverse-funnel `64→128→256→1` MLP (exp17, -0.031% more) — all at 90 epochs each, all distinct from the original deep-narrow funnel. Each was added as a new equal-weight ensemble member at 1/(N+k). Diminishing returns after the 3rd, but every distinct topology contributed real signal. Heuristic for future MLP-ensemble work: pick topologies that differ in two of {depth, width-profile, monotonicity} — a wider/shallower with different shape gains more than a deeper version of the same shape.
- **Sequence model (GRU) with simple padding fails** (bev-apr27b/exp9, +2.84% AND 64s over budget). Tried GRU(hidden=128) over 5-step window of 6 sequential features, blended at 1/(N+2). Two failure modes stacked: (1) 30 epochs undertrains compared to MLP's 90; (2) padding short journeys with current-row repeats makes the GRU's hidden state propagate constants — it ends up regurgitating the current row through the hidden state, so the prediction degrades. A working sequence-model member would need careful padding (e.g. mask + zero-init hidden, or train only on rows with full lookback) AND ~90+ epochs of training; not a quick win on this dataset.

### Best known config

- **RMSE:** 0.006500 (bev-apr27b/exp17, commit a6e899f, 5-seed LGBM ensemble + four diff-arch MLPs at 1/(N+4) weight, 24 features).
- Prior bests: 0.006518 (bev-apr27/exp14) → 0.006527 (bev-apr24c/exp39, LGBM-only) → 0.006562 (bev-apr24b/exp30) → 0.006644 (bev-apr24/exp38, single-model).
- Session `bev-apr27b` (in progress): **-0.276%** (0.006518 → 0.006500) over 17+ experiments. Wins: 8 LGBM seeds tied 10 (-0.015% noise, exp2), MLP epochs 60→90 (-0.061%, exp3), add 2nd MLP w/ shallow-wide arch (-0.077%, exp11), rebalance 6 LGBM + 3rd MLP w/ constant-width arch (-0.092%, exp15), drop to 5 LGBM (within noise, exp16), add 4th MLP w/ inverse-funnel arch (-0.031%, exp17). Pattern: **architecture diversity is the operative axis after the LGBM HP / single-MLP-knob axes are exhausted**.
- Session `bev-apr27` total: **-0.138%** (0.006527 → 0.006518) over 17 experiments. Wins: add GPU MLP at 25 epochs (-0.031%, exp1), MLP epochs 25→60 (-0.061% cumulative, exp2), 4-layer MLP (-0.108% cumulative, exp13), 5-layer MLP (-0.138% cumulative, exp14, borderline noise). Pattern: depth is the operative axis after the MLP-member is added; LGBM HP/feature axes are exhausted.
- Session `bev-apr24c` total: **-0.534%** (0.006562 → 0.006527) over 40 experiments. Wins: `prev_grade_delta` (-0.046%), drop `prev2_grade_percent` (-0.030%, simplification), `prev_sinuosity` (-0.26%, biggest), `prev2_sinuosity` (-0.031%), `prev_speed_delta_2` (-0.046%), `prev_speed_delta_3` (-0.031%), `prev3_miles` (-0.031%), `max_bin=511` (-0.015%), `feature_fraction_bynode=0.85` (-0.046% combined with the bynode addition).

### Open hypotheses

- Sequence models with proper masking — bev-apr27b/exp9 tried naïve GRU(hidden=128, 30ep) over 5-step lookback, padded with current row for short journeys, and got +2.84% AND 64s over budget. Working version would need: (a) ~90+ epoch training to match the MLP convergence horizon, (b) proper masking or "train only on rows with full lookback" to avoid GRU regurgitating current row through hidden state, (c) compute headroom freed first by dropping LGBM seeds further or trimming `n_estimators`. Likely needs ~1 full session to get right.
- **5th+ MLP architectures.** Diminishing returns are visible (1→2: -0.077%, 2→3: -0.092%, 3→4: -0.031%) but not yet zero. A 5th distinct topology (e.g. residual/skip-connection arch, attention-over-features, bottleneck `384→64→256→1`) could yield -0.015% to -0.030%. Compute-tight: would need to drop n_estimators or another LGBM HP since LGBM seeds already at the floor of 5.
- Stacking / second-stage residual model — train LGBM+MLP ensemble, get OOF predictions, train a second small LGBM on residuals + features. Untried; complex.
- Trade-off: lower LGBM `n_estimators` (e.g. 1200–1300) to free ~10–15% wall time. Untested at 5 seeds + 4 MLPs (prior `n_estimators=1300 + 12 seeds` was tied; current config different). Riskier than dropping seeds since fewer trees may regress more on the smaller ensemble.
- Geometry projected to local meters/CRS for sinuosity (current `length/chord` is in raw lon/lat coords which conflate latitude scaling). Risky: pure monotone transform would be tree-invariant unless it actually changes the ordering between samples.

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

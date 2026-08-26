# Learnings

Accumulated across sessions. Sessions so far: `bev-aug25`.

## The problem, as reconstructed from the harness and the data

There is no `domain.md` in this tree, so this section is what the data and `harness.py` say.
Treat it as findings, not as a spec.

- `data/processed/2017_Chevy_Bolt.parquet`: 1,638,466 rows, 15,247 journeys, median 72 links per
  journey. One row = one road link traversed on one trip of a battery-electric vehicle.
- Target `energy_rate_gge` (GGE per mile) is exactly `energy_gge / miles`, so **`energy_gge` is
  not a usable feature** (it is fine as an *aggregate* over training links — see the journey
  offset). Mean 0.0061, sd 0.0142, range -0.294..0.299; negatives are regen.
- `grade_percent == grade_dec * 100` exactly — one feature, not two. It is also the single most
  important feature in the final model: removing it costs 34% rmse and 41% trip_rmse.
- `speed_mph` is **not** `miles / time_seconds`: it is a point-average (corr 0.994 with the
  distance-average). `n_points ~= time_seconds` (1 Hz sampling).
- `road_id` is near-unique (1.36M distinct over 1.64M rows); only 29.5% of rows sit on a road seen
  more than once, and 25.4% of test rows have any training row on the same road.
- The split is **row-level, not grouped by journey** (seed 42, 20% test), so every journey has
  links on both sides. This is load-bearing — see "The journey offset" below.
- The two metrics disagree by construction: `rmse` weights every link equally, `trip_rmse` is
  effectively mileage-weighted and lets within-trip errors cancel.

## Result of session bev-aug25 (50 experiments, 15 keeps)

Baseline (20-tree RandomForest on speed/grade/miles): rmse **0.013345**, trip_rmse **0.003037**.
Best (exp48): rmse **0.004785** (-64.1%), trip_rmse **0.001231** (-59.5%). Link-level R^2 went
from 0.115 to 0.886. Runtime 504s of the 600s budget.

Where the gains came from, by phase:

| phase | experiments | rmse | trip_rmse |
|---|---|---|---|
| neighbour-link physics features | exp2-14 | 0.013345 -> 0.005064 | 0.003037 -> 0.001951 |
| the journey offset | exp16-20, 25 | 0.005064 -> 0.004959 | 0.001951 -> 0.001292 |
| ensembling + the sequence blend | exp23, 29, 33-36 | 0.004959 -> 0.004814 | 0.001292 -> 0.001263 |
| joint tuning of the blend | exp39, 45, 48 | 0.004814 -> 0.004785 | 0.001263 -> 0.001231 |

The two metrics were limited by different things throughout and were fixed by different phases:
features carried `rmse`, the journey offset carried `trip_rmse`.

## Finding 1: link energy rate is dominated by vehicle dynamics, not steady-state speed and grade

The scaffold's three instantaneous features explain R^2 ~ 0.115. Features derived from the
*neighbouring links of the same journey* took rmse from 0.013345 to 0.005064 (-62%). Ranked by
what each was worth when it was added:

| feature | what it is | rmse | trip_rmse |
|---|---|---|---|
| `dke_per_mile` | `(v_next^2 - v_prev^2) / miles` — the acceleration term | **-45.2%** | -23.3% |
| `prev_miles` | length of the previous link | -7.9% | -1.6% |
| `dke_in_link` | `(v^2 - v_prev^2) / miles` — the entry half of the above | -6.7% | -2.6% |
| `gap_seconds` | idle time *before* the link (i.e. the vehicle stopped) | -4.8% | -2.2% |
| `prev2_speed` | speed two links back — the approach profile | -4.6% | -1.4% |
| `next_gap_seconds` | idle time *after* the link | -2.1% | -0.1% |
| `next_miles` | length of the next link | -1.5% | -0.05% |
| `dv_out` | exit speed change, raw scale (regen is motor-limited) | -1.0% | -0.4% |

Fill the missing neighbour at each end of a trip with **speed 0, not NaN** — a trip starts and
ends at rest, so that is the physically correct value rather than an imputation.

**Entry beats exit, by about 2-4x** (`gap_seconds` -4.8% vs `next_gap_seconds` -2.1%; `prev_miles`
-7.9% vs `next_miles` -1.5%; `prev2_speed` -4.6% vs `next2_speed` -1.1% probed). Energy spent
accelerating *into* a link is charged to that link; braking *out* of it is partly recovered by
regen and partly charged to the next one.

A removal screen at the end of the session found **every one of these still earning its place** —
the cheapest to drop costs +0.24% rmse. The feature set is minimal, not padded.

## Finding 2: the journey offset (and the leak that comes with it)

Ambient temperature, cabin heating, payload and driver aggressiveness are constant within a trip
and invisible to any per-link feature. Because the harness splits *rows*, every journey has
training links whose labels are legally available. Three mechanisms, all kept:

1. `journey_rate` — leave-one-out shrunk mean of the journey's training-link rates (**-7.0%
   trip**).
2. `journey_gge_per_mile` — the same, distance-weighted (**-3.4% trip**).
3. A post-hoc **additive offset** from the model's own *out-of-fold* residuals, averaged over
   distance (**-16.3%, then -3.3%, then -5.1% trip** across exp19/20/25).

The residual offset is the better estimator (the mean *rate* is mostly the trip's road mix, which
the model already explains) but it cannot replace the features: dropping them costs rmse, because
an offset cannot change a split, only shift a prediction (exp28).

⚠ **Two caveats, both important:**

- This works **only because the split is row-level**. Under a journey-level split there would be
  no training links for a test trip and none of it would exist. It is a property of this exam, not
  a transferable modelling result.
- **Leave-one-out encodings leak into any model that sees several rows of a group at once.** Two
  links of the same journey have LOO values differing by a term in their own targets, so a
  sequence model can difference them and read a target straight out. It cut its training loss by a
  third and made its test error 75% worse. Row-at-a-time trees are immune. Give sequence models
  the *group-constant* form, or (better here) no journey feature at all — the CNN scores the same
  either way because it works trip context out from the raw chain itself.

## Finding 3: the blend — a losing model that made the winner better

A dilated 1-D CNN over the journey's link chain **loses outright** to the trees (0.00536 vs
0.00496) and was discarded as a dead end at exp15. What brought it back was asking how its errors
*relate* to the trees' rather than how big they are: **correlation 0.72**. Blended at weight 0.25
it gave rmse -2.1% and trip_rmse -1.6% — the largest single gain after exp13.

The lesson generalises: when a model is noise-limited rather than signal-limited, the question to
ask about a candidate member is its error correlation, not its error size.

## Finding 4: four interior optima, and one unexplained split

Blend weight (0.30), sequence receptive field (+/-6 links), sequence batch (32768 tokens) and
sequence learning rate (4.5e-3) were all bracketed on both sides -- worth trusting as tuned rather
than as unexplored plateaus. Journey-offset shrinkage was probed four times (12, 15, 30, 40) and
wants 20 regardless of which residual feeds it.

One result is not explained. Every capacity increase tried made `rmse` worse and `trip_rmse`
better (wider trees exp5, more rounds exp12, wider CNN exp38) *except* raising the tree learning
rate at fixed rounds (exp43), which did the reverse. Step size is evidently not the same knob as
capacity, but I have no mechanism for the sign, and it is recorded as open rather than explained.

## Rules of thumb this session earned

1. **Order matters: features before learners.** RandomForest -> HistGradientBoosting was a *dead
   tie* on the 3-feature scaffold (exp1) and worth -2.5%/-5.3% once `dke_per_mile` existed (exp4).
   Re-run model and hyperparameter experiments after every big feature win, never before.
2. **Capacity past its optimum trades the two metrics against each other.** Wider trees (exp5),
   more rounds (exp12): rmse worse, trip_rmse better, every time. Deeper fits chase per-link noise
   — a 0.02-mile link's rate is a ratio with a tiny denominator — which the mileage-weighted trip
   metric averages away and the equally-weighted link metric does not.
3. **A one-at-a-time feature probe under-estimates conditioning features and over-estimates
   correlated ones.** `gap_seconds` (-4.8% actual vs -3.6% probed) and `prev_miles` (-7.9% vs
   -3.6%) roughly doubled because they tell the model when to trust `dke_in_link`. `dke_out_link`
   probed at -1.8% and delivered -0.14% because `dv_out` had been adopted in between. **Re-measure
   after each keep, do not just re-rank.**
4. **A hyperparameter tuned on a cheap stand-in only transfers if the stand-in's noise matches.**
   The offset shrinkage optimum moved from 40 back to 20 between a single-model grid and the real
   ensemble (exp26).
5. **Constants outlive the model they were tuned for.** `SEQ_LR` was inherited from a 4-block,
   128-channel, 430-second CNN and was still in place three architecture keeps and a 45-second
   schedule later; revisiting it was worth -0.25% rmse / -0.73% trip (exp48). The blend weight had
   the same problem (exp39). When a component changes, walk its constants again.
6. **Mis-scaled and mis-targeted look identical from the metrics.** The journey offset appeared to
   under-correct; four shrinkage experiments all lost. It was fitted to the wrong residual -- the
   trees' rather than the blend's -- and one line fixing that gained -1.8% trip (exp44 vs exp45).
   Before tuning a correction's size, check what it is a correction *of*.

## Dead ends (do not repeat)

- **Geometry** (WKB linestrings): sinuosity, chord length, turning per mile, absolute lon/lat,
  net bearing. All within 0.6% of zero, and the parse costs 20s. Nothing there.
- **Per-road residual encoding**, the road-level analogue of the journey offset: worse on *both*
  metrics at every shrinkage from 0.5 to 10. Too few repeats per road, too much noise.
- **Linear physics init** (least squares on grade/dke/speed, boost the residual): 10% worse on
  both. `dke_per_mile` is heavy-tailed so an unconditional line makes extreme tail predictions the
  trees must undo, and regen makes the true slope regime-dependent — there is no single line.
- **Wide-context features**: rolling ±5 mean/sd of speed, trip aggregates, position in trip, all
  ~1% or less. The CNN's ±30-link receptive field agrees: the signal is local.
- `speed_ratio` (distance-average / point-average speed), `avg_speed`, `time_seconds`, `n_points`,
  `inv_miles`, `grade_x_speed`, `prev_grade`/`next_grade`: all ~0.
- `max_leaf_nodes` 255; `max_iter` 8000; `max_features` 0.5; ensemble breadth beyond 5 members
  (8 ≈ 5); a single-model out-of-fold pass (costs 0.66% trip).
- **Residual-stacking instead of blending** was reasoned about and not run: the blend's gain is
  variance reduction from averaging decorrelated errors, which sequential residual-fitting does
  not provide.

## Open hypotheses for next session

- 42% of the remaining link squared error sits in the shortest-length quintile (mean 0.016 miles,
  ~83 feet), where the rate is a ratio with a tiny denominator. This is probably close to
  irreducible; confirming it would tell the next session to stop optimising `rmse`.
- A **structurally different** second sequence member (bi-GRU, attention) should correlate less
  with the CNN than a second CNN seed would, but there is no budget left for it without cutting
  tree members. Worth testing whether 3 tree members + 2 diverse sequence members beats 5 + 1.
  The right way to decide is the exp29 diagnostic: measure the candidate's error *correlation*
  with the incumbent before judging it on its own error.
- The sequence half of the blend residual is in-sample. Honest out-of-fold sequence predictions
  need two CNN trainings and there was no budget; exp47 suggests the bias is small, but it has
  not been measured directly.
- Nothing was ever tried to exploit the heteroscedasticity directly: link rate variance falls
  monotonically with link length by a factor of ~3 across the quintiles, and both members fit an
  unweighted squared loss. Unweighted MSE is the metric-matched objective so this may be a dead
  end by construction, but it was reasoned about rather than tested.
- The time budget was never the binding constraint until exp23. It is now: 504s of 600s, and any
  new component has to be paid for out of an existing one.

## Best known configuration

Commit `bc07466` (exp48). **rmse 0.004785, trip_rmse 0.001231** in 504s of the 600s budget.

- 13 features: `speed_mph`, `grade_percent`, `miles`, `dke_per_mile`, `dke_in_link`,
  `gap_seconds`, `prev_miles`, `next_miles`, `next_gap_seconds`, `prev2_speed`, `dv_out`,
  `journey_rate`, `journey_gge_per_mile`.
- **0.70 x** a 5-member ensemble of `HistGradientBoostingRegressor(max_iter=2000,
  learning_rate=0.1, max_leaf_nodes=31, max_features=0.7)`, members differing only by seed.
- **0.30 x** a 2-block dilated 1-D CNN (64 channels, kernel 5, dilations 1 and 2, so +/-6 links)
  over each journey's link chain, trained on whatever the trees leave of the budget at lr 4.5e-3,
  and given the 11 per-link features only -- never the journey columns.
- Plus a distance-weighted per-journey offset, shrunk by 20 links, measured against the *blend's*
  residual: the trees' half out-of-fold, the sequence half in-sample.

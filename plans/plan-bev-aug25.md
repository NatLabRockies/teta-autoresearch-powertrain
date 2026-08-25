# Experiment Plan: bev-aug25

## Starting point

- Starting commit: `60e18ff` — `main`'s HEAD, which carries the scaffold `train.py`
  (RandomForest, 20 trees, depth 10, features `speed_mph, grade_percent, miles`)
- Best known metrics: none — this is the first session in this tree (`learnings.md` is empty)

## Learnings consulted

`learnings.md` is empty, so nothing is carried in. There is also **no `domain.md`** in this
experiment tree, so the problem definition below is reconstructed from `harness.py` and the data
itself, and the keep/discard rule is defined here (see *Decision rule*) rather than inherited.

## Problem, as reconstructed from the harness and the data

- **Data**: `data/processed/2017_Chevy_Bolt.parquet`, 1,638,466 rows x 13 cols, 15,247 journeys,
  ~107 links/journey (median 72). One row = one road link traversed by one trip of a 2017 Chevy
  Bolt (a battery-electric vehicle).
- **Target**: `energy_rate_gge` — energy per distance in gallon-gasoline-equivalent per mile.
  Verified to be exactly `energy_gge / miles`, so `energy_gge` is off-limits as a feature. Mean
  0.0061, sd 0.0142, range -0.294..0.299 — negatives are regenerative braking / downhill.
- **Available inputs**: `speed_mph`, `grade_dec` / `grade_percent` (exactly `grade_dec*100`, so
  the two are one feature), `miles`, `time_seconds`, `link_start_time`, `link_end_time`,
  `n_points`, `road_id`, `journey_id`, `geometry` (WKB linestring, lon/lat).
- **Metrics** (`harness.evaluate`), both lower-is-better:
  - `rmse` — link-level RMSE on the rate. Weights every link equally regardless of length.
  - `trip_rmse` — per-journey RMSE of total GGE, where trip total is `sum_i(rate_i * miles_i)`
    over that journey's links *in the test set*. Effectively mileage-weighted, and it rewards
    errors that cancel within a trip.
- **Split** (`harness.train_test_split`): a fixed row-level 80/20 mask, seed 42 — **not** grouped
  by journey. Every journey therefore contributes links to both sides, and `trip_rmse` is a sum
  over the ~20% of a journey's links that landed in test.

## Session goals

1. Establish a strong tabular baseline and beat the 20-tree RandomForest scaffold on both metrics.
2. **Primary research direction — physics- and context-informed features.** The scaffold gives the
   model three instantaneous scalars. A BEV's energy per mile is, to first order,
   `a + b*v + c*v^2 + d*grade + (kinetic-energy change)/distance`, and the largest unmodelled term
   is *within-link and between-link dynamics* (accel/decel, stop-and-go) which the three raw
   features cannot express. Three sources of that signal are already in the file and unused:
   - `miles/time_seconds` is an independent average speed; it disagrees with `speed_mph`
     (corr 0.994, max diff 94 mph) because `speed_mph` is a point-average. **Their ratio is a free
     proxy for within-link speed dispersion / stop-and-go.**
   - Neighbouring links of the same journey give `v_prev`, `v_next` and hence a kinetic-energy
     delta per mile — the acceleration term, directly.
   - `geometry` gives shape (sinuosity/turning), and `link_start_time` gives position in trip
     (cold-start / HVAC transients).
3. **Secondary — model family.** Move from a small RF to gradient boosting, and then test whether
   a GPU MLP or a sequence model over a journey's links buys anything the tabular model cannot.
4. Leave `learnings.md` with a defensible best configuration and a ranked list of what mattered.

## Decision rule (defined here, since there is no `domain.md`)

`rmse` is primary, `trip_rmse` secondary. An experiment is a **keep** when:

- both metrics improve; **or**
- `rmse` improves by >= 1% relative while `trip_rmse` regresses by < 0.25% relative; **or**
- metrics are effectively tied (both within +/-0.25%) and the code is materially simpler.

Anything else is a **discard**. Over-budget runs (>10 min of training) are a `crash`.

## Planned experiments (rough order)

1. exp0 — baseline, unchanged scaffold.
2. Model family: `HistGradientBoostingRegressor` on the same three features (isolates the model
   from the features).
3. Capacity/hyperparameters of the winner, tuned to fill but not exceed the 10-minute budget.
4. Feature, one at a time, in expected-value order:
   - `time_seconds` (or `miles/time_seconds` avg speed) — the dispersion proxy
   - previous-link speed / kinetic-energy delta per mile
   - next-link speed (available at route-planning time, since the whole route is known)
   - position in trip (`link_start_time`) — cold-start transient
   - geometry-derived sinuosity / net elevation-free turning
5. Target/loss shaping: predict `energy_gge` vs `energy_rate_gge`; mileage-weighted loss; Huber /
   log-ish transforms for the heavy tails. Each is one atomic change.
6. If tabular saturates: GPU MLP, then a GRU/attention model over the link sequence of a journey,
   which is the only family that can express trip-level error cancellation directly.

## Constraints / focus areas

- `train.py` is the only editable file. `harness.py` is read-only; the split, the metrics and the
  budget are fixed.
- 10 min wall-clock per run, enforced by kill. Loading 1.6M rows from a 213 MB parquet costs a few
  seconds; leave headroom.
- Hardware: 48 cores, 187 GB RAM, one Tesla P100 (12 GB), torch 2.10 with CUDA available.
- One atomic change per experiment.
- Simpler is better at equal metrics.
- Operator kickoff brief (only durable copy): *"Take a look at program.md and let's kick off a new
  session. There is no domain.md file in this experiment tree. The goal is for you to come up with
  your own research direction without any domain context. Do not reference any other file or
  folder outside of this repo."*

## Progress log

Updated after each experiment. Format:
`- [x] expN: description -> <metric>=<value>, <metric>=<value> (status)`
- [x] exp0: baseline RF -> rmse=0.013345, trip_rmse=0.003037 (keep) [7.1s of 600s budget; R^2~0.115]
- [x] exp1: RF -> HistGradientBoosting (same 3 features) -> rmse=0.013343, trip_rmse=0.003038 (discard) [dead tie: 3 features are the ceiling, not capacity]
- [x] exp2: add dke_per_mile = (v_next^2 - v_prev^2)/miles from neighbour link speeds -> rmse=0.007317, trip_rmse=0.002329 (keep) [rmse -45.17%, trip -23.31%]
- [x] exp3: add v_in (entry speed from the previous link) -> rmse=0.007192, trip_rmse=0.002335 (discard) [rmse -1.71%, trip +0.26%]
- [x] exp4: swap RandomForest for HistGradientBoosting (identical config to exp1, now with dke_per_mile present) -> rmse=0.007131, trip_rmse=0.002205 (keep) [rmse -2.54%, trip -5.32%]
- [x] exp5: raise max_leaf_nodes 31 -> 255 -> rmse=0.007163, trip_rmse=0.002191 (discard) [rmse +0.45%, trip -0.63%]
- [x] exp6: add dke_in_link, the entry-half of the kinetic-energy change ((v^2 - v_in^2)/miles) -> rmse=0.006653, trip_rmse=0.002147 (keep) [rmse -6.70%, trip -2.63%]
- [x] exp7: add gap_seconds, the idle time between the previous link ending and this one starting -> rmse=0.006336, trip_rmse=0.002099 (keep) [rmse -4.76%, trip -2.24%]
- [x] exp8: add prev_miles, the length of the previous link -> rmse=0.005837, trip_rmse=0.002065 (keep) [rmse -7.88%, trip -1.62%]
- [x] exp9: add next_miles, the length of the next link -> rmse=0.005748, trip_rmse=0.002064 (keep) [rmse -1.52%, trip -0.05%]
- [x] exp10: add next_gap_seconds, the idle time after the link -> rmse=0.005625, trip_rmse=0.002062 (keep) [rmse -2.14%, trip -0.10%]
- [x] exp11: raise max_iter 300 -> 2000 -> rmse=0.005360, trip_rmse=0.001985 (keep) [rmse -4.71%, trip -3.73%]
- [x] exp12: raise max_iter 2000 -> 8000 -> rmse=0.005379, trip_rmse=0.001975 (discard) [rmse +0.35%, trip -0.50%]
- [x] exp13: add prev2_speed, the speed two links back -> rmse=0.005114, trip_rmse=0.001958 (keep) [rmse -4.59%, trip -1.36%]
- [x] exp14: add dv_out, the exit speed change on a raw scale rather than per mile -> rmse=0.005064, trip_rmse=0.001951 (keep) [rmse -0.98%, trip -0.36%]
- [x] exp15: replace the GBDT with a dilated 1-D CNN reading the whole journey link sequence (same 11 features, +/-30 link receptive field, 431s on GPU) -> rmse=0.006440, trip_rmse=0.002078 (discard) [rmse +27.17%, trip +6.51%]

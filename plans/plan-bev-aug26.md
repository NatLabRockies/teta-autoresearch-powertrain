# Experiment Plan: bev-aug26

## Starting point

- Starting commit: `409009f` — `main`'s HEAD, which carries the best known `train.py`
- Best known metrics: none — this is the first session in this tree (`learnings.md` is empty)
- Baseline model: `RandomForestRegressor(n_estimators=20, max_depth=10, min_samples_split=10)`
  on `speed_mph, grade_percent, miles`

## Learnings consulted

`learnings.md` is empty — there are no prior sessions to carry forward. Everything below comes
from `domain.md` and a first look at the data, and it is this session's job to write the first
entries.

Data reconnaissance (1.64M links, 15,247 journeys, median 72 links/trip):

- `energy_rate_gge` is 25% negative (regenerative braking), median 0.0067, range roughly
  ±0.30 — asymmetric and heavy-tailed exactly as `domain.md` warns. Squared-error metrics will
  be dominated by the tails, so tail behavior is where the wins are.
- `miles` is both a model feature and the trip-aggregation weight in `trip_rmse`. Link RMSE and
  trip RMSE can therefore disagree: a rate error on a long link costs far more at trip level.
  Expect the two metrics to move apart, and read the split as information.
- `time_seconds`, `link_end_time`, `n_points`, `energy_gge` exist in the parquet but are NOT
  available at inference — they are outcomes, not inputs. Only `speed_mph`, `grade_percent`,
  `miles`, and `geometry` are legitimate.
- Hardware: 48 cores and one Tesla P100 (12GB) with `pytorch-gpu` in the env, so GPU neural nets
  fit inside the 10-minute budget.

## Session goals

1. Establish how much accuracy is available from the three named focus areas — geometry-derived
   features, one-link lookback, and GPU neural nets — and find a configuration that
   Pareto-dominates the RandomForest baseline on both `rmse` and `trip_rmse`.
2. Write the first real `learnings.md` for this tree: which feature families pay, which are dead
   ends, and where link RMSE and trip RMSE diverge.

## Planned experiments (rough order)

1. Baseline run — `train.py` as-is, exp0.
2. Cheap capacity check on the incumbent family: the baseline forest is small (20 trees,
   depth 10) and is very likely under-fit. One atomic change at a time (depth, then estimators)
   to find out whether the baseline number is a model-capacity artifact rather than a data
   ceiling. This is scoped as calibration, not as the session's direction.
3. **One-link lookback** — `domain.md` names acceleration as the dominant unmeasured driver of
   energy use and permits exactly one link of lookback. Add previous-link features one at a
   time: prev `speed_mph`, then the speed delta (a direct acceleration proxy), then prev
   `grade_percent`. First link of each trip needs a defined fill; that choice is itself a
   sub-experiment.
4. **Geometry-derived features** — decode the WKB linestrings. Candidates, added one at a time:
   sinuosity (path length ÷ endpoint distance), total absolute turn angle, max turn angle,
   point count normalized by distance. All are precomputable per link, so they cost nothing at
   inference time in the shortest-path search.
5. **Neural nets on GPU** — an MLP over the feature set that wins from (3) and (4), then
   consider a sequence model over the one-link lookback. Weigh accuracy against `domain.md`'s
   explicit inference-cost objective.
6. Combine the survivors; re-test interactions, since features that fail alone can pay together.

## Constraints / focus areas

Operator kickoff brief for this session, focus areas as selected at setup:

- **Geometry-derived features** — the `geometry` column is untouched by the baseline.
- **One-link lookback** — previous link only, never the next link.
- **Neural nets on GPU** — use the P100.
- Explicitly *not* a focus: GBM / tree-model tuning as a direction. Tree work is allowed only as
  cheap calibration of the baseline (item 2), not as where the session spends its budget.

Binding constraints from `domain.md`:

- Keep only on **Pareto dominance over both** `rmse` and `trip_rmse`. A win on one and a loss on
  the other is a `discard`.
- No filtering, reweighting, or dropping of rows before the split — including the regen tail.
- No link-position feature; position within a trajectory is unknown at inference.
- One-link lookback maximum, backward only.
- Inference cost is a real competing objective: energy is evaluated at every link traversal in a
  shortest-path search. Prefer features precomputable per link and models that are cheap to
  apply.
- Only `speed_mph`, `grade_percent`, `miles`, `geometry`, and things derived from them are legal
  inputs. Anything else in the parquet is an outcome of the simulation.

Open hypotheses to test:

- H1: The baseline forest is capacity-limited, not data-limited.
- H2: A previous-link speed delta is the single most valuable addition, because it is the only
  available proxy for acceleration.
- H3: Sinuosity and turn angle carry the cornering/braking signal that a straight-line average
  speed hides, and should help most on the negative (regen) tail.
- H4: `rmse` and `trip_rmse` will diverge under `miles`-correlated changes; a model that is
  better per link can be worse per trip if its errors correlate along a journey.

## Progress log

Updated after each experiment. Format:
`- [x] expN: description -> <metric>=<value>, <metric>=<value> (status)`

- [x] exp0: baseline RandomForest(20, depth 10) -> rmse=0.013345, trip_rmse=0.003037 (keep) — 7.3s
      of a 600s budget, so ~99% of the time budget is unused
- [x] exp1: max_depth 10 -> 20 -> rmse=0.013480, trip_rmse=0.003043 (discard) — both regressed
- [x] exp2: n_estimators 20 -> 100 -> rmse=0.013341, trip_rmse=0.003036 (discard) — 0.03% gain for
      5x inference cost, rejected on the cost criterion; calibration says the model is
      feature-limited, not model-limited
- [x] exp3: add prev_speed_mph lookback -> rmse=0.009164, trip_rmse=0.002990 (keep) — -31.3% link
- [x] exp4: add explicit speed_delta -> rmse=0.008293, trip_rmse=0.002819 (keep) — -9.5% / -5.7%
- [x] exp5: add ke_delta_per_mile -> rmse=0.008210, trip_rmse=0.002789 (keep) — -1.0% / -1.1%
- [x] exp6: add sinuosity -> rmse=0.008093, trip_rmse=0.002757 (keep) — -1.4% / -1.1%
- [x] exp7: add total_turn_degrees -> rmse=0.008088, trip_rmse=0.002756 (discard) — 0.05% for 25
      lines, subsumed by sinuosity
- [x] exp8: add prev_grade_percent -> rmse=0.008080, trip_rmse=0.002757 (keep) — -0.16% / flat
- [x] exp9: RandomForest -> GPU MLP 256x256 -> rmse=0.007811, trip_rmse=0.002419 (keep) — -3.3%
      link but -12.3% trip, the session's key structural finding
- [x] exp10: MAX_EPOCHS 200 -> 600 -> rmse=0.007837, trip_rmse=0.002453 (discard) — overfits
- checkpoint: learnings.md written at exp10. Best dd8f7e7, -41.5% link / -20.3% trip vs baseline
- [x] exp11: cosine LR anneal to zero -> rmse=0.007736, trip_rmse=0.002350 (keep) — -1.0% / -2.9%,
      first experiment designed from the smoothness rule and it landed as predicted
- [x] exp12: MAX_EPOCHS 200 -> 400 under the schedule -> rmse=0.007751, trip_rmse=0.002357
      (discard) — 200 epochs is a real optimum, not an oscillation artifact
- [x] exp13: drop speed_delta -> rmse=0.007730, trip_rmse=0.002344 (keep) — simplification win
- [x] exp14: drop ke_delta_per_mile -> rmse=0.007739, trip_rmse=0.002353 (discard) — nonlinear
      physics term is not redundant, unlike the linear one
- [x] exp15: add hours_per_mile (1/speed) -> rmse=0.007729, trip_rmse=0.002344 (discard) — tie
- [x] exp16: HIDDEN 256 -> 512 -> rmse=0.007735, trip_rmse=0.002358 (discard)
- [x] exp17: HIDDEN 256 -> 128 -> rmse=0.007741, trip_rmse=0.002348 (discard) — 256 is optimal;
      with exp12 and exp16 this establishes the model is information-limited
- [x] exp18: add junction_turn_degrees -> rmse=0.007399, trip_rmse=0.002308 (keep) — -4.3% / -1.5%
- [x] exp19: add prev_miles -> rmse=0.006954, trip_rmse=0.002266 (keep) — -6.0% / -1.8%, the
      session's surprise; reads as a reliability weight on prev_speed_mph
- [x] exp20: add prev_hours -> rmse=0.006952, trip_rmse=0.002266 (discard) — tie
- checkpoint: learnings.md updated at exp20. Best 8660022, -47.9% link / -25.4% trip vs baseline
- [x] exp21: signed junction turn -> rmse=0.006956, trip_rmse=0.002262 (discard) — split
- [x] exp22: miles^2-weighted loss -> rmse=0.007064, trip_rmse=0.002276 (discard) — both worse,
      including the metric it targeted; link errors cancel along a trip rather than adding
- [x] exp23: add prev_sinuosity -> rmse=0.006859, trip_rmse=0.002252 (keep) — -1.4% / -0.6%
- [x] exp24: ReLU -> SiLU -> rmse=0.006884, trip_rmse=0.002256 (discard)
- [x] exp25: third hidden layer -> rmse=0.006951, trip_rmse=0.002270 (discard)
- diagnostic: residuals binned by feature. Zero bias in feature space; 46% of squared error
      below 23 mph, 43% in the two shortest mile deciles
- [x] exp26: 3-model ensemble -> rmse=0.006822, trip_rmse=0.002235 (discard) — Pareto-better but
      3x inference cost, and the small gain shows fitting variance is nearly exhausted
- [x] exp27/28: batch 4096 / 16384 -> both split (discard) — 8192 is a true optimum
- [x] exp29: lr 2e-3 -> rmse=0.006877, trip_rmse=0.002250 (discard) — split
- [x] exp30: add vertices_per_mile -> rmse=0.006837, trip_rmse=0.002248 (keep) — -0.3% / -0.2%
- [x] exp31: drop prev_grade_percent -> rmse=0.006857 (discard) — now worth 2x what it was
      under the forest
- [x] exp32: corner_energy_per_mile -> split (discard) — copying a good feature's form is not enough
- [x] exp33: signed-log ke_delta -> rmse=0.006850 (discard) — the heavy tail is signal
- [x] exp34: linear skip connection -> split (discard)
- [x] exp35: link_bend_degrees -> exact tie (discard) — intra-link shape closed for good
- checkpoint: learnings.md updated at exp35
- [x] exp36: AdamW wd=1e-4 -> rmse=0.006837, trip_rmse=0.002242 (keep) — rmse tied, trip -0.27%
- [x] exp37: wd 1e-3 -> split (discard)
- [x] exp38: dropout 0.1 -> rmse -0.31% but trip_rmse +3.26% (discard) — train/predict mismatch
      is systematic bias, the one thing trip_rmse cannot absorb
- [x] exp39: drop sinuosity -> rmse=0.006999 (discard) — complements vertices_per_mile, not
      redundant with it
- [x] exp40: LayerNorm -> split (discard)
- checkpoint: usage and transcripts captured through exp40. Best 653f2d8
- [x] exp41: own-speed fill for first link -> rmse=0.006838 (discard) — from-rest is right
- [x] exp42: MAX_EPOCHS 300 under weight decay -> rmse=0.006856 (discard)
- [x] exp43: is_decelerating flag -> rmse=0.006850 (discard) — the regen kink already sits at
      zero of ke_delta_per_mile, and a kink at zero is what a ReLU *is*
- [x] exp44: centroid lat/lon diagnostic -> both worse (discard) — no exploitable
      region-specific signal remains; the model is genuinely physics-driven
- [x] exp45: 2x128 ensemble replacing 1x256 -> rmse=0.006836, trip_rmse=0.002242 (keep) —
      accuracy held at HALF the inference cost, by composing two previously rejected results
- [x] exp46: 3x96 -> (discard); exp47: 2x160 -> Pareto-better but +54% cost (discard);
      exp48: 2x112 -> (discard). Per-member width floor is 128; 2x128 is the knee
- [x] exp49: MAX_EPOCHS 280 for narrow members -> rmse=0.006828, trip_rmse=0.002239 (keep)
- [x] exp50: MAX_EPOCHS 320 -> rmse=0.006823, trip_rmse=0.002239 (keep)

## Outcome

Session limit reached at 50 experiments (~2h45m wall clock, well inside the 8h limit).
**13 keeps.** Final: `e315384`, rmse 0.006823, trip_rmse 0.002239 — **-48.9% link RMSE and
-26.3% trip RMSE** against the scaffold baseline, at roughly half the inference cost of the
mid-session best. Full findings in `learnings.md`.

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

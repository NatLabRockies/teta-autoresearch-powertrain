# Learnings

Accumulated across sessions. Sessions so far: `bev-aug25`.

## The problem, as reconstructed from the harness and the data

There is no `domain.md` in this tree, so this section is what the data and `harness.py` say.
Treat it as findings, not as a spec.

- `data/processed/2017_Chevy_Bolt.parquet`: 1,638,466 rows, 15,247 journeys, median 72 links per
  journey. One row = one road link traversed on one trip of a battery-electric vehicle.
- Target `energy_rate_gge` (GGE per mile) is exactly `energy_gge / miles`, so **`energy_gge` is
  not a usable feature**. Mean 0.0061, sd 0.0142, range -0.294..0.299; negatives are regen.
- `grade_percent == grade_dec * 100` exactly — they are one feature, not two.
- `speed_mph` is **not** `miles / time_seconds`: it is a point-average, so it disagrees with the
  distance/duration average (corr 0.994). `n_points ~= time_seconds` (1 Hz sampling).
- `road_id` is near-unique (1.36M distinct over 1.64M rows) — useless as a categorical.
- The split is **row-level, not grouped by journey** (seed 42, 20% test), so every journey has
  links on both sides and `trip_rmse` is a sum over only that journey's test links.
- The two metrics disagree by construction: `rmse` weights every link equally, `trip_rmse` is
  effectively mileage-weighted and lets within-trip errors cancel.

## The one big finding so far

**Link energy rate is dominated by vehicle dynamics, not by steady-state speed and grade.** The
scaffold's three instantaneous features (`speed_mph`, `grade_percent`, `miles`) explain only
R^2 ~ 0.115. Features derived from the *neighbouring links of the same journey* took link rmse from
0.013345 to 0.005625 (-58%) in nine experiments. Nothing else has come close.

Ranked by what each was worth when it was added (each is one experiment, in order):

| feature | what it is | rmse | trip_rmse |
|---|---|---|---|
| `dke_per_mile` | `(v_next^2 - v_prev^2) / miles` — the acceleration term | **-45.2%** | -23.3% |
| `prev_miles` | length of the previous link | -7.9% | -1.6% |
| `dke_in_link` | `(v^2 - v_prev^2) / miles` — the entry half of the above | -6.7% | -2.6% |
| `gap_seconds` | idle time *before* the link (i.e. the vehicle stopped) | -4.8% | -2.2% |
| `next_gap_seconds` | idle time *after* the link | -2.1% | -0.1% |
| `next_miles` | length of the next link | -1.5% | -0.05% |

Fill the missing neighbour at each end of a trip with **speed 0, not NaN** — a trip starts and
ends at rest, so that is the physically correct value rather than an imputation.

## Rules of thumb this session has earned

1. **Order matters: features before learners.** Swapping RandomForest for HistGradientBoosting was
   a *dead tie* on the 3-feature scaffold (exp1) and worth -2.5% / -5.3% on the same day once
   `dke_per_mile` was present (exp4). Capacity is worthless until the features carry signal, so
   re-run model/hyperparameter experiments after every big feature win, never before.
2. **Entry beats exit, by about 2x.** `gap_seconds` -4.8% vs `next_gap_seconds` -2.1%;
   `prev_miles` -7.9% vs `next_miles` -1.5%. Energy spent accelerating *into* a link is charged to
   that link; braking *out* of it is partly recovered by regen and partly charged to the next one.
3. **Features that condition an existing feature beat their isolated estimates.** `gap_seconds`
   (-4.8% actual vs -3.6% probed) and `prev_miles` (-7.9% vs -3.6%) both roughly doubled, because
   they tell the model when to trust `dke_in_link`'s assumption that the speed change happened on
   this link. A one-at-a-time feature probe systematically *under*-estimates such features.
4. **Wider trees trade the two metrics against each other.** `max_leaf_nodes` 31 -> 255 made rmse
   worse (+0.45%) and trip_rmse better (-0.63%). Deeper trees fit the per-link noise — a 0.02-mile
   link's rate is a ratio with a tiny denominator — which the mileage-weighted trip metric averages
   away but the equally-weighted link metric does not.

## Dead ends

- `speed_ratio` (= distance-average / point-average speed), `avg_speed`, `time_seconds`,
  `n_points`, `inv_miles`, `grade_x_speed`: all within +/-0.2% of zero. The dispersion-proxy idea
  that motivated `speed_ratio` did not survive contact with the data.
- `prev_grade` / `next_grade`, `grade_mean5`: ~-0.3%. Grade context is nearly worthless next to
  speed context.
- `max_leaf_nodes = 255` (see rule 4).

## Open hypotheses

- **`trip_rmse` has stalled** at ~0.00206 for four experiments while rmse keeps falling. It needs
  its own idea — mileage-weighted training, predicting `energy_gge` instead of the rate, or a
  model that sees the whole journey.
- **A sequence model over a journey's links** (GRU / 1-D CNN / attention) is the general form of
  every feature that has worked so far; hand-crafted +/-1 neighbours are a truncated version of it.
- `v_in` on its own was -1.7% rmse but +0.26% trip_rmse under a low-capacity RandomForest (exp3,
  discarded on the rule). Worth re-testing now that the model is a GBDT.
- Untouched: `geometry` (WKB linestrings — sinuosity, turning, absolute position), trip-level
  aggregates (`trip_miles`, `trip_n_links` each probed ~-1.8% on trip_rmse alone), the GPU.

## Best known configuration

Commit `7b08ce5` (exp10). `HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1,
max_leaf_nodes=31)` on nine features. **rmse 0.005625, trip_rmse 0.002062**, 10.1s of a 600s
budget — the time budget has not been a constraint even once.

# Learnings

Accumulated across sessions. Sessions so far: `bev-aug26`.

## Best known configuration

As of exp10 of `bev-aug26`, commit `dd8f7e7`:

- **Model**: MLP, 8 → 256 → 256 → 1, ReLU, Adam lr 1e-3, batch 8192, 200 epochs, MSE on a
  standardized target, on GPU.
- **Features**: `speed_mph`, `grade_percent`, `miles`, `prev_speed_mph`, `speed_delta`,
  `ke_delta_per_mile`, `sinuosity`, `prev_grade_percent`
- **Metrics**: `rmse` 0.007811, `trip_rmse` 0.002419
- Against the original scaffold baseline (`rmse` 0.013345, `trip_rmse` 0.003037) that is
  **−41.5% link RMSE and −20.3% trip RMSE**.
- Runtime 145s of the 600s budget.

## What works

1. **The one-link lookback is the single biggest lever.** Adding `prev_speed_mph` alone cut link
   RMSE by 31% (exp3). `domain.md` is right that acceleration is the dominant unmeasured driver
   of energy use, and the previous link's speed is a good enough proxy to recover most of it.
2. **Representation matters as much as information, for trees.** `speed_delta` bought a further
   9.5% (exp4) even though both `speed_mph` and `prev_speed_mph` were already present. An
   axis-aligned splitter cannot express a difference of two columns, so differences must be
   handed to it precomputed. Expect this to matter much less for an MLP, which can form linear
   combinations itself — worth re-testing whether `speed_delta` still earns its place now that
   the model is a network.
3. **Physics-shaped features beat raw ones, slightly.** `ke_delta_per_mile` = (v² − v_prev²)/2d
   carries the target's own units (energy per mile) and added ~1% on both metrics (exp5) on top
   of the raw delta.
4. **Geometry carries real signal.** `sinuosity` (path length ÷ straight-line endpoint distance)
   gave ~1.3% on both metrics (exp6), and it is a static road attribute so it is free at
   inference.
5. **Smoothness is what trip RMSE wants.** Swapping the forest for an MLP moved link RMSE 3.3%
   but trip RMSE 12.3% (exp9). This is the session's most useful structural insight — see below.

## What does not work

- **More forest capacity.** Depth 10 → 20 regressed both metrics (exp1). More trees (20 → 100)
  gained 0.03% for 5× the inference cost (exp2). The 3-feature baseline was *feature*-limited,
  not model-limited; an unused time budget is not evidence of under-capacity.
- **Intra-link turn angle.** Summed absolute heading change across interior vertices moved both
  metrics by 0.05% for ~25 lines of spherical math (exp7). It is almost entirely subsumed by
  sinuosity. Links average only **3.7 vertices**, so there is very little independent shape
  signal available — treat that as a ceiling on the whole intra-link geometry family.
- **Training the MLP longer.** 200 → 600 epochs regressed both metrics (exp10). 200 epochs is
  at or slightly past the optimum for this configuration.

## Structural insight: why the two metrics diverge

`trip_rmse` sums `rate × miles` over a journey, so what matters at trip level is not the size of
each link error but whether errors **cancel or accumulate** along the journey.

- Gains that come from fixing *random* per-link error show up in `rmse` and largely vanish in
  `trip_rmse` — exp3 (−31% link, −1.5% trip) and exp8 (−0.16% link, 0% trip) are both this.
- Gains that come from removing *systematic* bias show up disproportionately in `trip_rmse` —
  exp9 (−3.3% link, −12.3% trip) is this. A tree gives every row in a leaf the same value, so
  its per-leaf bias is systematic and adds up over ~72 links (the median trip); a smooth fit
  spreads error more randomly and lets it cancel.
- The reverse also holds: overtraining (exp10) hurt `trip_rmse` 4× more than `rmse`, because it
  makes the fit locally bumpy.

**Practical rule for this domain: to move `trip_rmse`, make the model smoother or less biased;
to move `rmse`, give it more information.** Since a keep requires Pareto dominance on both, the
most productive experiments are the ones that do one without costing the other.

## Open hypotheses

- Does `speed_delta` still earn its place under an MLP, which can form the difference itself?
  Removing it would be a simplification win if metrics hold (see "What works" #2).
- Network shape and size are barely explored: only 256×256 has been tried. Smaller is also
  directly valuable — `domain.md` names inference cost as a competing objective, and the MLP is
  ~68k FLOPs per link against a few hundred for the forest.
- **Junction turn angle** — the heading change *between* the previous link and the current one —
  is untested and is a genuinely different signal from the intra-link turn angle that failed in
  exp7. Intersection turns are where the hard braking is.
- Loss shaping is untouched. MSE on the standardized target optimizes link error directly;
  nothing yet targets the trip objective or the heavy regen tail.
- Every result so far is deterministic (fixed `random_state`, fixed split), so sub-1% deltas are
  real and reproducible rather than noise. Do not dismiss small movements as run-to-run variance.

## Protocol notes

- WKB decoding of all 1.64M geometries costs ~19s with `shapely.from_wkb` on the whole array —
  affordable, but decode once and derive every geometry feature from that single pass.
- The parquet contains `time_seconds`, `link_end_time`, `n_points` and `energy_gge`, none of
  which exist at inference. Only `speed_mph`, `grade_percent`, `miles`, `geometry` and things
  derived from them are legal inputs.

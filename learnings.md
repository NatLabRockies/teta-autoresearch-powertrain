# Learnings

Accumulated across sessions. Sessions so far: `bev-aug26`.

## Best known configuration

As of exp20 of `bev-aug26`, commit `8660022`:

- **Model**: MLP, 9 → 256 → 256 → 1, ReLU, Adam lr 1e-3 cosine-annealed to zero, batch 8192,
  200 epochs, MSE on a standardized target, on GPU.
- **Features**: `speed_mph`, `grade_percent`, `miles`, `prev_speed_mph`, `ke_delta_per_mile`,
  `sinuosity`, `junction_turn_degrees`, `prev_grade_percent`, `prev_miles`
- **Metrics**: `rmse` 0.006954, `trip_rmse` 0.002266
- Against the original scaffold baseline (`rmse` 0.013345, `trip_rmse` 0.003037) that is
  **−47.9% link RMSE and −25.4% trip RMSE**.
- Runtime 116s of the 600s budget.

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
6. **Cosine-annealing the learning rate to zero** gave −1.0% link / −2.9% trip (exp11). It was
   the first experiment *designed from* the smoothness rule, and the metric split came out in
   the predicted direction and ratio. Optimizer settling is cheap trip RMSE.
7. **The turn *between* links is real signal** — `junction_turn_degrees`, the heading change
   from the previous link's exit into this link's entry, gave −4.3% link / −1.5% trip (exp18).
   Note this is the opposite result from the intra-link turn angle that failed in exp7.
8. **`prev_miles` was the surprise of the session** (−6.0% link / −1.8% trip, exp19), a bigger
   win than the junction turn. The best reading is not "urban density" but that it tells the
   model *how much to trust* `prev_speed_mph`: an average speed over a long previous link is a
   poor estimate of the speed at its end, over a short one a good one. **Pair every lookback
   value with its own reliability context.**

## What does not work

- **More forest capacity.** Depth 10 → 20 regressed both metrics (exp1). More trees (20 → 100)
  gained 0.03% for 5× the inference cost (exp2). The 3-feature baseline was *feature*-limited,
  not model-limited; an unused time budget is not evidence of under-capacity.
- **Intra-link turn angle.** Summed absolute heading change across interior vertices moved both
  metrics by 0.05% for ~25 lines of spherical math (exp7). It is almost entirely subsumed by
  sinuosity. Links average only **3.7 vertices**, so there is very little independent shape
  signal available — treat that as a ceiling on the whole intra-link geometry family.
- **Training the MLP longer.** 200 → 600 epochs regressed both metrics (exp10), and still
  regressed at 400 epochs even after cosine annealing removed the oscillation that was the
  suspected cause (exp12). 200 epochs is a genuine optimum. Closed direction.
- **Network width.** 512 (exp16) and 128 (exp17) both regressed. 256 is a real optimum and
  width is settled. Worth recording that 128 costs only +0.15% on both metrics for roughly a
  quarter of the hidden-layer FLOPs — that is the fallback operating point if inference cost
  ever binds in RouteE Compass.
- **Single-variable transforms and two-input ratios.** `hours_per_mile` = 1/speed (exp15) and
  `prev_hours` = prev_miles/prev_speed (exp20) both tied exactly. See the hand-engineering rule
  below — these are a closed direction.

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

## When hand-engineering a feature pays

Three experiments pin this down precisely, and it is the most transferable rule the session
produced:

- `speed_delta` (= speed − prev_speed) was worth 9.5% **to a forest** (exp4) and became
  removable dead weight under an MLP (exp13, a simplification win). Axis-aligned splitters
  cannot express a difference of two columns; a linear layer forms one for free.
- `hours_per_mile` (= 1/speed, exp15) and `prev_hours` (= prev_miles/prev_speed, exp20) each
  tied to six decimal places.
- `ke_delta_per_mile` (= (v² − v_prev²)/2d) is **not** removable — dropping it regressed both
  metrics (exp14).

**The rule: hand-engineer a feature only when it is hard for the network to *approximate*, not
merely when it is nonlinear.** A one-dimensional warp of a single input, or a ratio of two
inputs already present, is something 256 ReLU units reproduce on their own. A term that divides
a squared difference by a *third* input is not. Note this rule is model-family dependent — the
whole first group is valuable again the moment the model goes back to being a tree.

## What is actually limiting the model

Training time (exp12), width up (exp16), and width down (exp17) were all ruled out in
succession. The model is **information-limited, not model-limited**: the same network that
could not use more capacity immediately used more information when `junction_turn_degrees` and
`prev_miles` arrived (exp18, exp19, −10% link RMSE between them). Architecture tuning is a
closed direction for now; features are the frontier.

## Open hypotheses

- Following the exp19 reliability insight: pair the other lookback values with reliability
  context, or give the *current* link the same treatment.
- Signed rather than absolute junction turn — left turns cross oncoming traffic and cost more
  than right turns, so the sign may carry real asymmetry that `abs()` currently discards.
- Loss shaping is untouched. MSE on the standardized target optimizes link error directly;
  nothing yet targets the trip objective or the heavy regen tail. Note the trip metric weights
  a link by `miles`, which the link loss does not.
- Depth (3+ hidden layers) is untested, though width being settled makes it a weak prospect.
- Every result so far is deterministic (fixed `random_state`, fixed split), so sub-1% deltas are
  real and reproducible rather than noise. Do not dismiss small movements as run-to-run variance.

## Protocol notes

- WKB decoding of all 1.64M geometries costs ~19s with `shapely.from_wkb` on the whole array —
  affordable, but decode once and derive every geometry feature from that single pass.
- The parquet contains `time_seconds`, `link_end_time`, `n_points` and `energy_gge`, none of
  which exist at inference. Only `speed_mph`, `grade_percent`, `miles`, `geometry` and things
  derived from them are legal inputs.

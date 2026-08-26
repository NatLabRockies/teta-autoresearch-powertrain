# Learnings

Accumulated across sessions. Sessions so far: `bev-aug26`.

## Best known configuration

As of exp35 of `bev-aug26`, commit `9ff5e20`:

- **Model**: MLP, 11 → 256 → 256 → 1, ReLU, Adam lr 1e-3 cosine-annealed to zero, batch 8192,
  200 epochs, MSE on a standardized target, on GPU.
- **Features**: `speed_mph`, `grade_percent`, `miles`, `prev_speed_mph`, `ke_delta_per_mile`,
  `sinuosity`, `junction_turn_degrees`, `prev_grade_percent`, `prev_miles`, `prev_sinuosity`,
  `vertices_per_mile`
- **Metrics**: `rmse` 0.006837, `trip_rmse` 0.002248
- Against the original scaffold baseline (`rmse` 0.013345, `trip_rmse` 0.003037) that is
  **−48.8% link RMSE and −26.0% trip RMSE**.
- Runtime 132s of the 600s budget.

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
- **Network depth** (exp25) and a **linear skip connection** (exp34) both failed, as did
  **SiLU** (exp24). With width and length already closed, the architecture is settled at
  2×256 ReLU against size *and* inductive-bias changes alike.
- **Every optimizer knob.** Batch 4096 (exp27), batch 16384 (exp28) and lr 2e-3 (exp29) all
  produced the same signature — link RMSE slightly worse, trip RMSE a sliver better. Both batch
  directions failing means 8192 is a true optimum. That the trip metric ticked down in all
  three says movements below ~0.2% on `trip_rmse` are at the metric's resolution floor and
  should not be read as signal.
- **Intra-link shape is closed for good.** Summed interior turn (exp7) and net entry-to-exit
  bend (exp35) both tied. At 3.7 vertices per link, sinuosity determines the rest.
- **Loss shaping by `miles²`** (exp22) made *both* metrics worse, including the one it targeted.
  See below. This also rules out predicting `energy_gge` instead of the rate without testing it,
  since MSE on `rate × miles` is exactly that weighted loss.
- **Compressing `ke_delta_per_mile` with a signed log** (exp33) regressed both metrics. The
  heavy tail is signal, not a conditioning defect: a large speed change over a very short link
  genuinely is a large energy rate. With a heavy-tailed target, matching the target's own scale
  behaviour beats making the input well-conditioned.
- **Turn handedness** (exp21). Signed rather than absolute junction turn split the metrics. The
  data is simulated over real drive-cycle traces, so any left-turn waiting is already in the
  trace's speed profile.
- **Copying a successful feature's functional form** (exp32). `corner_energy_per_mile` used the
  same three-input nonlinear shape as `ke_delta_per_mile` and gained nothing. The form is not
  what made that feature work; being the dominant physics of the target is.

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

**Read "smoothness" as systematic bias over a *region* of feature space, not as analytic
smoothness of the fit.** Swapping ReLU for SiLU (exp24) made both metrics worse and moved trip
*less* than link — the opposite of the predicted signature. ReLU kinks are far below the scale
at which regional bias operates.

A related caution from exp22: the natural way to target `trip_rmse` is to weight the training
loss by `miles²`, since a link's share of trip error looks like it should go as `miles²`. It
made both metrics worse. That decomposition assumes link errors within a trip are
**independent**, and they are not — they largely cancel. Trip error is driven by systematic
bias, not by summed variance, and reweighting shrinks the effective sample without touching the
bias.

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

## Where the remaining error actually is (diagnostic, after exp25)

Trained the best config and binned the test residuals. Two results, both load-bearing:

1. **Bias in feature space is essentially zero** — between 1e-5 and 1e-4 in every speed, grade,
   miles and turn decile, against a target standard deviation of 0.014. The model is already
   well calibrated, which is why every architecture and optimizer change after exp13 failed.
   There is no systematic regional bias left to remove.
2. **The error is concentrated in urban conditions.** The two slowest speed deciles (< 23 mph)
   hold **46%** of total squared error, the two shortest `miles` deciles hold **43%**, and the
   sharpest junction-turn quintile holds **30%**. These are exactly the conditions where a
   link-average speed hides the most about the real speed profile.

**Caution for whoever reads this next:** binning residuals by the *target* shows a large
apparent bias in the top and bottom deciles. That is a statistical artifact — conditioning on an
extreme `y` selects on noise, so regression to the mean appears even for an optimal predictor.
Do not chase it. Bin by *features*, never by the target.

The exp26 ensemble is the confirmation: averaging three independent fits bought only 0.6%. If
fitting variance were a meaningful share of what is left, it would have bought far more. **The
model is close to the information ceiling available under one-link-lookback inference.**

## What is actually limiting the model

Training time (exp12), width up (exp16), and width down (exp17) were all ruled out in
succession. The model is **information-limited, not model-limited**: the same network that
could not use more capacity immediately used more information when `junction_turn_degrees` and
`prev_miles` arrived (exp18, exp19, −10% link RMSE between them). Architecture tuning is a
closed direction for now; features are the frontier.

## Open hypotheses

- **The lookback family is complete and was the richest seam in the session.** `speed`, `grade`,
  `miles` and `sinuosity` are all shifted, and the junction turn consumes the previous link's
  exit heading. Under a strict one-link limit there is nothing left to shift. Any future gain
  has to come from a new source.
- **Re-test marginal keeps after a model-family change — value moves both ways.** `speed_delta`
  was worth 9.5% to the forest and became removable (exp13); `prev_grade_percent` was worth
  0.16% to the forest and is now worth roughly twice that (exp31). Do not assume a
  family change only prunes.
- Regularization proper (dropout, weight decay) is still untested.
- The 46% of error below 23 mph is the standing target. Nothing tried so far distinguishes rows
  *within* that regime; a feature that does is where a real gain would come from.
- Every result so far is deterministic (fixed `random_state`, fixed split), so sub-1% deltas are
  real and reproducible rather than noise. Do not dismiss small movements as run-to-run variance.

## Protocol notes

- WKB decoding of all 1.64M geometries costs ~19s with `shapely.from_wkb` on the whole array —
  affordable, but decode once and derive every geometry feature from that single pass.
- The parquet contains `time_seconds`, `link_end_time`, `n_points` and `energy_gge`, none of
  which exist at inference. Only `speed_mph`, `grade_percent`, `miles`, `geometry` and things
  derived from them are legal inputs.

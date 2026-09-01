# Experiment Plan: bev-aug31

## Starting point

- Starting commit: `a51dbb4` — `main`'s HEAD, the original scaffold (20-tree RandomForest on
  `speed_mph`, `grade_percent`, `miles`)
- Best known metrics: none — this is the first session in the `physics-bounded` tree.
  `learnings.md` is empty.

## Learnings consulted

`learnings.md` here is empty, so the learnings carried in come from the two sibling trees,
`../unguided` (`bev-aug25`) and `../domain-guided` (`bev-aug26`). Both optimized **accuracy only**
— neither had `physics.py` — so every number below is a prior about *accuracy*, and none of them
is evidence about physical plausibility. Treat them as ranked hypotheses, not as settled results.

- **Neighbour-link dynamics are the dominant accuracy lever, by a wide margin.** Unguided got
  −62% link RMSE from them, led by `dke_per_mile = (v_next² − v_prev²)/miles` at −45% on its own.
  Domain-guided, restricted to a one-link *backward* lookback, got −31% from `prev_speed_mph`
  alone. This tree has no lookback restriction (see brief), so the two-sided form is available.
- **Entry beats exit by 2–4×** (unguided): energy spent accelerating *into* a link is charged to
  it; braking out of it is partly regenerated and partly charged to the next link. Order the
  neighbour features accordingly.
- **Fill the missing neighbour at each trip end with speed 0, not NaN** — a trip starts and ends
  at rest, so that is the physically correct value.
- **Smoothness moves `trip_rmse`; information moves `rmse`** (domain-guided). Forest → MLP was
  −3.3% link but −12.3% trip, because a tree's per-leaf bias is systematic and accumulates over
  the ~72 links of a median trip. Since a keep needs Pareto dominance, prefer changes that do one
  without costing the other.
- **Hand-engineer a feature only when the model cannot cheaply approximate it** (domain-guided).
  Differences of two columns are free to an MLP and worth ~10% to a forest; a term dividing a
  squared difference by a *third* input (`ke_delta_per_mile`) is not free to either.
- **Geometry is worth ~1.3% at best and costs ~20s to decode** — real in domain-guided, ~0 in
  unguided. Low priority here, and the decode has to be paid once.
- **Dead ends already paid for, do not repeat**: per-road residual encoding; wide-context rolling
  windows and trip aggregates (signal is local); `speed_ratio`, `avg_speed`, `inv_miles`,
  `grade_x_speed`, `prev_grade`/`next_grade` as raw additions; a linear physics init that fits an
  unconditional least-squares line and boosts the residual (10% worse on both — regen makes the
  true slope regime-dependent, so there is no single line).
- **Columns that do not exist at inference**: `time_seconds`, `link_end_time`, `n_points`,
  `energy_gge`. `grade_percent == grade_dec * 100` exactly, so they are one feature.

## Session goals

1. **Reach `physics_pass = 1.0` first.** Under `domain.md` no amount of RMSE counts until the
   model stops breaking physical law. Expect the scaffold forest to fail: a tree is constant
   outside its training range and the sweep runs to 5 miles, ten times the longest training link.
2. **Then minimize `rmse` and `trip_rmse` inside the passing set**, using the sibling trees'
   feature findings — which have never been tested under a physics constraint.
3. Establish, for `learnings.md`, whether physical constraint costs accuracy or buys it.

## The central idea: a structurally feasible parametrization

The core hypothesis of this session, worked out before running anything, is that all nine checks
can be satisfied **by construction** rather than by tuning, if the model predicts the *link's
total energy* in the shape physics actually gives it:

```
E(v, g, d, …) = d · [ A + B·g₊ − C·g₋ ] + T
```

`A` is a per-mile road-load rate, `B` and `C` are per-mile-per-grade-percent climb and descent
slopes, `T` is a fixed per-link transient cost, and every one of them may be an arbitrary
function of any features. The prediction the harness scores is `E / d`.

This shape matters because the ceiling in `physics.py` decomposes the same way: `resistance` and
`accessory` are proportional to `d`, `transient` is not, and `potential` is proportional to both
`d` and `g`. So bounding each head separately bounds the total exactly.

With `k_pot = MASS·G·MI_TO_M/100/J_PER_GGE` (the potential energy of one grade-percent over one
mile), squashing the four heads into

| head | range | enforced by |
|---|---|---|
| `A` | `[0.5·res_pm(v), res_pm(v)/η + acc_pm(v)]` | `flat_energy_positive`, `energy_grows_with_distance`, `absolute_ceiling` |
| `B` | `[k_pot, k_pot/η]` | `climb_floor`, `absolute_ceiling` |
| `C` | `[0, η·k_pot]` | `monotone_in_grade`, `round_trip_convexity`, `regen_ceiling`, `absolute_floor` |
| `T` | `[0, transient(v)/η]` | `absolute_ceiling` |

satisfies all nine, algebraically, for **any** input — verified by hand across flat, climb and
descent for every check. That is the property that makes it worth building: it is not a clamp on
the output, it is a model family whose every member is physically legal, so physics stops being
something to chase and the whole rest of the session is free to optimize RMSE.

Note the one place this is *stricter* than the checks: pinning `B ≤ k_pot/η` says the marginal
cost of grade never exceeds `mgh` at peak drivetrain efficiency, while the actual ceiling has
`transient` and `accessory` slack a real climb may be using. If climbs come out underpredicted,
the fix is to let `B` spend that slack (`B·g ≤ k_pot·g/η + slack_pm`) rather than to abandon the
structure.

## Planned experiments (rough order)

1. **exp0** — baseline, unmodified scaffold. Establishes both the accuracy floor and *which*
   physics checks a plain forest breaks.
2. **exp1** — the structured parametrization above, heads driven by an MLP on the three scaffold
   features only. Same information as the baseline, different functional form. This is the
   session's pivot; everything after it assumes physics is solved and works on accuracy.
   - Fallback if it underfits badly: keep the structure, widen `B`'s ceiling as noted above.
3. **exp2+** — port the sibling trees' feature wins one at a time into the structured model, in
   the order their measured effect sizes suggest: `dke_per_mile`, then `prev_miles`,
   `dke_in_link`, `gap_seconds`, `prev2_speed`, then the exit-side features.
4. Then: architecture and optimizer (width, depth, epochs, LR schedule), which domain-guided
   found closed only *after* features were exhausted — "features before learners".
5. Then: whether any head can be simplified away (e.g. is `C` really needed, or does `B` suffice
   with a signed grade?) — simplification wins count.

## Constraints / focus areas

Operator brief for this session, recorded here as the only durable copy:

- This tree differs from `../unguided` and `../domain-guided` in **two** ways: real physical
  constraints are enforced, and **the model is not being built for RouteE Compass**. It may use
  **the context of all links in a trip** at inference, so the one-link-lookback limit from
  `domain-guided` and the inference-cost objective both do **not** apply here. Two-sided and
  whole-chain features are in scope.
- **Features only, no labels** (operator, asked directly): full bidirectional link context is
  fair game, but no feature may read `energy_gge` / `energy_rate_gge` from any link, including
  leave-one-out journey encodings. The unguided tree's "journey offset" is therefore **out of
  scope this session**, even though it was that tree's biggest `trip_rmse` lever — it works only
  because the harness splits rows rather than journeys, so it is a property of the exam.
- Still binding from `domain.md`: no filtering or reweighting of rows before the split; no
  link-position feature; no post-hoc clamping or clipping to buy `physics_pass`.
- The physics sweep gives each synthetic link its own single-link journey, so any trip-context
  feature must be derived inside the `predict` closure and will evaluate at its steady-state
  value there (neighbour = self, zero acceleration, no idle gap). Features must be *defined* so
  that this is meaningful, not merely non-crashing.

## Progress log

Updated after each experiment. Format:
`- [x] expN: description -> <metric>=<value>, <metric>=<value> (status)`

- [x] exp0: baseline scaffold RandomForest -> rmse=0.013345, trip_rmse=0.003037, physics_pass=0.0 (keep, 6/9 checks fail)
- [x] exp1: structured physics parametrization, 4 bounded heads -> rmse=0.013397, trip_rmse=0.003060, physics_pass=1.0 (keep, first physically legal model)
- [x] exp2: add prev_speed_mph -> rmse=0.010816, trip_rmse=0.003008, physics_pass=1.0 (keep, -19.3% rmse)
- [x] exp3: add next_speed_mph -> rmse=0.010305, trip_rmse=0.002871, physics_pass=1.0 (keep, -4.7%/-4.6%)
- [x] exp4: add prev_miles -> rmse=0.009984, trip_rmse=0.002880 (discard, trip_rmse +0.3%)
- [x] exp5: add prev2_speed_mph -> rmse=0.010262, trip_rmse=0.002873 (discard, near-tie)
- [x] exp6: climb term spends ceiling headroom -> rmse=0.010287, trip_rmse=0.002867 (keep, -0.17%/-0.14%)
- [x] exp7: signed transient (released kinetic energy) -> rmse=0.007597, trip_rmse=0.002439 (keep, -26.2%/-14.9%)
- [x] exp8: re-test prev_miles -> rmse=0.006932, trip_rmse=0.002315 (keep, -8.8%/-5.1%)
- [x] exp9: add next_miles -> rmse=0.006757, trip_rmse=0.002287 (keep, -2.5%/-1.2%)
- [x] exp10: release sized from the link own peak -> rmse=0.006757, trip_rmse=0.002277 (keep, tie/-0.44%)
- [x] exp11: re-test prev2_speed_mph -> rmse=0.006583, trip_rmse=0.002258 (keep, -2.6%/-0.8%)
- [x] exp12: add next2_speed_mph -> rmse=0.006488, trip_rmse=0.002238 (keep, -1.4%/-0.9%)
- [x] exp13: add prev_grade_percent -> rmse=0.006478, trip_rmse=0.002218 (keep, -0.15%/-0.89%)
- [x] exp14: add next_grade_percent -> rmse=0.006480, trip_rmse=0.002218 (discard, tie)
- [x] exp15: widen 128 -> 256 -> rmse=0.006530, trip_rmse=0.002204 (discard, metrics split)

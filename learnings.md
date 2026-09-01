# Learnings

Accumulated across sessions. Sessions so far: `bev-aug31` (in progress).

This tree differs from `../unguided` and `../domain-guided` in two ways, and both change what is
worth trying:

1. **`physics.py` is enforced.** A run with `physics_pass: 0.0` is a discard whatever its RMSE.
2. **The model is not for RouteE Compass.** It may use the whole trip's link chain at inference,
   so there is no one-link-lookback limit and inference cost is not an objective.

## The result that matters: physics is a model family, not a penalty

The scaffold forest fails **six of nine checks** — `climb_floor` on 81% of cases, an implied
drivetrain efficiency of **1.29** (it bills a climb for less than the height it gains), and
`length_invariance` 8.83. This is not a tuning problem. An unconstrained regressor has no reason
to respect gravity, and no amount of capacity or data gives it one.

**The whole session turns on one construction (exp1).** Predict the link's *total energy* in the
shape physics gives it, rather than predicting the rate:

```
E(v, g, d, …) = d · [ A + B·g₊ − C·g₋ ] + T          and the harness scores E / d
```

- `A` — per-mile road-load rate
- `B`, `C` — per-mile climb and descent slopes, per grade-percent
- `T` — a fixed per-link transient cost, the acceleration energy a link average hides

This works because **the ceiling in `physics.py` decomposes exactly the same way**: `resistance`
and `accessory` scale with `d`, `transient` does not, `potential` scales with both `d` and `g`.
So squashing each head into its own physically legal interval bounds the total exactly, with no
slack wasted:

| head | range | what it protects |
|---|---|---|
| `A` | `[res/2, res/η + acc]` | `flat_energy_positive`, `energy_grows_with_distance`, `absolute_ceiling` |
| `B` | `[k_pot, k_pot/η + slack/g]` | `climb_floor`, `absolute_ceiling` |
| `C` | `[0, η·k_pot]` | `monotone_in_grade`, `round_trip_convexity`, `regen_ceiling`, `absolute_floor` |
| `T` | `[−η·transient·gate, transient/η]` | `absolute_ceiling`, `absolute_floor` |

with `k_pot = MASS·G·MI_TO_M/100/J_PER_GGE` and `res/2` because `physics.py` builds the
distance-growth bound with `FLOOR_SAFETY_FACTOR = 0.5`.

**All nine checks passed on the first run of this construction**, and every experiment since has
passed all nine without a single one being aimed at physics. `physics_violation_rate` and
`long_link_violation_rate` have been 0.0 since exp1.

**The cost of being physically legal is under 1%.** exp1 scored `rmse` 0.013397 against the
unconstrained forest's 0.013345 — +0.39% — while seeing *only speed* at the network. Grade and
length were carried entirely by the algebra. Everything after that was free upside.

### Three rules this construction imposes

- **A head may not depend on the current link's own grade.** A head that varied with grade could
  make a climb cheaper than flat ground. Grade enters through the algebra only.
- **A head may not depend on the current link's own length.** Same argument for
  `energy_grows_with_distance`. This is why length-dependence of the *rate* has to arrive as
  `T/d` rather than by feeding `miles` to the network — which is also the physically correct
  shape, so nothing is lost.
- **Do the structural algebra in float64.** The heads sit exactly on their bounds when a sigmoid
  saturates, and `physics.py` compares against float64 constants at `TOL = 1e-12`. In float32 the
  climb slope lands a part in 1e7 below `k_pot` and `climb_floor` reads as violated by 4e-9 GGE.
  The network can stay in float32; only the algebra needs the precision.

## The single biggest accuracy result: audit what the structure *cannot represent*

exp7 was worth **−26.2% rmse and −14.9% trip_rmse** — more than every feature in the session put
together — and it added **no information at all**. It added a *sign*.

With `T ≥ 0` and `A ≥ res/2`, a flat link could never be predicted below half its road load. But
a quarter of the data sits at or below zero: link energy is negative under regenerative braking,
and the model was structurally incapable of saying so.

The fix keeps the guarantee intact. Let `T` go negative, but gate the negative part on the
kinetic energy the link **actually releases**:

```
released = max( max(ke(v_prev), ke(v)) − ke(v_next), 0 )      # exp10 widened this to the link's own peak
T        = (transient/η)·σ(t) − σ(u)·min(released, η·transient)
```

`released` is **zero at steady state**, and every synthetic link in the physics sweep *is* steady
state (each is its own single-link journey, so both neighbour speeds equal the link's own). So
`flat_energy_positive` stays true by construction on the sweep while the model is free to predict
regen on real links. The cap `η·transient` is exactly the `absolute_floor` bound.

**The transferable rule: before adding features to a constrained model, enumerate what its
parametrization cannot express.** A bound that is *correct* can still be *binding in the wrong
direction*. Contrast exp6, which relaxed the climb bound on the same suspicion and bought only
0.15% — the climb slope was barely binding, so the diagnostic value exceeded the accuracy value.
Both experiments are the same question asked of two different heads; only one head was the
problem.

## Re-test everything after a structural change — this is not optional here

Three features were screened *before* exp7 and were wrong at the time:

| feature | before exp7 | after exp7 |
|---|---|---|
| `prev_miles` | discard (rmse −3.1%, **trip +0.31%**) | keep, **−8.8% / −5.1%** (exp8) |
| `prev2_speed_mph` | discard (rmse −0.4%, trip +0.07%) | keep, **−2.6% / −0.8%** (exp11) |

The mechanism is specific and worth remembering: before the signed transient, a sharper estimate
of the entry speed had nowhere useful to go, because the model could only ever *add* energy. Once
the structure could subtract, the same features became valuable. **A feature screen is valid only
against the architecture it was run on.**

## Feature findings

Ranked by what each was worth when added, all under the structured model:

| feature | rmse | trip_rmse | note |
|---|---|---|---|
| `prev_speed_mph` (exp2) | −19.3% | −1.7% | the single largest feature |
| `prev_miles` (exp8) | −8.8% | −5.1% | only after exp7 |
| `next_speed_mph` (exp3) | −4.7% | −4.6% | |
| `prev2_speed_mph` (exp11) | −2.6% | −0.8% | only after exp7 |
| `next_miles` (exp9) | −2.5% | −1.2% | |
| `sinuosity` (exp18) | −1.9% | −2.2% | worth *more* here than in domain-guided |
| `next2_speed_mph` (exp12) | −1.4% | −0.9% | |
| `junction_turn_degrees` (exp19) | −1.1% | −0.2% | a quarter of its domain-guided value |
| `prev_grade_percent` (exp13) | −0.15% | −0.89% | six times more trip than link |
| `entry_kinetic_gge` (exp17) | −0.19% | −0.14% | marginal, see below |

- **Entry beats exit by 2–3×, consistently**, across all three matched pairs (`speed`, `miles`,
  and `2`-out speeds). Same ratio the sibling trees found, and the physics is the reason:
  acceleration into a link is billed to it in full; braking out is split with the next link.
- **Grade breaks that symmetry entirely.** `prev_grade_percent` pays (−0.89% trip);
  `next_grade_percent` is an exact tie and was discarded (exp14). The terrain *behind* determines
  the momentum the vehicle actually arrives with; the terrain ahead only matters through the exit
  speed, which `next_speed_mph` already states.
- **`prev_grade_percent` is six times more valuable at trip level than link level** — the
  signature of removing systematic bias rather than adding information. Terrain is spatially
  correlated, so a whole trip through hills was biased the same direction on every link. The
  unguided tree measured this feature at ~0; scoring an unstructured rate, it could not see it.
- **Hand-engineering a kinetic term is nearly worthless here** (`entry_kinetic_gge`, −0.19%),
  where it was the top feature in the unguided tree. The reason is precise: there it carried a
  division by distance, which is genuinely hard; here the *structure* already performs that
  division, leaving only a squaring, which 128 ReLU units approximate for free. **The value of a
  hand-engineered feature is set by what the model cannot do, and structure changes that.**
- **Geometry pays in this tree** (sinuosity −1.9%/−2.2%), against ~0 in unguided. The sweep has
  no `geometry` column, so `predict` supplies sinuosity 1.0 — correct, because a synthetic link
  is straight. `physics.py` explicitly requires a model consuming a non-determined column to
  derive it in its own closure.

## What does not work

- **Widening the network 128 → 256** (exp15): rmse +0.8%, trip_rmse −0.6%. This is the exact
  metric split the unguided tree reported for *every* capacity increase. `length_invariance`
  jumped 0.13 → 1.22, which makes the mechanism visible: the extra capacity went into making the
  fixed cost `T` vary more sharply per link, i.e. into fitting per-link noise.
- **Halving the epochs to 150** (exp16): both metrics worse. 300 epochs is not slack.
- **`next_grade_percent`** (exp14): exact tie. See the asymmetry note above.

## Deliberately out of scope this session (operator brief)

- **No feature may read energy from any link**, including leave-one-out journey encodings. The
  unguided tree's "journey offset" was its biggest `trip_rmse` lever and is excluded here: it
  works only because the harness splits rows rather than journeys, so it is a property of the
  exam rather than a model.
- **Timing-derived features (`gap_seconds`, `next_gap_seconds`) were not used**, although the
  unguided tree found `gap_seconds` worth −4.8%. Idle time between links is a simulation output,
  not something a router knows about a road it has not driven. This is a judgement call, not a
  measured dead end — someone who disagrees should test it.

## Protocol notes

- **Machine load swings run time by ~2.4×** for identical code (98s to 455s for the same 300
  epochs). This killed the ensemble plan at exp16: two members would fit at the fast end and
  truncate against the training-time cap at the slow end, making the result depend on GPU
  contention. Anything whose cost is near the budget is not reproducible here.
- **Decode the WKB geometry once per frame.** Two decodes (~20s each) were enough to push a run
  into its training cap at exp19. Feature preprocessing competes with the training budget.
- `run_with_budget` forks, so CUDA must not be initialized in the parent — pick the device
  *inside* `train_model`, never at module scope.
- Results are deterministic (fixed seed, fixed split), so sub-1% deltas are real.

## Open hypotheses

- The `C` (descent) head is bounded at `η·k_pot`, but `regen_ceiling` actually allows up to
  `k_pot/η` on shallow grades — a factor of `1/η²` more recovery. `implied_eta_regen` sits at
  0.82 against the 0.917 cap, so it does not look binding, which is why it has not been tried.
- The heads cannot depend on the current link's length. A monotone-in-`d` construction would let
  them, but `length_invariance` is already 0.06–0.42, so the model does not appear to be
  straining against this.
- Untouched: learning rate, batch size, depth, weight decay, activation, and any form of
  ensembling or blending (blocked by the load variance, not by evidence).
- A chain model (dilated CNN over the journey's links) feeding the same four bounded heads would
  keep every guarantee, since the guarantee depends only on how the heads are squashed and not on
  how they are computed. The ±2 hand-built window is still paying ~1% per feature, so the chain
  is not obviously exhausted.

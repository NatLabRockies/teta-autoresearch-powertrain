# Learnings

Accumulated across sessions. Sessions so far: `bev-aug31` (45 experiments, 13 keeps).

This tree differs from `../unguided` and `../domain-guided` in two ways, and both change what is
worth trying:

1. **`physics.py` is enforced.** A run with `physics_pass: 0.0` is a discard whatever its RMSE.
2. **The model is not for RouteE Compass.** It may use the whole trip's link chain at inference,
   so there is no one-link-lookback limit and inference cost is not an objective.

## Best known configuration

Commit `2b5c37e` (exp45; the model is exp34's, `52b9c48`, plus a verified-inert reformat).

- **`rmse` 0.005915, `trip_rmse` 0.002084, `physics_pass` 1.0**, every violation rate 0.0.
- Against the scaffold baseline (`rmse` 0.013345, `trip_rmse` 0.003037, **6 of 9 checks failing**)
  that is **−55.7% link RMSE and −31.4% trip RMSE, while going from illegal to legal**.
- A structured physics model (see below) whose four heads are driven by a 128 → 128 ReLU MLP,
  AdamW lr 1e-3, weight decay 1e-4, cosine-annealed to zero, batch 8192, 300 epochs, MSE on the
  rate, on GPU. 418s of the 600s budget.
- **15 network inputs**: `speed_mph`, `prev_speed_mph`, `next_speed_mph`, `prev_miles`,
  `next_miles`, `prev2_miles`, `prev2_speed_mph`, `next2_speed_mph`, `prev_grade_percent`,
  `entry_kinetic_gge`, `exit_kinetic_gge`, `sinuosity`, `junction_turn_degrees`,
  `prev_sinuosity`, `next_junction_turn_degrees` — plus `grade_percent` and `miles`, which are
  consumed by the *structure* rather than by the network.

## The result that matters: physics is a model family, not a penalty

The scaffold forest fails **six of nine checks** — `climb_floor` on 81% of cases, an implied
drivetrain efficiency of **1.29** (it bills a climb for less than the height it gains), and
`length_invariance` 8.83. This is not a tuning problem. An unconstrained regressor has no reason
to respect gravity, and no amount of capacity or data gives it one.

**The session turns on one construction (exp1).** Predict the link's *total energy* in the shape
physics gives it, rather than predicting the rate:

```
E(v, g, d, …) = d · [ A + B·g₊ − C·g₋ ] + T          and the harness scores E / d
```

- `A` — per-mile road-load rate
- `B`, `C` — per-mile climb and descent slopes, per grade-percent
- `T` — a per-link transient cost, the acceleration energy a link average hides

This works because **the ceiling in `physics.py` decomposes exactly the same way**: `resistance`
and `accessory` scale with `d`, `transient` does not, `potential` scales with both `d` and `g`.
So squashing each head into its own physically legal interval bounds the total exactly:

| head | range | what it protects |
|---|---|---|
| `A` | `[res/2, res/η + acc]` | `flat_energy_positive`, `energy_grows_with_distance`, `absolute_ceiling` |
| `B` | `[k_pot, k_pot/η + slack/g]` | `climb_floor`, `absolute_ceiling` |
| `C` | `[0, η·k_pot]` | `monotone_in_grade`, `round_trip_convexity`, `regen_ceiling`, `absolute_floor` |
| `T` | `[−η·transient·gate, (transient/η)·growth]` | `absolute_ceiling`, `absolute_floor` |

with `k_pot = MASS·G·MI_TO_M/100/J_PER_GGE` and `res/2` because `physics.py` builds the
distance-growth bound with `FLOOR_SAFETY_FACTOR = 0.5`.

**All nine checks passed on the first run of this construction**, and only one experiment in the
session (exp24) has failed physics since — and that one exposed a bug rather than a limitation.

**The cost of being physically legal is under 1%.** exp1 scored `rmse` 0.013397 against the
unconstrained forest's 0.013345 — +0.39% — while the network saw *only speed*. Grade and length
were carried entirely by the algebra. Everything after that was free upside.

### Three rules the construction imposes

- **No head may depend on the current link's own grade.** A head that varied with grade could
  make a climb cheaper than flat ground. Grade enters through the algebra only.
- **A head may depend on the current link's length only if it is monotone in it** — see the exp31
  entry below, which is where a large chunk of the session's accuracy came from.
- **Do the structural algebra in float64.** The heads sit exactly on their bounds when a sigmoid
  saturates, and `physics.py` compares against float64 constants at `TOL = 1e-12`. In float32 the
  climb slope lands a part in 1e7 below `k_pot` and `climb_floor` reads as violated by 4e-9 GGE.
  The network can stay in float32; only the algebra needs the precision.

## The two biggest wins came from the same move — and it is not "add a feature"

**Audit what the parametrization cannot express, then re-derive the bound from the check rather
than from the first sufficient condition that happened to work.**

**exp7 (−26.2% rmse, −14.9% trip) added no information at all — it added a sign.** With `T ≥ 0`
and `A ≥ res/2`, a flat link could never be predicted below half its road load, yet a quarter of
the data sits at or below zero: link energy goes negative under regenerative braking and the
model was structurally incapable of saying so. The fix keeps the guarantee: let `T` go negative,
gated on the kinetic energy the link **actually releases**,

```
released = max( ke(v_prev) − ke(v_next), 0 )
T        = (transient/η)·σ(t)·growth − σ(u)·min(released, η·transient)
```

`released` is **zero at steady state**, and every synthetic sweep link is steady state, so
`flat_energy_positive` stays true by construction while the model is free to predict regen on
real links. The cap `η·transient` is exactly the `absolute_floor` bound.

**exp31 (−4.8% rmse, −2.2% trip) removed a constraint that was never required.** Heads were
barred from depending on link length to protect `energy_grows_with_distance` — but that check is
already satisfied by `A ≥ res/2` alone, so **any `T` non-decreasing in length is legal**. A
half-mile link at 30 mph average plausibly contains more acceleration events than a hundred-foot
one at the same average, and a length-independent `T` says they contain the same. Implemented as
a saturating factor `1 − σ(w)·exp(−d/λ)` with `λ` learned, which reduces to the old model at
`σ(w) = 0`, so the relaxation only adds freedom.

Two controls make this specific rather than lucky: giving `A` the same length freedom bought
nothing (exp32, tie/+0.29%), and it added the same two head outputs — so exp31 is not an artifact
of extra parameters. A cruise rate should not care how the road was segmented; a transient cost
should. The data agrees with both.

**The contrast that sharpens the rule:** exp6 relaxed the *climb* bound on the same suspicion and
bought 0.15%. Same question, different head, and only one head was the problem. Three independent
results now say the grade response is not the limitation — exp6 (slack barely binding), exp27
(descent shape irrelevant), exp36 (opening the slack gate earlier is worse). Potential energy is
the one part of the physics a link average does *not* hide, so it is nearly determined and extra
freedom there only fits noise.

## Re-test everything after a structural change

Three verdicts flipped when re-run after exp7 or exp31:

| feature | before | after |
|---|---|---|
| `prev_miles` | discard (rmse −3.1%, **trip +0.31%**) | keep, **−8.8% / −5.1%** (exp8) |
| `prev2_speed_mph` | discard (rmse −0.4%, trip +0.07%) | keep, **−2.6% / −0.8%** (exp11) |
| `next_junction_turn_degrees` | discard (physics fail, exp24) | keep, −0.38% / −0.23% (exp26) |

Before the signed transient, a sharper estimate of the entry speed had nowhere useful to go,
because the model could only ever *add* energy. Once the structure could subtract, the same
features became valuable.

**But the rule has a limit worth recording: it flipped every *feature* verdict it was applied to
and neither *capacity* verdict.** Width 256 was re-tested after exp31 (exp33) and returned an
almost identical result to exp15. A feature's value depends on what the model can do with it; the
capacity trade is a property of the two metrics and does not move when the model does.

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
| `exit_kinetic_gge` (exp29) | tie | −0.9% | judgement call, see the JSONL |
| `prev_sinuosity` (exp23) | −0.5% | −0.4% | |
| `next_junction_turn_degrees` (exp26) | −0.4% | −0.2% | |
| `prev_grade_percent` (exp13) | −0.15% | −0.89% | six times more trip than link |
| `entry_kinetic_gge` (exp17) | −0.19% | −0.14% | marginal on add, −0.46% trip to remove |
| `prev2_miles` (exp34) | −0.10% | −0.33% | |

**The window is ±2 links and the edge is sharp.** `prev3_speed_mph` fails (exp30, rmse −0.23% /
trip +0.89%) and `next2_miles` is actively harmful (exp35). Both sibling trees independently
found the signal is local; this is the third measurement of the same thing, and it is why no
sequence model over the whole trip was built despite the tree's brief allowing one.

**Which features are two-sided, and which are not:**

- **Speed and length are two-sided**, with entry beating exit by a consistent **2–3×** across
  every matched pair. The physics is the reason: acceleration into a link is billed to it in
  full; braking out is split with the next link by regen.
- **Grade and geometry are one-sided.** `next_grade_percent` (exp14) and `next_sinuosity`
  (exp42) are both exact ties, and `prev2_grade_percent` (exp39) is worse on both. The terrain
  and shape *behind* determine the momentum the vehicle actually arrives with; ahead, they only
  matter through the exit speed, which `next_speed_mph` already states directly.
- **`prev_grade_percent` is six times more valuable at trip level than link level** — the
  signature of removing systematic bias rather than adding information. Terrain is spatially
  correlated, so a whole trip through hills was biased the same direction on every link. The
  unguided tree, scoring an unstructured rate, measured this feature at ~0.

**Hand-engineering a kinetic term is nearly worthless here** (`entry_kinetic_gge`, −0.19%), where
it was the top feature in the unguided tree. There it carried a division by distance, which is
genuinely hard; here the *structure* already performs that division, leaving only a squaring,
which 128 ReLU units approximate for free. **The value of a hand-engineered feature is set by
what the model cannot do, and structure changes that.**

**The feature set is minimal, not padded.** Two removal screens (exp28 `entry_kinetic_gge`,
exp38 `next2_speed_mph`) both cost both metrics. `entry_kinetic_gge` is worth *more* to remove
(−0.46% trip) than it was to add (−0.14%): the neighbour lengths and turns that arrived after it
tell the model when to trust it. A one-at-a-time probe measures a feature before its conditioners
exist, so a removal screen is the only honest final check.

## The capacity/convergence frontier is closed — do not re-explore it

Every knob that changes effective capacity produced **the same trade**: rmse worse, trip_rmse
better, in the same ratio.

| change | rmse | trip_rmse |
|---|---|---|
| width 128 → 256 (exp15) | +0.80% | −0.63% |
| lr 1e-3 → 2e-3 (exp20) | +0.10% | −0.56% |
| a third hidden layer (exp21) | +0.35% | −1.16% |
| width 256 re-tested after exp31 (exp33) | +1.00% | −0.57% |

And every change that reduces convergence produced **the reverse**: epochs 300 → 150 (exp16, both
worse), and a 2-member ensemble at half the epochs each (exp22, rmse −0.13% / trip +0.46%). The
current model sits on the knee, and **no reallocation of the same compute improves both metrics**.
Weight decay 1e-4 is also load-bearing (exp44, removing it costs 0.29% trip).

The mechanism: extra capacity reduces regional bias, which `trip_rmse` rewards, while fitting
per-link noise, which `rmse` punishes. `length_invariance` makes it visible — it jumped 0.13 →
1.22 under the wider net, i.e. the extra capacity went into making the fixed cost `T` vary more
sharply per link.

Note that **averaging is legal here**: every physics check is a linear inequality in the
prediction, so the average of legal models is itself legal. Ensembling failed on budget, not on
principle — see the protocol note below.

## Protocol notes

- **Results are bit-deterministic at a matched epoch count.** exp40 re-ran the incumbent
  unchanged and reproduced every metric to six decimals including `length_invariance`. Sub-0.5%
  deltas are real; the *only* confound is epoch truncation.
- **Machine load swings run time by ~2.4×** for identical code (98s to 542s for the same 300
  epochs). This is the session's main methodological problem. It killed the ensemble plan at
  exp16 — two members would fit at the fast end and truncate at the slow end — and it left
  exp36–exp39, exp41 and exp43 running 269–297 epochs against a 300-epoch incumbent.
- **When a confound cannot be removed, measure the model's sensitivity to it rather than
  estimating a correction from a different experiment.** At exp41 I corrected a trip regression
  using exp16's epoch scaling and concluded the change was probably a win. That was wrong: the
  same configuration run at 269, 291 and 292 epochs returned `rmse` 0.005901/0.005900/0.005901
  and `trip_rmse` 0.002087/0.002089/0.002087, i.e. it is nearly insensitive to a 10% shortfall,
  and the regression was real (exp43). Sensitivity does not transfer between configurations.
- **A physics failure names the check, not the change.** exp24's `flat_energy_positive` failure
  was caused by exp10, fourteen experiments earlier, which had widened the release gate to the
  link's own speed. Under the 0-fill convention a synthetic sweep link is indistinguishable from
  a real launch-and-stop, so the widened gate evaluated to `ke(v)` instead of zero and the model
  had been free to predict negative level-ground energy ever since — it simply had not. **A
  guarantee that holds in fact is not a guarantee**; exp25 restored the construction at a cost of
  0.42% trip_rmse and that was the right trade in this tree.
- **Decode the WKB geometry once per frame.** Two decodes (~20s each) pushed a run into its
  training cap at exp19. Feature preprocessing competes with the training budget.
- `run_with_budget` forks, so CUDA must not be initialized in the parent — pick the device
  *inside* `train_model`, never at module scope.
- The physics sweep has no `geometry` column, so `predict` must supply neutral values itself:
  `sinuosity` 1.0 and `junction_turn_degrees` 0.0, both correct because a synthetic link is
  straight and entered head-on. `physics.py` explicitly requires this.

## Deliberately out of scope this session (operator brief)

- **No feature may read energy from any link**, including leave-one-out journey encodings. The
  unguided tree's "journey offset" was its biggest `trip_rmse` lever and is excluded here: it
  works only because the harness splits rows rather than journeys, so it is a property of the
  exam rather than a model.
- **Timing-derived features (`gap_seconds`, `next_gap_seconds`) were not used**, although the
  unguided tree found `gap_seconds` worth −4.8%. Idle time between links is a simulation output,
  not something a router knows about a road it has not driven. This is a judgement call, not a
  measured dead end — someone who disagrees should test it, and it is the single most promising
  untested feature if the constraint is relaxed.

## Open hypotheses for the next session

- **The corner-gated regen channel is a near-miss worth revisiting** (exp37/41/43): an extra
  release term gated on the turn at the link's far end, which is exactly zero on the sweep
  because a synthetic link has no geometry. It is the *safe* way to reach the intra-link physics
  exp10 reached unsafely. Measured three times at rmse −0.24% / trip +0.14%, a real 1.7:1 trade,
  so it needs to be paired with something that moves trip_rmse rather than adopted alone.
- **The `T−` release cap binds hard on heavily decelerating links.** A link entered at 60 and
  left at 10 releases `ke(60)`, but the cap is `η·3·ke(v_avg)`; at low average speeds that is far
  less. This is the exam's own `absolute_floor`, so it cannot be exceeded — but it means heavy
  regen links are systematically underpredicted and that is where the remaining `rmse` most
  likely lives. Worth confirming by binning residuals on `ke(prev) − ke(next)`.
- **Nothing was tried to exploit heteroscedasticity**, and the residuals were never binned by
  feature. Both sibling trees found that diagnostic worth more than several experiments; there
  was no budget for it here. Bin by *features*, never by the target — conditioning on an extreme
  `y` manufactures apparent bias even for an optimal predictor.
- Untested: batch size (4096 doubles the step count and would not fit the budget), activation
  function, warmup, label smoothing, stochastic weight averaging.
- A chain model over the whole trip is **probably a non-direction**: three independent
  measurements now say the signal is local to ±2 links. If tried anyway, note that the guarantee
  depends only on how the heads are squashed and not on how they are computed, so a CNN would
  keep every check — but its heads must exclude the *centre* link's own grade, and its padding
  must make a length-1 journey reduce to steady state.

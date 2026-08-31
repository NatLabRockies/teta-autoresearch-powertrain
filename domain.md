# Domain

## Context

Our training data represents simulated vehicle runs over drive cycle traces (typically 1hz).
At each point we simulate the vehicle dynamics and get an energy estimation for that point.
We take these point level results and aggregate them up to the trip/road segment level.
Then, the road segments have attributes like total distance, average speed, average road gradiant, time to traverse, etc.

This tree targets a **BEV (battery electric)** vehicle, the 2017 Chevy Bolt. Link energy can be
**negative** (regenerative braking), so the target distribution is asymmetric and heavy-tailed.

## Constraints

Do not filter or remove data points to reduce error. The model must be able to predict all values in the dataset, including extreme energy rates such as heavy regenerative braking. Filtering outliers artificially lowers RMSE without improving the model's actual predictive capability — we need accurate predictions across the full distribution.

Do not include a feature like link position since at inference time, we will not know the position of a link relative to a whole trajectory.

## Physical plausibility

Accuracy on held-out links is necessary but not sufficient. A model can post an excellent RMSE and still be unusable if it violates the laws of physics.

**The model must obey physics.** `physics.py` scores the model over a synthetic sweep of links — every combination of speed, grade and length — and checks the predictions against physical law. No ground truth is involved; these are statements that must hold for any vehicle. Level ground cannot produce net energy. A steeper climb cannot cost less than a shallower one. A hill and its return leg cannot together cost less than the same distance of flat road. A climb must cost at least the potential energy it gains, and a descent cannot return more energy than the hill ever held. Above all, no prediction may exceed the energy the link could possibly demand, or fall below the energy it could possibly hold, computed from the vehicle's mass, rolling resistance and drag.

The vehicle constants in `physics.py` are the real ones, read out of the FASTSim vehicle file that generated this dataset (`fastsim-vehicles/v1/fastsim-3/bev/chevrolet/bolt/2017/base/r1.yaml`): mass 1757.77 kg, Cd 0.29 over 2.845 m² of frontal area, rolling resistance 0.0073, wheel inertia 0.815 kg·m², battery-to-wheel efficiency 0.917, 250 W baseline accessory draw. These are not assumptions about a plausible BEV — they are the numbers the simulator integrated to produce every row of the training data, so a model that violates a bound built from them is contradicting its own training data.

What keeps those bounds from flagging a correct model is not slack in the constants but two explicit allowances, `TRANSIENT_KINETIC_MULTIPLE` and `ACCESSORY_WATTS`, covering what link averaging hides: a link averaging 20 mph may contain a stop, a launch to 40 and a long coast, and it is billed for all of it. Both are calibrated against the data rather than guessed — binned into the 491 sweep cells holding at least 30 real links, no cell mean violates any bound, and the shipped values sit a step beyond where violations first reach zero. A violation is therefore a real one. Do not treat a near-miss as a rounding error.

## Evaluation

`harness.evaluate()` returns:

- `rmse` — link-level root mean squared error on energy_rate_gge (GGE/mile).
- `trip_rmse` — trip total-energy RMSE in GGE.
- `physics_pass` — 1.0 only if every check passed, 0.0 otherwise.
- `physics_violation_rate` — fraction of all checked cases that violated some law.
- `long_link_violation_rate` — the same fraction, restricted to links beyond 0.5 miles.
- `length_invariance` — largest relative spread of the per-mile rate across link lengths at fixed speed and grade. 0.0 means a prediction does not depend on how the route was segmented.

`evaluate()` takes the model's `predict` as a required argument, since the sweep has to ask the model about links that are not in the test set. Passing it is not optional. If the model cannot score the synthetic frame at all, every check is recorded as an error and `physics_pass` is 0.0 — a model that cannot be asked about an arbitrary link cannot be used in a router.

The check-by-check detail is written to stderr, so it lands in `run.log` next to the `metrics:` line. Read it whenever `physics_pass` is 0.0: it names the failing check and the worst link.

## What counts as better

1. **`physics_pass` must be 1.0.** A run with `physics_pass: 0.0` is a `discard` whatever its RMSE. A model that breaks physical law is not a better model with a caveat; it is the wrong answer arrived at quickly.
2. Among runs that pass, keep only a change that Pareto-dominates the current best on `rmse` and `trip_rmse`, as before.
3. If nothing passes yet, treat `physics_violation_rate` and `long_link_violation_rate` as the metrics to minimize, and keep a change that reduces either without regressing the other. Reaching a passing model is the first objective; optimizing RMSE within the passing set is the second.

Do not chase `physics_pass` by clamping or post-processing the output. Clipping predictions into the legal range hides the defect from the checks while leaving the learned function exactly as wrong as it was. The function itself has to be right — through the model family, the features, or a physically grounded structure such as predicting a residual against a road-load model rather than the rate directly.

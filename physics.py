"""Physical plausibility checks (fixed, do not modify).

Where `harness.evaluate()` measures accuracy against held-out data, this module
asks a different question: is the learned function *physically possible*, and
does it still behave when handed a link longer than anything in the training
data? Both are answered from a synthetic sweep of links, so no ground truth is
needed and any model can be checked at any time.

Ported from `routee.powertrain.validation.physics`, with two differences that
matter:

- **The target here is a rate.** `energy_rate_gge` is GGE *per mile*, so every
  bound below is compared against `rate * miles` — the energy the link actually
  costs — rather than against the prediction directly.
- **The sweep runs out to 5 miles.** Training links top out at 0.4999 mi
  (median 0.040), but a router asks about long links all the time. A model that
  is only correct inside the training range is not usable, and a tree-based
  model is constant outside it by construction, so the long half of the sweep
  is scored separately and reported as its own metric.

Two kinds of statement are produced:

- **Checks** — pass/fail predicates. The first four need no vehicle knowledge at
  all, so a failure is unambiguous. The rest are computed from the vehicle
  parameters, which are known exactly here, so all of them run.
- **Diagnostics** — descriptive numbers with no pass/fail: flat-ground economy
  by speed, implied drivetrain and regeneration efficiency, and how stable the
  per-mile rate is with link length.

Every vehicle constant below comes from the FASTSim file that generated the
training data, so the bounds describe *this* vehicle rather than a plausible
one. What keeps them from flagging correct models is not fudged constants but
two explicit, separately calibrated allowances — `TRANSIENT_KINETIC_MULTIPLE`
and `ACCESSORY_WATTS` — which cover what link averaging hides. Slack that is
named and measured can be audited; slack buried in a wrong drag coefficient
cannot.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# A model scores a frame of synthetic links and returns `energy_rate_gge`
# (GGE per mile), one value per row, in row order.
Predict = Callable[[pd.DataFrame], "np.ndarray | pd.Series | list[float]"]

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

G = 9.80665
"""Standard gravity, m/s^2."""

MI_TO_M = 1609.344
MPH_TO_MS = 0.44704
AIR_DENSITY = 1.225
"""Sea-level air density, kg/m^3."""

J_PER_GGE = 1.2132e8
"""Joules in one gallon of gasoline equivalent (lower heating value)."""

KWH_PER_GGE = J_PER_GGE / 3.6e6
"""33.7 kWh per GGE — the conversion the economy diagnostic reads through."""

# ---------------------------------------------------------------------------
# Vehicle parameters — 2017 Chevrolet Bolt
# ---------------------------------------------------------------------------
#
# Every number in this block is read straight out of the FASTSim vehicle file
# that generated the training data:
#
#   fastsim-vehicles/v1/fastsim-3/bev/chevrolet/bolt/2017/base/r1.yaml
#
# That provenance is what makes these bounds sharp. They are not a description
# of "a plausible BEV" — they are the constants the simulator itself integrated
# to produce every row of `energy_gge` in the dataset, so a model that violates
# them contradicts its own training data rather than merely contradicting a
# guess about the vehicle.

MASS_KG = 1757.77
"""`mass_kilograms` — simulated mass, including the 136 kg cargo load."""

DRAG_COEF = 0.29
"""`chassis.drag_coef`."""

FRONTAL_AREA_M2 = 2.845
"""`chassis.frontal_area_square_meters`."""

CDA_M2 = DRAG_COEF * FRONTAL_AREA_M2
"""Drag area, 0.825 m^2."""

CRR = 0.0073
"""`chassis.wheel_rr_coef` — coefficient of rolling resistance."""

NUM_WHEELS = 4
WHEEL_INERTIA_KG_M2 = 0.815
"""`chassis.wheel_inertia_kilogram_square_meters`."""

WHEEL_RADIUS_M = 0.336
"""`chassis.wheel_radius_meters`."""

ROTATIONAL_MASS_KG = NUM_WHEELS * WHEEL_INERTIA_KG_M2 / WHEEL_RADIUS_M**2
"""Wheel rotation, expressed as the extra 28.9 kg it costs to accelerate."""

EFFECTIVE_MASS_KG = MASS_KG + ROTATIONAL_MASS_KG
"""Mass to use for kinetic energy. Spinning the wheels up is real work, and
FASTSim bills it (`energy_whl_inertia_joules`), so leaving it out would put the
ceiling below the simulator that produced the data."""

ETA_RES = 0.9848857801796105
"""`pt_type.BEV.res.eff_interp.Constant` — battery round-trip efficiency."""

ETA_MOTOR_PEAK = 0.95
"""Peak of `pt_type.BEV.em.eff_interp_achieved` (the curve runs 0.84–0.95)."""

ETA_TRANSMISSION = 0.98
"""`pt_type.BEV.transmission.eff_interp`."""

ETA_DRIVE = ETA_RES * ETA_MOTOR_PEAK * ETA_TRANSMISSION
"""Best-case battery-to-wheel efficiency, 0.917.

The *peak* motor efficiency, deliberately: this divides the ceiling, so the
highest efficiency the driveline can reach gives the lowest — and therefore
safest to violate — ceiling that is still physically honest. The vehicle cannot
do better than this, so nothing legitimate sits above it."""

ETA_REGEN = ETA_DRIVE
"""Wheel-to-battery efficiency on a descent. Symmetric with `ETA_DRIVE`: the
same three components in the same direction. Being the largest recovery the
hardware allows, it puts the floor as low as it can honestly go."""

PWR_AUX_BASE_WATTS = 250.0
"""`pwr_aux_base_watts` — the simulator's baseline accessory draw."""

ECONOMY_BAND_MI_PER_KWH = (1.5, 8.0)
"""Plausible flat-ground economy envelope for a BEV. This one is *not* from the
vehicle file: it is a wide sanity envelope for the whole powertrain class, kept
loose on purpose because it drives a diagnostic with no pass/fail. The Bolt's
own EPA figure is about 3.9 mi/kWh, comfortably inside it."""

# ---------------------------------------------------------------------------
# Allowances
# ---------------------------------------------------------------------------
#
# The constants above describe steady-state cruise, but a row of the training
# data is a link *average*: a link averaging 20 mph may hold a stop, a launch to
# 40, and a long coast, and it is billed for all of it. Bounds built from
# steady-state road load alone therefore flag the ground truth itself — at the
# real constants, roughly 2% of real links sit above a no-allowance ceiling.
#
# Both allowances below were calibrated against the data rather than guessed.
# Binning the dataset into the 491 sweep cells that hold at least 30 real links
# and taking each cell's mean energy — which is what a well-fit model predicts —
# a 1000 W accessory draw with twice the kinetic term is the point where
# violations reach zero. The shipped values sit a further step beyond that, so
# no correct model is flagged.

TRANSIENT_KINETIC_MULTIPLE = 3.0
"""Accelerations a link may contain, in units of one stop-to-average-speed
launch. Calibration needs 2; this is 3."""

ACCESSORY_WATTS = 2000.0
"""Accessory draw the ceiling allows, well above the simulator's 250 W base.

A link creeping at 0.13 mph for 70 seconds really did draw about 520 W in the
training data, so the base figure alone does not bound observed behavior; and
at walking pace the kinetic and road-load terms both vanish, leaving this as
the only thing holding the ceiling off the floor. Calibration needs 1000 W."""

FLOOR_SAFETY_FACTOR = 0.5
"""How far below the real road load the distance-growth floor sits.

`energy_grows_with_distance` is a lower bound, so it needs the real constants
scaled *down* rather than up. Across the sweep cells with real support, the
observed energy growth per added mile is 1.09 times the road-load work at the
median and never falls below 0.795 of it, so half is clear of anything the
data does."""

# ---------------------------------------------------------------------------
# Sweep geometry
# ---------------------------------------------------------------------------

SPEEDS_MPH = np.array([5.0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70])
GRADES_PCT = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0])

TRAINING_MAX_MILES = 0.5
"""Longest link in the training data (max is 0.4999). Everything in the sweep
beyond this is extrapolation, and is scored separately."""

DISTANCES_MI = np.array([0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
"""Out to 5 miles — ten times the longest link the model was ever trained on."""

SHALLOW_GRADE_PCT = 3.0
"""Grade band used to fit implied efficiencies. Steep grades saturate in most
forests, which would bias the slope."""

TOL = 1e-12


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class Check:
    """The outcome of one physical predicate over the synthetic sweep."""

    name: str
    reason: str
    n_tested: int = 0
    n_violations: int = 0
    #: Violations among the sweep cases longer than `TRAINING_MAX_MILES`.
    n_long_tested: int = 0
    n_long_violations: int = 0
    #: Size of the worst violation, in GGE. Negative.
    worst_margin: float | None = None
    #: The link that produced the worst violation.
    worst_case: dict[str, float] = field(default_factory=dict)
    #: Set when the predicate could not be evaluated at all.
    error: str | None = None

    @property
    def failed(self) -> bool:
        return self.n_violations > 0 or self.error is not None

    @property
    def violation_rate(self) -> float:
        return self.n_violations / self.n_tested if self.n_tested else 0.0


@dataclass
class Report:
    """Everything the physical validation found for one model."""

    checks: list[Check] = field(default_factory=list)
    #: Flat-ground economy in mi/kWh, keyed by speed in mph.
    flat_economy: dict[str, float] = field(default_factory=dict)
    #: Speeds whose flat-ground economy falls outside `ECONOMY_BAND_MI_PER_KWH`.
    economy_outliers: list[float] = field(default_factory=list)
    #: Implied battery-to-wheel efficiency, from the climb-side slope.
    implied_eta_drive: float | None = None
    #: Implied regeneration efficiency, from the descent-side slope. Above 1.0
    #: is energy from nowhere.
    implied_eta_regen: float | None = None
    #: Largest relative spread of the per-mile rate across link lengths. Zero
    #: means predictions do not depend on how a route was segmented.
    length_invariance: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(c.failed for c in self.checks)

    def metrics(self) -> dict[str, float]:
        """The physics half of what `harness.evaluate()` returns.

        Four keys, always present with the same names whatever the model is, so
        a column in the results TSV means the same thing in every row.
        """
        tested = sum(c.n_tested for c in self.checks)
        violations = sum(c.n_violations for c in self.checks)
        long_tested = sum(c.n_long_tested for c in self.checks)
        long_violations = sum(c.n_long_violations for c in self.checks)
        return {
            "physics_pass": 1.0 if self.passed else 0.0,
            "physics_violation_rate": violations / tested if tested else 1.0,
            "long_link_violation_rate": (
                long_violations / long_tested if long_tested else 1.0
            ),
            "length_invariance": (
                float("nan")
                if self.length_invariance is None
                else self.length_invariance
            ),
        }

    def format(self) -> str:
        """A human-readable rendering, for the diagnostic block in `run.log`.

        `physics_pass: 0.0` on its own says a physical law was broken but not
        which one, and the next experiment needs to know which one.
        """
        lines = ["=" * 68]
        lines.append(
            f"physics validation: {'PASS' if self.passed else 'FAIL'}"
            f"   (mass {MASS_KG:.0f} kg, sweep to {DISTANCES_MI.max():g} mi)"
        )
        lines.append("=" * 68)
        for c in self.checks:
            if c.error is not None:
                lines.append(f"{c.name:<28} ERROR    {c.error}")
                continue
            status = "fail" if c.failed else "pass"
            long_rate = (
                c.n_long_violations / c.n_long_tested if c.n_long_tested else 0.0
            )
            lines.append(
                f"{c.name:<28} {status:<8} "
                f"{100 * c.violation_rate:6.1f}%   "
                f"long {100 * long_rate:6.1f}%"
            )
            if c.failed:
                where = ", ".join(f"{k}={v:g}" for k, v in c.worst_case.items())
                lines.append(f"{'':<28} {c.reason}")
                lines.append(f"{'':<28} worst {c.worst_margin:.6g} GGE at {where}")
        lines.append("-" * 68)
        if self.implied_eta_drive is not None:
            lines.append(f"{'implied eta_drive':<28} {self.implied_eta_drive:.3f}")
        if self.implied_eta_regen is not None:
            flag = "   <- energy from nowhere" if self.implied_eta_regen > 1.0 else ""
            lines.append(
                f"{'implied eta_regen':<28} {self.implied_eta_regen:.3f}{flag}"
            )
        if self.length_invariance is not None:
            lines.append(f"{'length invariance':<28} {self.length_invariance:.4f}")
        if self.flat_economy:
            lines.append(f"{'flat economy mi/kWh':<28} {json.dumps(self.flat_economy)}")
        if self.economy_outliers:
            speeds = ", ".join(f"{s:g}" for s in self.economy_outliers)
            lines.append(f"{'economy out of band at':<28} {speeds} mph")
        for note in self.notes:
            lines.append(f"note: {note}")
        lines.append("=" * 68)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------


def build_frame(
    speed_mph: np.ndarray,
    grade_percent: np.ndarray,
    miles: np.ndarray,
) -> pd.DataFrame:
    """One synthetic link per row, carrying every physically determined column.

    A link at a given speed, grade and length fixes its own traversal time, so
    `time_seconds` is derived here rather than swept. The bookkeeping columns
    (`journey_id`, `road_id`, the link timestamps) are filled with benign
    values: each row is its own single-link journey, which is also the neutral
    choice for a one-link-lookback model — the previous link is the link
    itself, i.e. steady-state cruise.

    A model consuming a column that is not physically determined by speed,
    grade and length — `geometry`, say — must derive it inside its own
    `predict` closure. There is no correct synthetic value for it here.
    """
    hours = np.divide(
        miles, speed_mph, out=np.full(miles.shape, np.nan), where=speed_mph > 0
    )
    seconds = hours * 3600.0
    return pd.DataFrame(
        {
            "journey_id": np.arange(len(miles)),
            "road_id": np.arange(len(miles)),
            "miles": miles,
            "speed_mph": speed_mph,
            "grade_percent": grade_percent,
            "grade_dec": grade_percent / 100.0,
            "time_seconds": seconds,
            "link_start_time": np.zeros(len(miles)),
            "link_end_time": seconds,
            # Trajectories are sampled at 1 Hz, so a link holds about one point
            # per second of traversal.
            "n_points": np.maximum(np.round(seconds), 1.0),
        }
    )


def _score(predict: Predict, frame: pd.DataFrame) -> np.ndarray:
    """Predicted energy for each link, in GGE — the rate times the distance."""
    rate = np.asarray(predict(frame), dtype=float).reshape(-1)
    if rate.shape[0] != len(frame):
        raise ValueError(
            f"predict returned {rate.shape[0]} values for {len(frame)} links"
        )
    return rate * frame["miles"].to_numpy(dtype=float)


def _check(
    name: str,
    reason: str,
    margin: np.ndarray,
    context: Mapping[str, np.ndarray],
) -> Check:
    """Summarize one predicate from its per-case margins.

    `margin` is negative exactly where the predicate is violated, and its
    magnitude there is how far into the impossible the prediction went.
    """
    margin = np.asarray(margin, dtype=float)
    finite = np.isfinite(margin)
    # A non-finite margin is a violation: it means the prediction that produced
    # it was NaN or infinite, which is not a pass.
    violated = ~finite | (margin < -TOL)
    is_long = np.asarray(context["distance_mi"]) > TRAINING_MAX_MILES

    check = Check(
        name=name,
        reason=reason,
        n_tested=int(margin.size),
        n_violations=int(violated.sum()),
        n_long_tested=int(is_long.sum()),
        n_long_violations=int((violated & is_long).sum()),
    )
    if check.n_violations:
        worst = int(np.argmin(np.where(finite, margin, -np.inf)))
        check.worst_margin = float(margin[worst])
        check.worst_case = {k: float(v[worst]) for k, v in context.items()}
    return check


def _fit_slope(x: np.ndarray, y: np.ndarray) -> float | None:
    """Least-squares slope of `y` on `x` through the available points."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2 or np.ptp(x[mask]) < TOL:
        return None
    return float(np.polyfit(x[mask], y[mask], 1)[0])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def check_physics(predict: Predict) -> Report:
    """Check a trained model against physical law over a synthetic sweep.

    `predict` maps a frame of synthetic links (see `build_frame`) to
    `energy_rate_gge`, one value per row, in row order. It is called three
    times, on matched flat / climb / descent copies of the same grid, so the
    round-trip statistic is exact rather than interpolated.
    """
    report = Report()

    # Matched (flat, climb, descent) triples at one speed, grade magnitude and
    # length.
    sp, gm, di = (
        a.ravel()
        for a in np.meshgrid(SPEEDS_MPH, GRADES_PCT, DISTANCES_MI, indexing="ij")
    )

    try:
        flat = _score(predict, build_frame(sp, np.zeros_like(gm), di))
        climb = _score(predict, build_frame(sp, gm, di))
        descent = _score(predict, build_frame(sp, -gm, di))
    except Exception as exc:  # noqa: BLE001 - any failure here is a failed check
        report.checks = [
            Check(name=name, reason=reason, error=f"{type(exc).__name__}: {exc}")
            for name, reason in _CHECK_NAMES
        ]
        report.notes.append(
            "the model could not score the synthetic sweep; a model that cannot "
            "be asked about an arbitrary link cannot be used in a router"
        )
        return report

    context = {"speed_mph": sp, "grade_pct": gm, "distance_mi": di}
    tiled = {k: np.tile(v, 3) for k, v in context.items()}
    everything = np.concatenate([flat, climb, descent])
    checks: list[Check] = []

    # -- constant-free checks -----------------------------------------------

    checks.append(
        _check(
            "finite",
            "predictions must not be NaN or infinite",
            # Zero where finite, NaN where not: `_check` counts non-finite
            # margins as violations.
            np.where(np.isfinite(everything), 0.0, np.nan),
            tiled,
        )
    )

    # A vehicle moving on level ground cannot produce net energy.
    checks.append(
        _check(
            "flat_energy_positive",
            "level-ground energy must be positive",
            flat,
            context,
        )
    )

    # Steeper is never cheaper: both the climb and the descent must sit on the
    # correct side of level ground.
    checks.append(
        _check(
            "monotone_in_grade",
            "energy must not decrease as grade increases",
            np.minimum(climb - flat, flat - descent),
            context,
        )
    )

    # A hill and its return leg cannot together cost less than the same
    # distance of level ground.
    checks.append(
        _check(
            "round_trip_convexity",
            "a round trip over a hill cannot cost less than flat ground",
            climb + descent - 2 * flat,
            context,
        )
    )

    # -- does the model survive a link longer than it was trained on? --------
    #
    # Extending a link at the same speed and grade adds road to be covered, and
    # that added road has to be paid for: at minimum the rolling resistance,
    # aerodynamic drag and gravity acting over the extra distance, at an
    # efficiency no better than perfect. Monotonicity alone is too weak a test
    # here — the failure this is built to catch is *saturation*, where a model
    # stops responding to distance past its training range and reports the same
    # total energy for a 5-mile link as for a half-mile one. That is constant,
    # not decreasing, so it slips past a monotonicity predicate untouched.
    #
    # Descents are excluded: a longer descent legitimately returns more energy,
    # so its total falls.
    checks.append(
        _check(
            "energy_grows_with_distance",
            "extending a link must cost at least the road-load work over the "
            "added distance",
            _distance_growth(sp, gm, di, flat, climb),
            {k: np.tile(v, 2) for k, v in context.items()},
        )
    )

    # -- mass-dependent bounds ----------------------------------------------

    rise_m = di * MI_TO_M * gm / 100.0
    potential = MASS_KG * G * rise_m / J_PER_GGE
    velocity = sp * MPH_TO_MS
    # Effective mass, not curb mass: spinning up the wheels is work too.
    kinetic = 0.5 * EFFECTIVE_MASS_KG * velocity**2 / J_PER_GGE
    # What a link may spend on acceleration the average speed does not reveal.
    transient = TRANSIENT_KINETIC_MULTIPLE * kinetic
    resistance = (
        (CRR * MASS_KG * G + 0.5 * AIR_DENSITY * CDA_M2 * velocity**2)
        * (di * MI_TO_M)
        / J_PER_GGE
    )
    # Accessories are billed per unit time, so a slow link costs more of them.
    seconds = np.divide(
        di * MI_TO_M, velocity, out=np.zeros_like(velocity), where=velocity > 0
    )
    accessory = ACCESSORY_WATTS * seconds / J_PER_GGE

    # Lifting the vehicle is work no drivetrain can avoid.
    checks.append(
        _check(
            "climb_floor",
            "a climb must cost at least its potential energy",
            (climb - flat) - potential,
            context,
        )
    )

    # A descent saves two things: the energy it recovers from the hill, and the
    # road-load energy the flat leg would have spent covering that ground
    # anyway. Bounding only the first understates the legitimate saving.
    recoverable = np.minimum(resistance, np.abs(potential))
    max_saving = recoverable / ETA_DRIVE + ETA_REGEN * (np.abs(potential) - recoverable)
    checks.append(
        _check(
            "regen_ceiling",
            "a descent cannot return more energy than the hill holds",
            max_saving - (flat - descent),
            context,
        )
    )

    signed_potential = np.concatenate([np.zeros_like(potential), potential, -potential])
    ceiling = (
        np.maximum(signed_potential, 0.0)
        + np.tile(transient, 3)
        + np.tile(resistance, 3)
    ) / ETA_DRIVE + np.tile(accessory, 3)
    floor = -ETA_REGEN * (np.maximum(-signed_potential, 0.0) + np.tile(transient, 3))

    checks.append(
        _check(
            "absolute_ceiling",
            "prediction exceeds the energy the link could possibly demand",
            ceiling - everything,
            tiled,
        )
    )
    checks.append(
        _check(
            "absolute_floor",
            "prediction returns more energy than the link could possibly hold",
            everything - floor,
            tiled,
        )
    )

    report.checks = checks
    _add_diagnostics(report, sp, gm, di, flat, climb, descent, potential)
    return report


def _min_work_per_meter(velocity: np.ndarray, grade_pct: np.ndarray) -> np.ndarray:
    """Least energy any vehicle could spend covering one metre, in GGE.

    The real road load scaled by `FLOOR_SAFETY_FACTOR`, at perfect drivetrain
    efficiency, so that it sits below anything the data does. On a descent steep
    enough for gravity to carry the vehicle the bound goes to zero: a
    freewheeling car spends nothing, and a negative floor would be a claim about
    how much it must *recover*, which this is not.
    """
    force = FLOOR_SAFETY_FACTOR * (
        CRR * MASS_KG * G
        + 0.5 * AIR_DENSITY * CDA_M2 * velocity**2
        + MASS_KG * G * grade_pct / 100.0
    )
    return np.maximum(force, 0.0) / J_PER_GGE


def _distance_growth(
    sp: np.ndarray,
    gm: np.ndarray,
    di: np.ndarray,
    flat: np.ndarray,
    climb: np.ndarray,
) -> np.ndarray:
    """Per-case margin for `energy_grows_with_distance`.

    For each (speed, grade) group the distances are walked in increasing order
    and each link is compared against the shorter one before it: the energy it
    adds must cover at least the road-load work over the distance it adds. The
    shortest link in each group has nothing to compare against and is scored as
    a pass.
    """
    velocity = sp * MPH_TO_MS
    margins = []
    for values, grade in ((flat, np.zeros_like(gm)), (climb, gm)):
        per_meter = _min_work_per_meter(velocity, grade)
        margin = np.zeros_like(values)
        for speed in np.unique(sp):
            for grade_value in np.unique(gm):
                idx = np.flatnonzero((sp == speed) & (gm == grade_value))
                if idx.size < 2:
                    continue
                order = idx[np.argsort(di[idx])]
                added_m = np.diff(di[order]) * MI_TO_M
                margin[order[1:]] = np.diff(values[order]) - (
                    per_meter[order[1:]] * added_m
                )
        margins.append(margin)
    return np.concatenate(margins)


def _add_diagnostics(
    report: Report,
    sp: np.ndarray,
    gm: np.ndarray,
    di: np.ndarray,
    flat: np.ndarray,
    climb: np.ndarray,
    descent: np.ndarray,
    potential: np.ndarray,
) -> None:
    """Fill in the descriptive half of the report."""
    # -- flat-ground economy, read against a plausible envelope --------------
    #
    # `flat` holds level-ground energy for every case whatever grade magnitude
    # it pairs with, so grouping by speed alone is what is wanted here.
    for speed in np.unique(sp):
        mask = sp == speed
        rate_per_mile = np.divide(
            flat[mask],
            di[mask],
            out=np.full(int(mask.sum()), np.nan),
            where=di[mask] > 0,
        )
        mean_rate = float(np.nanmean(rate_per_mile))
        if not np.isfinite(mean_rate) or abs(mean_rate) < TOL:
            continue
        economy = 1.0 / (mean_rate * KWH_PER_GGE)
        report.flat_economy[f"{speed:g}"] = round(economy, 3)
        low, high = ECONOMY_BAND_MI_PER_KWH
        if not (low <= economy <= high):
            report.economy_outliers.append(float(speed))

    # -- implied efficiencies, from the shallow-grade slopes -----------------
    shallow = (np.abs(gm) > TOL) & (np.abs(gm) <= SHALLOW_GRADE_PCT)
    if shallow.any():
        # Slope of the extra energy a climb costs against the energy it must
        # store as height: the reciprocal of drivetrain efficiency.
        climb_slope = _fit_slope(potential[shallow], (climb - flat)[shallow])
        if climb_slope is not None and climb_slope > TOL:
            report.implied_eta_drive = 1.0 / climb_slope
            if report.implied_eta_drive > 1.0:
                report.notes.append(
                    f"implied drivetrain efficiency is "
                    f"{report.implied_eta_drive:.2f}; above 1.0 means the model "
                    "bills a climb for less than the height it gains"
                )
        regen_slope = _fit_slope(potential[shallow], (flat - descent)[shallow])
        if regen_slope is not None:
            report.implied_eta_regen = regen_slope
            if regen_slope > 1.0:
                report.notes.append(
                    f"implied regeneration efficiency is {regen_slope:.2f}; above "
                    "1.0 means the model returns energy the hill never contained"
                )

    # -- does the answer depend on how the route was segmented? --------------
    spreads = []
    for speed in np.unique(sp):
        for grade in np.unique(gm):
            mask = (sp == speed) & (gm == grade)
            if mask.sum() < 2:
                continue
            rate = np.divide(
                climb[mask],
                di[mask],
                out=np.full(int(mask.sum()), np.nan),
                where=di[mask] > 0,
            )
            scale = float(np.nanmean(np.abs(rate)))
            if np.isfinite(scale) and scale > TOL:
                spread = float(np.nanmax(rate) - np.nanmin(rate)) / scale
                spreads.append(spread)
    if spreads:
        report.length_invariance = float(np.max(spreads))


#: Every check this module can emit, in report order. Used to fill in a report
#: when the sweep itself could not be scored.
_CHECK_NAMES = (
    ("finite", "predictions must not be NaN or infinite"),
    ("flat_energy_positive", "level-ground energy must be positive"),
    ("monotone_in_grade", "energy must not decrease as grade increases"),
    (
        "round_trip_convexity",
        "a round trip over a hill cannot cost less than flat ground",
    ),
    (
        "energy_grows_with_distance",
        "extending a link must cost at least the road-load work over the added "
        "distance",
    ),
    ("climb_floor", "a climb must cost at least its potential energy"),
    ("regen_ceiling", "a descent cannot return more energy than the hill holds"),
    (
        "absolute_ceiling",
        "prediction exceeds the energy the link could possibly demand",
    ),
    (
        "absolute_floor",
        "prediction returns more energy than the link could possibly hold",
    ),
)

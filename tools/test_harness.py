"""Tests for harness.py.

`harness.py` is the fixed point: the split, the metrics, the time budget, and
the report format. `evaluate` defines what "better" means for every experiment
in every session, so it is the one thing in this repo that must not drift, and
`report` is what puts a run's metadata into the results files — a session's
record is only as trustworthy as these.
"""

import io
import json
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
import pandas as pd

import physics
from harness import (
    evaluate,
    report,
    rmse,
    run_with_budget,
    train_test_split,
    trip_rmse,
)


def _flat_rate(df: pd.DataFrame) -> np.ndarray:
    """A stand-in model for the physics sweep: a constant positive rate.

    These tests are about `evaluate`'s contract — which metrics come back, and
    that the arrays stay keyword-only — not about whether a particular model is
    physically plausible, so the stub only has to be callable and correctly
    shaped.
    """
    return np.full(len(df), 0.006)


class RmseTests(unittest.TestCase):
    def test_zero_when_exact(self) -> None:
        actual = np.array([1.0, 2.0, 3.0])
        self.assertEqual(rmse(actual, actual.copy()), 0.0)

    def test_known_value(self) -> None:
        # errors of +1, -1, +1, -1 -> sqrt(mean(1,1,1,1)) == 1.0
        actual = np.array([1.0, 2.0, 3.0, 4.0])
        predicted = np.array([2.0, 1.0, 4.0, 3.0])
        self.assertAlmostEqual(rmse(actual, predicted), 1.0)


class TripRmseTests(unittest.TestCase):
    """Trip energy is sum(rate * miles) over the links of a journey."""

    def test_hand_computed_two_journeys(self) -> None:
        # journey A: 2 links, journey B: 2 links
        journey_id = np.array(["A", "A", "B", "B"])
        miles = np.array([2.0, 3.0, 1.0, 4.0])
        actual = np.array([0.10, 0.20, 0.50, 0.25])
        predicted = np.array([0.15, 0.20, 0.50, 0.20])

        # A actual = 0.10*2 + 0.20*3 = 0.80 ; A pred = 0.15*2 + 0.20*3 = 0.90
        # B actual = 0.50*1 + 0.25*4 = 1.50 ; B pred = 0.50*1 + 0.20*4 = 1.30
        # diffs = -0.10, +0.20 -> sqrt(mean(0.01, 0.04)) = sqrt(0.025)
        expected = float(np.sqrt(0.025))
        self.assertAlmostEqual(
            trip_rmse(actual, predicted, journey_id, miles), expected
        )

    def test_zero_when_exact(self) -> None:
        journey_id = np.array(["A", "A", "B"])
        miles = np.array([1.0, 2.0, 3.0])
        actual = np.array([0.1, 0.2, 0.3])
        self.assertEqual(trip_rmse(actual, actual.copy(), journey_id, miles), 0.0)

    def test_offsetting_link_errors_cancel_within_a_trip(self) -> None:
        """The bias/variance split that motivates tracking both metrics.

        Two link errors of equal magnitude and opposite sign cancel in the
        trip total, so trip_rmse is 0 while link rmse is not.
        """
        journey_id = np.array(["A", "A"])
        miles = np.array([1.0, 1.0])
        actual = np.array([0.10, 0.20])
        predicted = np.array([0.15, 0.15])

        self.assertAlmostEqual(trip_rmse(actual, predicted, journey_id, miles), 0.0)
        self.assertGreater(rmse(actual, predicted), 0.0)

    def test_journey_order_does_not_matter(self) -> None:
        """Grouping is by id, not by adjacency — interleaved rows are fine."""
        journey_id = np.array(["A", "B", "A", "B"])
        miles = np.array([2.0, 1.0, 3.0, 4.0])
        actual = np.array([0.10, 0.50, 0.20, 0.25])
        predicted = np.array([0.15, 0.50, 0.20, 0.20])

        expected = float(np.sqrt(0.025))
        self.assertAlmostEqual(
            trip_rmse(actual, predicted, journey_id, miles), expected
        )


class EvaluateTests(unittest.TestCase):
    def test_returns_both_metrics(self) -> None:
        journey_id = np.array(["A", "A", "B", "B"])
        miles = np.array([2.0, 3.0, 1.0, 4.0])
        actual = np.array([0.10, 0.20, 0.50, 0.25])
        predicted = np.array([0.15, 0.20, 0.50, 0.20])

        # The physics report goes to stderr; swallow it so the test output
        # stays readable.
        with redirect_stderr(io.StringIO()):
            result = evaluate(
                actual,
                predicted,
                journey_id=journey_id,
                miles=miles,
                predict=_flat_rate,
            )

        self.assertEqual(
            set(result),
            {
                "rmse",
                "trip_rmse",
                "physics_pass",
                "physics_violation_rate",
                "long_link_violation_rate",
                "length_invariance",
            },
        )
        self.assertAlmostEqual(result["rmse"], rmse(actual, predicted))
        self.assertAlmostEqual(
            result["trip_rmse"], trip_rmse(actual, predicted, journey_id, miles)
        )

    def test_journey_id_and_miles_are_keyword_only(self) -> None:
        """Positional misuse must fail loudly rather than silently mis-score."""
        arr = np.array([1.0, 2.0])
        with self.assertRaises(TypeError):
            evaluate(arr, arr, arr, arr, predict=_flat_rate)  # type: ignore[misc]

    def test_physics_cannot_be_skipped(self) -> None:
        """`predict` is required: a run cannot decline to be checked."""
        arr = np.array([1.0, 2.0])
        with self.assertRaises(TypeError):
            evaluate(arr, arr, journey_id=arr, miles=arr)  # type: ignore[call-arg]


class PhysicsTests(unittest.TestCase):
    """`physics.py` is part of the fixed point, so its verdicts must not drift."""

    def test_sweep_reaches_five_miles(self) -> None:
        """The extrapolation range is the point; it has to actually be swept."""
        self.assertEqual(physics.DISTANCES_MI.max(), 5.0)
        self.assertTrue((physics.DISTANCES_MI > physics.TRAINING_MAX_MILES).any())

    def test_road_load_model_passes(self) -> None:
        """A physically derived model must not be flagged.

        The bounds are meant to catch divergence, not to fail correct models.
        This rate is the FASTSim road load itself, at the vehicle's own
        efficiency, plus its baseline accessory draw — as close to the truth as
        a closed-form model gets. If the checks flag this, they are wrong.
        """

        def road_load(df: pd.DataFrame) -> np.ndarray:
            v = df["speed_mph"].to_numpy() * physics.MPH_TO_MS
            grade = df["grade_percent"].to_numpy() / 100.0
            # Force in newtons: rolling, aero, and gravity along the slope.
            force = (
                physics.CRR * physics.MASS_KG * physics.G
                + 0.5 * physics.AIR_DENSITY * physics.CDA_M2 * v**2
                + physics.MASS_KG * physics.G * grade
            )
            joules_per_m = np.where(
                force > 0,
                force / physics.ETA_DRIVE,
                force * physics.ETA_REGEN,
            )
            energy_j = joules_per_m * df["miles"].to_numpy() * physics.MI_TO_M
            energy_j += physics.PWR_AUX_BASE_WATTS * df["time_seconds"].to_numpy()
            rate_per_mile = energy_j / df["miles"].to_numpy()
            return rate_per_mile / physics.J_PER_GGE

        report = physics.check_physics(road_load)
        failed = [c.name for c in report.checks if c.failed]
        self.assertEqual(failed, [], msg=report.format())

    def test_constants_match_the_fastsim_vehicle_file(self) -> None:
        """The bounds are only sharp if these are the simulator's own numbers.

        Pinned against
        `fastsim-vehicles/v1/fastsim-3/bev/chevrolet/bolt/2017/base/r1.yaml`,
        so that an edit here has to be a deliberate one.
        """
        self.assertEqual(physics.MASS_KG, 1757.77)
        self.assertEqual(physics.DRAG_COEF, 0.29)
        self.assertEqual(physics.FRONTAL_AREA_M2, 2.845)
        self.assertEqual(physics.CRR, 0.0073)
        self.assertEqual(physics.WHEEL_INERTIA_KG_M2, 0.815)
        self.assertEqual(physics.WHEEL_RADIUS_M, 0.336)
        self.assertEqual(physics.PWR_AUX_BASE_WATTS, 250.0)
        self.assertAlmostEqual(physics.CDA_M2, 0.82505)
        self.assertAlmostEqual(physics.ETA_DRIVE, 0.9169287, places=6)

    def test_constant_rate_model_is_caught(self) -> None:
        """A model that ignores grade cannot be paying for the hill it climbs."""
        report = physics.check_physics(lambda df: np.full(len(df), 0.006))
        failed = {c.name for c in report.checks if c.failed}
        self.assertIn("climb_floor", failed)
        self.assertEqual(report.metrics()["physics_pass"], 0.0)

    def test_length_saturation_is_caught(self) -> None:
        """The failure mode a forest has outside its training range.

        Total energy is capped past the longest link ever seen, so a 5-mile
        link costs no more than a half-mile one.
        """

        def saturating(df: pd.DataFrame) -> np.ndarray:
            miles = df["miles"].to_numpy()
            capped = np.minimum(miles, physics.TRAINING_MAX_MILES)
            # A fixed energy per link above the cap means a falling per-mile
            # rate, which is what the model would actually report.
            return 0.006 * capped / miles

        report = physics.check_physics(saturating)
        failed = {c.name for c in report.checks if c.failed}
        self.assertIn("energy_grows_with_distance", failed)
        self.assertGreater(report.metrics()["long_link_violation_rate"], 0.0)
        self.assertGreater(report.length_invariance or 0.0, 0.0)

    def test_unscorable_model_fails_every_check(self) -> None:
        """A model that cannot answer about an arbitrary link is not usable."""

        def needs_geometry(df: pd.DataFrame) -> np.ndarray:
            return df["geometry"].to_numpy()

        report = physics.check_physics(needs_geometry)
        self.assertFalse(report.passed)
        self.assertTrue(all(c.error is not None for c in report.checks))
        self.assertEqual(report.metrics()["physics_pass"], 0.0)


class TrainTestSplitTests(unittest.TestCase):
    def test_split_is_deterministic_and_total(self) -> None:
        df = pd.DataFrame({"x": range(1000)})

        train_a, test_a = train_test_split(df, test_size=0.2, random_seed=42)
        train_b, test_b = train_test_split(df, test_size=0.2, random_seed=42)

        self.assertEqual(len(train_a) + len(test_a), len(df))
        self.assertEqual(list(test_a["x"]), list(test_b["x"]))
        self.assertEqual(list(train_a["x"]), list(train_b["x"]))

    def test_split_is_disjoint(self) -> None:
        df = pd.DataFrame({"x": range(1000)})
        train, test = train_test_split(df, test_size=0.2, random_seed=42)
        self.assertEqual(set(train["x"]) & set(test["x"]), set())


def _capture(
    model_family: str = "RandomForest",
    features: list[str] | None = None,
    total_seconds: float = 9.44,
) -> dict[str, str]:
    """Run `report` and return its output as {line-prefix: rest-of-line}."""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        report(
            {"rmse": 0.0395901, "trip_rmse": 0.421},
            model_family=model_family,
            features=["a", "b"] if features is None else features,
            total_seconds=total_seconds,
        )
    lines = buffer.getvalue().strip().splitlines()
    return dict(line.split(": ", 1) for line in lines)


class ReportTests(unittest.TestCase):
    def test_metrics_line_is_valid_json_with_the_given_keys(self) -> None:
        out = _capture()
        self.assertEqual(
            json.loads(out["metrics"]), {"rmse": 0.03959, "trip_rmse": 0.421}
        )

    def test_meta_line_carries_family_features_and_duration(self) -> None:
        meta = json.loads(_capture()["meta"])
        self.assertEqual(
            meta,
            {
                "model_family": "RandomForest",
                "features": ["a", "b"],
                "total_seconds": 9.4,
            },
        )

    def test_nothing_is_reported_twice(self) -> None:
        # Two lines, no restatement: a value printed in both a human-readable
        # line and a JSON line is two places for it to be read from and two
        # places for it to disagree.
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            report(
                {"rmse": 0.0395901},
                model_family="RandomForest",
                features=["a", "b"],
                total_seconds=9.44,
            )
        lines = buffer.getvalue().strip().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("metrics: "))
        self.assertTrue(lines[1].startswith("meta: "))

    def test_family_is_whatever_train_py_says(self) -> None:
        # Free text, not a checked enum — a new family needs no change here.
        meta = json.loads(_capture(model_family="LookbackGRU")["meta"])
        self.assertEqual(meta["model_family"], "LookbackGRU")

    def test_metric_names_are_never_hardcoded(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            report(
                {"mape": 1.5, "bias": -0.25},
                model_family="MLP",
                features=["a"],
                total_seconds=1.0,
            )
        self.assertEqual(
            json.loads(buffer.getvalue().splitlines()[0].removeprefix("metrics: ")),
            {"mape": 1.5, "bias": -0.25},
        )

    def test_features_are_snapshotted_not_aliased(self) -> None:
        # Reporting must not be able to observe a later mutation of the list
        # `train.py` passed in.
        declared = ["a", "b"]
        out = _capture(features=declared)
        declared.append("junk")
        self.assertEqual(json.loads(out["meta"])["features"], ["a", "b"])


def _instant() -> None:
    """Returns immediately. Module-level so a spawned child can import it."""


def _far_over_budget() -> None:
    import time

    time.sleep(120)


class RunWithBudgetTests(unittest.TestCase):
    def test_completes_within_budget(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            run_with_budget(_instant, 60)
        self.assertNotIn("timed out", buffer.getvalue())

    def test_over_budget_run_is_killed_not_merely_reported(self) -> None:
        # The budget has to bound the wall clock: `program.md` counts an
        # over-budget run as a crash, which it is not if the run keeps going
        # after the log says it was skipped. 120s of work, 0.5s of budget —
        # if this returns promptly, the child was actually killed.
        buffer = io.StringIO()
        started = time.monotonic()
        with redirect_stdout(buffer):
            run_with_budget(_far_over_budget, 0.5)
        elapsed = time.monotonic() - started
        self.assertIn("timed out after 0.5s", buffer.getvalue())
        self.assertLess(elapsed, 30)

    def test_budget_defaults_to_the_fixed_one(self) -> None:
        # `train.py` calls `run_with_budget(train_model)` with no number, so
        # the budget cannot be changed from the file the agent edits.
        import inspect

        import harness

        default = inspect.signature(harness.run_with_budget).parameters["seconds"]
        self.assertEqual(default.default, harness.TIME_BUDGET_SECONDS)


if __name__ == "__main__":
    unittest.main()

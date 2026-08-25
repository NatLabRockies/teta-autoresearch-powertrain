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
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

from harness import (
    evaluate,
    report,
    rmse,
    run_with_budget,
    train_test_split,
    trip_rmse,
)


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

        result = evaluate(actual, predicted, journey_id=journey_id, miles=miles)

        self.assertEqual(set(result), {"rmse", "trip_rmse"})
        self.assertAlmostEqual(result["rmse"], rmse(actual, predicted))
        self.assertAlmostEqual(
            result["trip_rmse"], trip_rmse(actual, predicted, journey_id, miles)
        )

    def test_journey_id_and_miles_are_keyword_only(self) -> None:
        """Positional misuse must fail loudly rather than silently mis-score."""
        arr = np.array([1.0, 2.0])
        with self.assertRaises(TypeError):
            evaluate(arr, arr, arr, arr)  # type: ignore[misc]


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

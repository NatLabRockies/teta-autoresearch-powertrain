"""The fixed point (do not modify).

Everything in `train.py` is fair game. Everything here is not, and the split
between the two files is the whole experimental control:

- **The time budget.** Every experiment gets the same wall clock, and
  `run_with_budget` enforces it by killing the run rather than by asking.
- **The split.** The same rows are held out in every experiment, so a result
  is comparable to the one before it.
- **The metrics.** `evaluate()` defines what "better" means. If the thing
  being optimized could be edited by the thing doing the optimizing, a
  session would be free to improve its score by redefining the score. This
  covers the physical plausibility checks in `physics.py` too: they run on
  every call, and `evaluate()` takes the model's `predict` as a required
  argument so that a run cannot quietly decline to be checked.
- **The report format.** `report()` is the only thing that prints the
  `metrics:` and `meta:` lines, so what lands in the results files cannot
  drift no matter what `train.py` becomes.

`train.py` owns the model: what it is, which features it uses, how it is fit.
It reports by calling `report()` and runs by calling `run_with_budget()`.
"""

from __future__ import annotations

import json
import multiprocessing
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

import physics

# ---------------------------------------------------------------------------
# Time budget (fixed, do not modify)
# ---------------------------------------------------------------------------

# Wall clock allowed for one run. Lives here rather than in `train.py` so an
# experiment cannot buy an improvement by giving itself more time.
TIME_BUDGET_SECONDS = 10 * 60

# How long a timed-out run gets to shut down cleanly before it is killed.
_TERMINATE_GRACE_SECONDS = 5


# ---------------------------------------------------------------------------
# Fixed Data utilities (do not modify)
# ---------------------------------------------------------------------------


def train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train/test sets."""
    rng = np.random.default_rng(random_seed)
    mask = rng.random(len(df)) < test_size
    test_df = df[mask].reset_index(drop=True)
    train_df = df[~mask].reset_index(drop=True)
    return train_df, test_df


# ---------------------------------------------------------------------------
# Evaluation Metrics (fixed, do not modify)
# ---------------------------------------------------------------------------
#


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Link-level root mean squared error on energy_rate_gge (GGE/mile)."""
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def trip_rmse(
    actual: np.ndarray,
    predicted: np.ndarray,
    journey_id: np.ndarray,
    miles: np.ndarray,
) -> float:
    """Trip total-energy RMSE in GGE.

    For each trip j: trip_gge = sum_i(rate_i * miles_i) over the links of j.
    Returns sqrt(mean((actual_trip_gge - pred_trip_gge)^2)) across trips.
    """
    df = pd.DataFrame(
        {
            "journey_id": journey_id,
            "actual_gge": np.asarray(actual) * np.asarray(miles),
            "pred_gge": np.asarray(predicted) * np.asarray(miles),
        }
    )
    trip = df.groupby("journey_id", sort=False)[["actual_gge", "pred_gge"]].sum()
    diff = trip["actual_gge"].to_numpy() - trip["pred_gge"].to_numpy()
    return float(np.sqrt(np.mean(diff**2)))


def evaluate(
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    journey_id: np.ndarray,
    miles: np.ndarray,
    predict: physics.Predict,
) -> dict[str, float]:
    """Evaluate a trained model: accuracy on held-out data, plus physics.

    `journey_id` and `miles` must align 1:1 with `actual` / `predicted`.

    `predict` is how the model answers a question about a link that is not in
    the test set. `physics.check_physics` calls it on a synthetic sweep — every
    combination of speed, grade and length out to 5 miles — to ask whether the
    learned function is physically possible and whether it still behaves on
    links ten times longer than any it was trained on. Accuracy alone cannot
    see either failure: both live entirely outside the data.

    It takes a DataFrame of synthetic links (see `physics.build_frame`) and
    returns `energy_rate_gge` for each, in row order — for most models a
    one-liner over whatever feature list `train.py` already defines:

        predict=lambda df: model.predict(df[LINK_FEATURES])

    The detailed check-by-check report goes to stderr rather than stdout, so
    `report()` remains the only thing writing the two parse-target lines while
    the diagnosis of a failure still lands in `run.log`.
    """
    report = physics.check_physics(predict)
    print(report.format(), file=sys.stderr)
    return {
        "rmse": rmse(actual, predicted),
        "trip_rmse": trip_rmse(actual, predicted, journey_id, miles),
        **report.metrics(),
    }


# ---------------------------------------------------------------------------
# Reporting and running (fixed, do not modify)
# ---------------------------------------------------------------------------


def report(
    metrics: Mapping[str, float],
    *,
    model_family: str,
    features: Sequence[str],
    total_seconds: float,
) -> None:
    """Print a run's result in the format `program.md` parses.

    Exactly two lines, each a single JSON object, each fact appearing once:

    - `metrics:` — exactly what `evaluate()` returned. Metric names are never
      hardcoded here; whatever keys are handed in are what gets reported.
    - `meta:` — everything about the run that is not a metric. `model_family`
      is a coarse label for the kind of model ("RandomForest", "MLP", "CNN",
      "GRU", "Linear"); it is free text, and it exists so results can be
      grouped afterwards rather than to gate anything.

    Nothing is restated in a second, human-friendlier form. A per-metric
    `rmse: 0.039590` line next to a `metrics:` line holding the same number is
    two places for it to be read from and two places for it to disagree, and
    the JSON is legible enough as it is. Anything derivable is left out for
    the same reason — the feature count is `len(features)`.
    """
    print(f"metrics: {json.dumps({k: round(v, 6) for k, v in metrics.items()})}")
    meta = {
        "model_family": model_family,
        "features": list(features),
        "total_seconds": round(total_seconds, 1),
    }
    print(f"meta: {json.dumps(meta)}")


def run_with_budget(
    fn: Callable[[], Any], seconds: float = TIME_BUDGET_SECONDS
) -> None:
    """Run `fn` in a worker process, killing it if it outlasts the budget.

    A separate process, not a thread: a training loop holding the GIL cannot
    be interrupted, and `program.md` counts an over-budget run as a crash — so
    the budget has to actually bound the wall clock rather than just print a
    line about it while the run continues to completion.

    `seconds` defaults to `TIME_BUDGET_SECONDS` and `train.py` should not pass
    it; the argument exists so the tests can exercise a timeout in under a
    second.

    `fn` runs in a child process, so it must be importable — a module-level
    function, not a closure or a lambda — and its return value is discarded.
    Report from inside `fn`; the child inherits stdout.
    """
    process = multiprocessing.Process(target=fn)
    process.start()
    process.join(seconds)
    if not process.is_alive():
        return
    print(f"\ntimed out after {seconds}s, skipping")
    # SIGTERM first so the child can unwind, then SIGKILL if it will not go.
    process.terminate()
    process.join(_TERMINATE_GRACE_SECONDS)
    if process.is_alive():
        process.kill()
        process.join()

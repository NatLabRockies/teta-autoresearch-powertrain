import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from harness import (
    evaluate,
    report,
    run_with_budget,
    train_test_split,
)

# --- shared defaults ---
# A coarse label for the kind of model below — "RandomForest", "MLP", "CNN",
# "GRU", "Linear". Free text, recorded with every result so experiments can be
# grouped by family afterwards. Keep it matched to what `build_model` returns.
MODEL_FAMILY = "GBDT"

LINK_FEATURES = [
    "speed_mph",
    "grade_percent",
    "miles",
    "dke_per_mile",
    "dke_in_link",
    "gap_seconds",
    "prev_miles",
    "next_miles",
    "next_gap_seconds",
    "prev2_speed",
    "dv_out",
]

TARGET = "energy_rate_gge"

# --- data config ---
DATA_PATH = "data/processed/2017_Chevy_Bolt.parquet"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive link features that need the journey's link sequence.

    `dke_per_mile` is the specific kinetic-energy change across a link,
    (v_out^2 - v_in^2) / distance, using the neighbouring links' speeds as the
    entry/exit speeds. It is the acceleration term of the energy balance, which
    nothing in the instantaneous (speed, grade, length) triple can express.
    A trip begins and ends at rest, so the missing neighbour at each end is
    filled with a speed of 0 rather than imputed.

    `dke_in_link` is the first half of that change, from the entry speed up to
    the link's own speed. Together with the total it locates where inside the
    link the speed changed, which matters because accelerating and regenerating
    are not the same price: a link that speeds up then slows down burns energy
    even when its net kinetic-energy change is zero.

    `gap_seconds` is the idle time between the previous link ending and this one
    starting. A non-zero gap means the vehicle actually came to a stop, so the
    entry speed was really 0 and the whole entry kinetic energy had to be
    rebuilt -- something the previous link's *average* speed cannot reveal.

    `prev_miles` is how long the previous link was. It sets how much road the
    entry speed change had to happen over: the same entry speed reached across
    a mile and across fifty feet are different accelerations, and the energy
    cost of a speed change depends on the distance it is spread across.
    `next_miles` is the same quantity for the exit side. `next_gap_seconds` is
    the exit-side mirror of `gap_seconds`: a stop *after* this link means the
    vehicle braked all the way to rest on it, so the exit speed was really 0.

    `prev2_speed` is the speed two links back. It describes the *approach
    profile* -- whether the vehicle was already slowing for a while or arrived
    at speed -- which the immediate neighbour cannot say. Its exit-side mirror
    is worth four times less, the same entry/exit asymmetry seen throughout.

    `dv_out` is the exit speed change on a raw scale rather than per mile.
    Regenerative braking is limited by the motor, so how much of a deceleration
    can be recovered depends on the speed drop itself, not on the drop divided
    by the distance it happened over.
    """
    by_journey = df.groupby("journey_id", sort=False)["speed_mph"]
    v_in = by_journey.shift(1).fillna(0.0)
    v_out = by_journey.shift(-1).fillna(0.0)
    df["dke_per_mile"] = (v_out**2 - v_in**2) / df["miles"]
    df["dke_in_link"] = (df["speed_mph"] ** 2 - v_in**2) / df["miles"]
    df["prev2_speed"] = by_journey.shift(2)
    df["dv_out"] = v_out - df["speed_mph"]
    by_miles = df.groupby("journey_id", sort=False)["miles"]
    df["prev_miles"] = by_miles.shift(1)
    df["next_miles"] = by_miles.shift(-1)
    prev_end = df.groupby("journey_id", sort=False)["link_end_time"].shift(1)
    df["gap_seconds"] = df["link_start_time"] - prev_end
    next_start = df.groupby("journey_id", sort=False)["link_start_time"].shift(-1)
    df["next_gap_seconds"] = next_start - df["link_end_time"]
    return df


# Strength of the shrink toward zero, in links. A journey needs roughly this
# many training links before its own residual is trusted over no correction.
JOURNEY_PRIOR_LINKS = 20.0


def journey_offset(
    train_df: pd.DataFrame, test_df: pd.DataFrame, residual: np.ndarray
) -> np.ndarray:
    """Per-journey correction, estimated from the model's training residuals.

    Two trips over the same roads do not cost the same energy: ambient
    temperature, cabin heating, payload and how hard the driver pushes are all
    constant within a trip and invisible to any per-link feature. The harness
    splits rows rather than journeys, so every journey has training links whose
    labels are legitimately available.

    What those links should contribute is not their mean rate -- that is mostly
    the trip's road and speed mix, which the model already explains -- but their
    mean *residual*, which is what the per-link features could not account for.
    Shrinking toward zero by `JOURNEY_PRIOR_LINKS` keeps a journey with two
    training links from moving on noise, and leaves a journey with none alone.
    """
    by_journey = pd.Series(residual).groupby(train_df["journey_id"].to_numpy())
    total = by_journey.sum()
    count = by_journey.count()
    j = test_df["journey_id"]
    return (
        j.map(total).fillna(0.0).to_numpy()
        / (j.map(count).fillna(0.0).to_numpy() + JOURNEY_PRIOR_LINKS)
    )


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    # sort by journey and time
    df = df.sort_values(["journey_id", "link_start_time"])
    return add_features(df)


def build_model() -> HistGradientBoostingRegressor:
    model_params = {
        "max_iter": 2000,
        "learning_rate": 0.1,
        "max_leaf_nodes": 31,
        "early_stopping": False,
        "random_state": 52,
    }
    return HistGradientBoostingRegressor(**model_params)


def train_model() -> dict[str, float]:
    """Train and evaluate. Returns results dict."""
    t0 = time.time()

    df = load_data()
    train_df, test_df = train_test_split(df, test_size=0.2, random_seed=42)

    y_train = train_df[TARGET].to_numpy(dtype=np.float32)
    y_test = test_df[TARGET].to_numpy(dtype=np.float32)
    journey_id_te = test_df["journey_id"].to_numpy()
    miles_te = test_df["miles"].to_numpy(dtype=np.float32)

    model = build_model()
    model.fit(train_df[LINK_FEATURES], y_train)
    residual = y_train - model.predict(train_df[LINK_FEATURES])
    predicted = model.predict(test_df[LINK_FEATURES]) + journey_offset(
        train_df, test_df, residual
    )

    results = evaluate(y_test, predicted, journey_id=journey_id_te, miles=miles_te)

    report(
        results,
        model_family=MODEL_FAMILY,
        features=LINK_FEATURES,
        total_seconds=time.time() - t0,
    )

    return results


if __name__ == "__main__":
    run_with_budget(train_model)

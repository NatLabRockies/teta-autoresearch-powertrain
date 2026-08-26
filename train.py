import time

import numpy as np
import pandas as pd
import shapely
from sklearn.ensemble import RandomForestRegressor

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
MODEL_FAMILY = "RandomForest"

LINK_FEATURES = [
    "speed_mph",
    "grade_percent",
    "miles",
    "prev_speed_mph",
    "speed_delta",
    "ke_delta_per_mile",
    "sinuosity",
]

TARGET = "energy_rate_gge"

# --- data config ---
DATA_PATH = "data/processed/2017_Chevy_Bolt.parquet"


EARTH_RADIUS_MILES = 3958.7613


def straight_line_miles(geometry: pd.Series) -> np.ndarray:
    """Great-circle distance between each linestring's two endpoints."""
    geoms = shapely.from_wkb(geometry.to_numpy())
    coords = shapely.get_coordinates(geoms)
    n_points = shapely.get_num_coordinates(geoms)
    end = np.cumsum(n_points) - 1
    start = end - n_points + 1

    lon1, lat1 = np.radians(coords[start, 0]), np.radians(coords[start, 1])
    lon2, lat2 = np.radians(coords[end, 0]), np.radians(coords[end, 1])
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2.0 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(a))


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    # sort by journey and time
    df = df.sort_values(["journey_id", "link_start_time"])
    # One-link lookback: the speed on the preceding link of the same journey.
    # domain.md allows exactly one backward link and no knowledge of the next.
    # The first link of a journey has no predecessor and is filled with 0 --
    # the vehicle starts from rest, so that is the physically true value.
    df["prev_speed_mph"] = df.groupby("journey_id")["speed_mph"].shift(1).fillna(0.0)
    # Explicit acceleration proxy. A tree splits on one axis at a time and so
    # cannot express speed_mph - prev_speed_mph from the two columns alone.
    df["speed_delta"] = df["speed_mph"] - df["prev_speed_mph"]
    # Specific kinetic energy change per unit distance, (v^2 - v_prev^2)/(2d).
    # This is the physics form of speed_delta: it carries the same units as the
    # target (energy per mile), so it should map onto the target more directly
    # than a raw speed difference does.
    df["ke_delta_per_mile"] = (
        df["speed_mph"] ** 2 - df["prev_speed_mph"] ** 2
    ) / (2.0 * df["miles"])
    # Geometry: how much longer the link is than the straight line between its
    # endpoints. A curvier link forces cornering at a given average speed, and
    # the value is a static road attribute so it is free at inference time.
    # Loop links (start == end) would divide by zero, so the floor is the
    # smallest link length in the data rather than an arbitrary epsilon.
    df["sinuosity"] = df["miles"] / np.maximum(
        straight_line_miles(df["geometry"]), 1e-4
    )
    return df


def build_model() -> RandomForestRegressor:
    model_params = {
        "n_estimators": 20,
        "max_depth": 10,
        "min_samples_split": 10,
        "random_state": 52,
        "n_jobs": -1,  # use all cores
    }
    return RandomForestRegressor(**model_params)


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
    predicted = model.predict(test_df[LINK_FEATURES])

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

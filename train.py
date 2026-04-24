import time
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, TimeoutError

from lightgbm import LGBMRegressor
from shapely import from_wkb


from fixed_utils import (
    evaluate,
    train_test_split,
)

# --- shared defaults ---
TIME_BUDGET_SECONDS = 10 * 60

LINK_FEATURES = [
    "speed_mph",
    "grade_percent",
    "miles",
    "prev_speed_mph",
    "prev_grade_percent",
    "speed_delta",
    "grade_delta",
    "speed_times_grade",
    "grade_times_miles",
    "prev_miles",
    "prev2_speed_mph",
    "prev3_speed_mph",
    "prev_speed_delta",
    "prev2_grade_percent",
    "prev2_miles",
    "abs_bearing_delta",
    "prev_abs_bearing_delta",
    "sinuosity",
]

TARGET = "energy_rate_gge"

# --- data config ---
# Registry of powertrain types. Each session targets exactly one.
POWERTRAINS = {
    "bev": {
        "name": "2017_Chevy_Bolt",
        "data_path": "data/processed/2017_Chevy_Bolt.parquet",
        "energy_type": "bev",
    },
    "ice": {
        "name": "2016_Toyota_Camry",
        "data_path": "data/processed/2016_Toyota_Camry.parquet",
        "energy_type": "ice",
    },
    "phev": {
        # Filled in when PHEV raw data lands in data/processed/.
        "name": "TBD_PHEV",
        "data_path": "data/processed/TBD_PHEV.parquet",
        "energy_type": "phev",
    },
}

POWERTRAIN = "bev"  # session selector — the only domain-related line to change
CONFIG = POWERTRAINS[POWERTRAIN]


def make_model():
    return LGBMRegressor(
        num_leaves=255,
        n_estimators=500,
        min_child_samples=50,
        learning_rate=0.05,
        random_state=52,
        verbose=-1,
    )


def train_model() -> dict:
    """Train and evaluate. Returns results dict."""
    t0 = time.time()

    # data
    df = pd.read_parquet(CONFIG["data_path"])

    # sort by journey and time
    df = df.sort_values(["journey_id", "link_start_time"])

    df["prev_speed_mph"] = (
        df.groupby("journey_id")["speed_mph"].shift(1).fillna(df["speed_mph"])
    )
    df["prev_grade_percent"] = (
        df.groupby("journey_id")["grade_percent"].shift(1).fillna(df["grade_percent"])
    )
    df["speed_delta"] = df["speed_mph"] - df["prev_speed_mph"]
    df["grade_delta"] = df["grade_percent"] - df["prev_grade_percent"]
    df["speed_times_grade"] = df["speed_mph"] * df["grade_percent"]
    df["grade_times_miles"] = df["grade_percent"] * df["miles"]
    df["prev_miles"] = df.groupby("journey_id")["miles"].shift(1).fillna(df["miles"])
    df["prev2_speed_mph"] = (
        df.groupby("journey_id")["speed_mph"].shift(2).fillna(df["speed_mph"])
    )
    df["prev3_speed_mph"] = (
        df.groupby("journey_id")["speed_mph"].shift(3).fillna(df["speed_mph"])
    )
    df["prev_speed_delta"] = df["prev_speed_mph"] - df["prev2_speed_mph"]
    df["prev2_grade_percent"] = (
        df.groupby("journey_id")["grade_percent"].shift(2).fillna(df["grade_percent"])
    )
    df["prev2_miles"] = df.groupby("journey_id")["miles"].shift(2).fillna(df["miles"])

    def _bearing_and_sinuosity(hex_str):
        g = from_wkb(bytes.fromhex(hex_str))
        coords = list(g.coords)
        if len(coords) < 2:
            return 0.0, 1.0
        lon1, lat1 = coords[0]
        lon2, lat2 = coords[-1]
        bearing = np.degrees(np.arctan2(lon2 - lon1, lat2 - lat1))
        chord = ((lon2 - lon1) ** 2 + (lat2 - lat1) ** 2) ** 0.5
        sinuosity = g.length / chord if chord > 1e-9 else 1.0
        return bearing, sinuosity

    _bs = df["geometry"].apply(_bearing_and_sinuosity)
    df["_bearing"] = _bs.apply(lambda t: t[0])
    df["sinuosity"] = _bs.apply(lambda t: t[1])
    prev_bearing = df.groupby("journey_id")["_bearing"].shift(1).fillna(df["_bearing"])
    delta = (df["_bearing"] - prev_bearing + 180) % 360 - 180
    df["abs_bearing_delta"] = delta.abs()
    df["prev_abs_bearing_delta"] = (
        df.groupby("journey_id")["abs_bearing_delta"]
        .shift(1)
        .fillna(df["abs_bearing_delta"])
    )

    train_df, test_df = train_test_split(df, test_size=0.2, random_seed=42)

    y_train = train_df[TARGET].to_numpy(dtype=np.float32)
    y_test = test_df[TARGET].to_numpy(dtype=np.float32)

    model = make_model()
    model.fit(train_df[LINK_FEATURES], y_train)
    predicted = model.predict(test_df[LINK_FEATURES])

    results = evaluate(y_test, predicted)

    for k, v in results.items():
        print(f"{k}: {v:.6f}")

    # meta
    total_seconds = time.time() - t0
    print(f"total_seconds: {total_seconds:.1f}")
    print(f"features: {','.join(LINK_FEATURES)}")

    return results


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(train_model)
        try:
            future.result(timeout=TIME_BUDGET_SECONDS)
        except TimeoutError:
            print(
                f"\n{CONFIG['name']}: timed out after {TIME_BUDGET_SECONDS}s, skipping"
            )
            executor.shutdown(wait=False, cancel_futures=True)

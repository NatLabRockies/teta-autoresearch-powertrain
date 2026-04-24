import time
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, TimeoutError

from sklearn.ensemble import HistGradientBoostingRegressor


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
    return HistGradientBoostingRegressor(max_iter=1000, random_state=52)


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

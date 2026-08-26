import time

import numpy as np
import pandas as pd
import shapely
import torch
from torch import nn

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
MODEL_FAMILY = "MLP"

LINK_FEATURES = [
    "speed_mph",
    "grade_percent",
    "miles",
    "prev_speed_mph",
    "ke_delta_per_mile",
    "sinuosity",
    "junction_turn_degrees",
    "prev_grade_percent",
    "prev_miles",
    "prev_sinuosity",
]

TARGET = "energy_rate_gge"

# --- data config ---
DATA_PATH = "data/processed/2017_Chevy_Bolt.parquet"

EARTH_RADIUS_MILES = 3958.7613

# --- training config ---
HIDDEN = 256
BATCH_SIZE = 4096
LEARNING_RATE = 1e-3
MAX_EPOCHS = 200
# Wall clock the fitting loop may use. The harness kills the process at its own
# budget and counts that as a crash, so the loop stops itself with room to
# spare for data loading and scoring.
TRAIN_SECONDS = 400
SEED = 52


def _heading(coords: np.ndarray, tail: np.ndarray, head: np.ndarray) -> np.ndarray:
    """Compass-plane bearing of each tail -> head segment, in radians."""
    mean_lat = np.radians((coords[head, 1] + coords[tail, 1]) / 2)
    east = (coords[head, 0] - coords[tail, 0]) * np.cos(mean_lat)
    north = coords[head, 1] - coords[tail, 1]
    return np.arctan2(north, east)


def geometry_features(geometry: pd.Series) -> dict[str, np.ndarray]:
    """Per-link shape values, decoding the WKB once and deriving all of them."""
    geoms = shapely.from_wkb(geometry.to_numpy())
    coords = shapely.get_coordinates(geoms)
    n_points = shapely.get_num_coordinates(geoms)
    end = np.cumsum(n_points) - 1
    start = end - n_points + 1

    # Great-circle distance between each linestring's two endpoints.
    lon1, lat1 = np.radians(coords[start, 0]), np.radians(coords[start, 1])
    lon2, lat2 = np.radians(coords[end, 0]), np.radians(coords[end, 1])
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return {
        "endpoint_miles": 2.0 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(a)),
        # Bearings of the first and last segments, so the turn taken at the
        # junction between two consecutive links can be measured.
        "entry_heading": _heading(coords, start, start + 1),
        "exit_heading": _heading(coords, end - 1, end),
    }


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    # sort by journey and time
    df = df.sort_values(["journey_id", "link_start_time"])
    # One-link lookback: the speed on the preceding link of the same journey.
    # domain.md allows exactly one backward link and no knowledge of the next.
    # The first link of a journey has no predecessor and is filled with 0 --
    # the vehicle starts from rest, so that is the physically true value.
    df["prev_speed_mph"] = df.groupby("journey_id")["speed_mph"].shift(1).fillna(0.0)
    # The other half of the one-link lookback: the grade the vehicle was on as
    # it entered this link. A flat first link is the natural fill for a trip
    # start, matching the at-rest assumption used for prev_speed_mph.
    df["prev_grade_percent"] = (
        df.groupby("journey_id")["grade_percent"].shift(1).fillna(0.0)
    )
    # Third of the one-link lookback triple. A short preceding link means dense
    # urban context -- closely spaced intersections -- which changes how much
    # of this link is spent accelerating away from the last one.
    df["prev_miles"] = df.groupby("journey_id")["miles"].shift(1).fillna(0.0)
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
    geom = geometry_features(df["geometry"])
    df["sinuosity"] = df["miles"] / np.maximum(geom["endpoint_miles"], 1e-4)
    # Shape of the link the vehicle came off. A curvy preceding link means it
    # was already cornering on entry, which bears on how much of prev_speed_mph
    # actually carried through. A straight link, 1.0, is the fill for a start.
    df["prev_sinuosity"] = (
        df.groupby("journey_id")["sinuosity"].shift(1).fillna(1.0)
    )
    # How far the vehicle turned at the junction it entered this link through.
    # exp7 showed the bending *inside* a link is subsumed by sinuosity, but an
    # intersection turn is a different event: it is where a driver actually
    # brakes and reaccelerates. Uses the one-link lookback, so a trip's first
    # link has no junction and is filled with a straight-ahead 0 degrees.
    df["exit_heading"] = geom["exit_heading"]
    prev_exit = df.groupby("journey_id")["exit_heading"].shift(1)
    turn = geom["entry_heading"] - prev_exit.to_numpy()
    df["junction_turn_degrees"] = np.nan_to_num(
        np.abs(np.degrees(np.arctan2(np.sin(turn), np.cos(turn))))
    )
    return df


def build_model(n_features: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(n_features, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, 1),
    )


def train_model() -> dict[str, float]:
    """Train and evaluate. Returns results dict."""
    t0 = time.time()
    torch.manual_seed(SEED)

    df = load_data()
    train_df, test_df = train_test_split(df, test_size=0.2, random_seed=42)

    y_train = train_df[TARGET].to_numpy(dtype=np.float32)
    y_test = test_df[TARGET].to_numpy(dtype=np.float32)
    journey_id_te = test_df["journey_id"].to_numpy()
    miles_te = test_df["miles"].to_numpy(dtype=np.float32)

    x_train = train_df[LINK_FEATURES].to_numpy(dtype=np.float32)
    x_test = test_df[LINK_FEATURES].to_numpy(dtype=np.float32)

    # Standardize from training statistics only. Gradient descent needs inputs
    # on a common scale, and the target is ~1e-2 so it is scaled up too --
    # squared error on the standardized target is proportional to squared error
    # on the raw one, so this does not change what is being optimized.
    x_mean, x_std = x_train.mean(axis=0), x_train.std(axis=0)
    x_std[x_std == 0] = 1.0
    y_mean, y_std = float(y_train.mean()), float(y_train.std())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xt = torch.from_numpy((x_train - x_mean) / x_std).to(device)
    yt = torch.from_numpy((y_train - y_mean) / y_std).to(device).unsqueeze(1)
    xe = torch.from_numpy((x_test - x_mean) / x_std).to(device)

    model = build_model(len(LINK_FEATURES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Anneal the step size to zero over the run. A constant rate leaves the
    # weights oscillating around the minimum when the loop stops, and exp10
    # showed that bumpiness costs trip_rmse far more than it costs link rmse.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS
    )
    loss_fn = nn.MSELoss()

    n = xt.shape[0]
    t_train = time.time()
    for _ in range(MAX_EPOCHS):
        order = torch.randperm(n, device=device)
        for start in range(0, n, BATCH_SIZE):
            idx = order[start : start + BATCH_SIZE]
            optimizer.zero_grad()
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            optimizer.step()
        scheduler.step()
        if time.time() - t_train > TRAIN_SECONDS:
            break

    model.eval()
    with torch.no_grad():
        predicted = model(xe).squeeze(1).cpu().numpy() * y_std + y_mean

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

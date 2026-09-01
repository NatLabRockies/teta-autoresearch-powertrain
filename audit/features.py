"""Feature construction for both arms, plus the causal substitutes.

The as-run blocks are transcribed from each arm's final `train.py` (unguided
`bc07466`, domain-guided `e315384`) and must stay equivalent in behaviour — if they
drift, the audit stops reproducing the reported numbers and the whole comparison is
void. `audit_accuracy.py` checks that by reproducing both reported scores before it
does anything else.

Two things here are deliberately *not* imported from the trees. The model definitions
and the feature code are transcriptions, checked by the reproduction test rather than
by import, because importing from a tree would make the audit depend on the thing it
is auditing. The one file that IS copied verbatim is `harness.py`, which defines the
split and the metrics; byte-identity there is checkable with a single sha256 and is
what makes "same held-out rows, same metric" true rather than asserted.

The `causal_*` block is the audit's own work. For each feature the contract puts
outside the Compass envelope, it builds the best substitute a forward search could
actually compute. Nothing in that block reads a link the vehicle has not yet
traversed. Several features have no substitute at all — see `contract.py` on the
`TRACE` and `LABEL` tiers — and those are left to the training mean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shapely

DATA_PATH = "data/2017_Chevy_Bolt.parquet"

TARGET = "energy_rate_gge"

EARTH_RADIUS_MILES = 3958.7613


def load_sorted() -> pd.DataFrame:
    """Load and apply the sort both arms applied before splitting.

    The fixed split draws `rng.random(len(df)) < 0.2` against row *order*, so the
    sort is load-bearing: a different order is a different test set. Both arms sort
    by `["journey_id", "link_start_time"]`, so the audit does too.
    """
    df = pd.read_parquet(DATA_PATH)
    return df.sort_values(["journey_id", "link_start_time"]).reset_index(drop=True)


# --- unguided arm, as run (bc07466) --------------------------------------------


def add_unguided_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transcribed from unguided/train.py `add_features`. Do not 'improve'.

    Note the neighbour fills: a trip begins and ends at rest, so the missing
    neighbour speed at each end is 0 rather than imputed. `prev2_speed` is left as
    NaN where it does not exist — HistGradientBoosting handles NaN natively and the
    arm relied on that.
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

    # Kept for the causal substitutes below, which need the entry speed.
    df["_v_in"] = v_in
    return df


#: Shrink strength for the journey effect, in links. Transcribed from the arm.
JOURNEY_PRIOR_LINKS = 20.0


def add_journey_effect(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Transcribed from unguided/train.py `add_journey_effect`.

    This is the `LABEL`-tier pair. Both columns are built from the *target* of other
    links in the same journey: leave-one-out on the training side, plain shrunk mean
    on the test side. Reproduced exactly, because the audit's first job is to
    reproduce the arm's number — and then scored away, because at inference there are
    no labels to average.
    """
    stats = train_df.groupby("journey_id")[TARGET].agg(["sum", "count"])
    gge = train_df.groupby("journey_id")[["energy_gge", "miles"]].sum()
    prior = float(train_df[TARGET].mean())
    prior_gge = float(train_df["energy_gge"].sum() / train_df["miles"].sum())
    k = JOURNEY_PRIOR_LINKS
    k_miles = JOURNEY_PRIOR_LINKS * float(train_df["miles"].mean())

    own = train_df["journey_id"].map(stats["sum"]).to_numpy()
    n = train_df["journey_id"].map(stats["count"]).to_numpy()
    y = train_df[TARGET].to_numpy()
    train_df["journey_rate"] = (own - y + k * prior) / (n - 1 + k)
    e = train_df["energy_gge"].to_numpy()
    d = train_df["miles"].to_numpy()
    own_e = train_df["journey_id"].map(gge["energy_gge"]).to_numpy()
    own_d = train_df["journey_id"].map(gge["miles"]).to_numpy()
    train_df["journey_gge_per_mile"] = (own_e - e + k_miles * prior_gge) / (
        own_d - d + k_miles
    )

    own = test_df["journey_id"].map(stats["sum"]).fillna(0.0).to_numpy()
    n = test_df["journey_id"].map(stats["count"]).fillna(0.0).to_numpy()
    test_df["journey_rate"] = (own + k * prior) / (n + k)
    own_e = test_df["journey_id"].map(gge["energy_gge"]).fillna(0.0).to_numpy()
    own_d = test_df["journey_id"].map(gge["miles"]).fillna(0.0).to_numpy()
    test_df["journey_gge_per_mile"] = (own_e + k_miles * prior_gge) / (own_d + k_miles)


# --- domain-guided arm, as run (e315384) ---------------------------------------


def _heading(coords: np.ndarray, tail: np.ndarray, head: np.ndarray) -> np.ndarray:
    """Compass-plane bearing of each tail -> head segment, in radians."""
    mean_lat = np.radians((coords[head, 1] + coords[tail, 1]) / 2)
    east = (coords[head, 0] - coords[tail, 0]) * np.cos(mean_lat)
    north = coords[head, 1] - coords[tail, 1]
    return np.arctan2(north, east)


def _geometry_features(geometry: pd.Series) -> dict[str, np.ndarray]:
    """Transcribed from domain-guided/train.py `geometry_features`."""
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
    return {
        "endpoint_miles": 2.0 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(a)),
        "n_vertices": n_points.astype(np.float64),
        "entry_heading": _heading(coords, start, start + 1),
        "exit_heading": _heading(coords, end - 1, end),
    }


def add_domain_guided_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transcribed from domain-guided/train.py `load_data`. Do not 'improve'."""
    df["prev_speed_mph"] = df.groupby("journey_id")["speed_mph"].shift(1).fillna(0.0)
    df["prev_grade_percent"] = (
        df.groupby("journey_id")["grade_percent"].shift(1).fillna(0.0)
    )
    # NOTE: this arm's prev_miles fills with 0.0; the unguided arm leaves its own
    # prev_miles as NaN. Same name, different column — they are built separately and
    # never share storage. `build_all` keeps the unguided one and derives this arm's
    # under a private name.
    df["_dg_prev_miles"] = df.groupby("journey_id")["miles"].shift(1).fillna(0.0)
    df["ke_delta_per_mile"] = (df["speed_mph"] ** 2 - df["prev_speed_mph"] ** 2) / (
        2.0 * df["miles"]
    )

    geom = _geometry_features(df["geometry"])
    df["sinuosity"] = df["miles"] / np.maximum(geom["endpoint_miles"], 1e-4)
    df["vertices_per_mile"] = geom["n_vertices"] / df["miles"]
    df["prev_sinuosity"] = df.groupby("journey_id")["sinuosity"].shift(1).fillna(1.0)

    df["_exit_heading"] = geom["exit_heading"]
    prev_exit = df.groupby("journey_id")["_exit_heading"].shift(1)
    turn = geom["entry_heading"] - prev_exit.to_numpy()
    df["junction_turn_degrees"] = np.nan_to_num(
        np.abs(np.degrees(np.arctan2(np.sin(turn), np.cos(turn))))
    )
    return df


# --- causal substitutes (the audit's own work) ---------------------------------

#: Suffix marking a causal stand-in for an out-of-contract feature.
CAUSAL = "__causal"


def add_causal_substitutes(df: pd.DataFrame) -> pd.DataFrame:
    """Best causal value for each out-of-contract unguided feature.

    Every column here is computable by a forward search. None of them reads a link
    ahead of the vehicle. Requires `add_unguided_features` to have run.

    Only two of the eight out-of-contract features get one:

    - `dke_per_mile` — its causal half is `(speed**2 - v_in**2) / miles`, which is
      exactly `dke_in_link`, already in the model. So the substitute is real but
      buys nothing new: the model has it twice.
    - `prev2_speed` — already causal. The substitute is the column itself; what it
      costs is one float per search label, not accuracy.

    `next_miles`, `dv_out`, `gap_seconds`, `next_gap_seconds`, `journey_rate` and
    `journey_gge_per_mile` have none, for the reasons in `contract.py`. They go to
    the training mean in every audited configuration. That is not the audit being
    lazy — it is the finding.
    """
    df[f"dke_per_mile{CAUSAL}"] = (df["speed_mph"] ** 2 - df["_v_in"] ** 2) / df[
        "miles"
    ]
    df[f"prev2_speed{CAUSAL}"] = df["prev2_speed"]
    return df


def build_all(df: pd.DataFrame) -> pd.DataFrame:
    """Every column any audit configuration needs, on one frame."""
    df = add_unguided_features(df)
    df = add_domain_guided_features(df)
    df = add_causal_substitutes(df)
    return df


#: Columns the domain-guided model reads, resolved to the actual frame columns.
#: `prev_miles` is the one name both arms use for different fills.
def domain_guided_matrix(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    cols = [("_dg_prev_miles" if f == "prev_miles" else f) for f in features]
    return df[cols].to_numpy(dtype=np.float64)

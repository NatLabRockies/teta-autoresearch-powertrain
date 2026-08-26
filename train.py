import math
import time

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
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
MODEL_FAMILY = "GBDT+CNN"

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
    "journey_rate",
    "journey_gge_per_mile",
]

# The sequence model reads the raw link chain and works out trip context for
# itself, so it is given the per-link features only. The journey columns are
# deliberately withheld: their leave-one-out form is safe for a model that sees
# one row at a time and a leak for one that sees a whole journey at once (see
# `sequence_predictions`).
SEQUENCE_FEATURES = [f for f in LINK_FEATURES if not f.startswith("journey_")]

TARGET = "energy_rate_gge"

# --- sequence member ---
CHANNELS = 64
DILATIONS = (1, 2)
KERNEL = 5
SEQ_LR = 3e-3
SEQ_SECONDS = 60.0
# Elapsed time by which sequence training must stop, whatever the trees took.
SEQ_DEADLINE = 545.0
MAX_TOKENS_PER_BATCH = 32768
# Weight on the sequence member in the blend. Its errors correlate 0.72 with
# the trees', so a minority weight removes variance neither model can remove
# alone; past ~0.35 its own higher error starts to dominate.
BLEND = 0.30

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


# Strength of the shrink toward the global mean, in links. A journey needs
# roughly this many training links before its own mean is trusted over the
# population mean.
JOURNEY_PRIOR_LINKS = 12.0


def add_journey_effect(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Give every link its own journey's mean rate, from training links only.

    Two trips over the same roads do not cost the same energy: ambient
    temperature, cabin heating, payload and how hard the driver pushes are all
    constant within a trip and invisible to any per-link feature. The harness
    splits rows rather than journeys, so each journey has training links whose
    labels are legitimately available, and their mean is a direct estimate of
    that trip-constant offset.

    A training link would otherwise see its own target, so its own contribution
    is removed leave-one-out. Both sides shrink toward the global mean, which
    also covers a test link whose journey has no training links at all.

    `journey_gge_per_mile` is the same offset weighted by distance -- the
    journey's training energy divided by its training miles. It is the quantity
    the trip metric actually sums, and it down-weights the very short links
    whose rate is a ratio with a tiny denominator.
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


def out_of_fold_residual(train_df: pd.DataFrame, y_train: np.ndarray) -> np.ndarray:
    """Training residuals from a model that did not see the row it is scoring.

    In-sample residuals are shrunk toward zero -- the model has already fitted
    part of the trip effect it is being asked to measure -- so a journey offset
    built from them under-states the correction. Two folds, each scored by the
    model fitted on the other, give an honest one.
    """
    fold = np.arange(len(train_df)) % 2
    residual = np.empty(len(train_df), dtype=np.float64)
    for held_out in (0, 1):
        fit = fold != held_out
        model = Ensemble()
        model.fit(train_df.loc[fit, LINK_FEATURES], y_train[fit])
        rows = ~fit
        residual[rows] = y_train[rows] - model.predict(
            train_df.loc[rows, LINK_FEATURES]
        )
    return residual


def journey_offset(
    train_df: pd.DataFrame, test_df: pd.DataFrame, residual: np.ndarray
) -> np.ndarray:
    """Per-journey correction for whatever the fitted model still gets wrong.

    The journey features above hand the model the trip's mean rate, but a
    boosted tree can only use it as one more split variable; it cannot apply it
    as an exact per-trip shift. The mean training residual of a journey is that
    leftover shift, measured after the model has done its best. Shrinking toward
    zero keeps a journey with two training links from moving on noise, and
    leaves a journey with none alone.

    The mean is taken over distance rather than over links: total residual
    energy divided by total residual miles. A trip's leftover is a quantity of
    energy, and a hundred-foot link's rate is a ratio with a hundred-foot
    denominator -- averaging links equally lets the noisiest of them set the
    correction for the whole trip.
    """
    keys = train_df["journey_id"].to_numpy()
    miles = train_df["miles"].to_numpy()
    energy = pd.Series(residual * miles).groupby(keys).sum()
    distance = pd.Series(miles).groupby(keys).sum()
    floor = JOURNEY_PRIOR_LINKS * float(miles.mean())
    j = test_df["journey_id"]
    return j.map(energy).fillna(0.0).to_numpy() / (
        j.map(distance).fillna(0.0).to_numpy() + floor
    )


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    # sort by journey and time
    df = df.sort_values(["journey_id", "link_start_time"])
    return add_features(df)


# How many boosted models to average, and how much of the feature set each
# split of each of them may look at.
N_ENSEMBLE = 5
FEATURE_SUBSAMPLE = 0.7


def build_model(seed: int = 52) -> HistGradientBoostingRegressor:
    model_params = {
        "max_iter": 2000,
        "learning_rate": 0.1,
        "max_leaf_nodes": 31,
        "max_features": FEATURE_SUBSAMPLE,
        "early_stopping": False,
        "random_state": seed,
    }
    return HistGradientBoostingRegressor(**model_params)


class Block(nn.Module):
    """Residual dilated-convolution block over the link axis."""

    def __init__(self, dilation: int) -> None:
        super().__init__()
        pad = dilation * (KERNEL - 1) // 2
        self.conv1 = nn.Conv1d(
            CHANNELS, CHANNELS, KERNEL, padding=pad, dilation=dilation
        )
        self.conv2 = nn.Conv1d(CHANNELS, CHANNELS, 1)
        self.norm = nn.GroupNorm(1, CHANNELS)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.act(self.conv1(self.norm(x))))


class LinkSequenceNet(nn.Module):
    """Reads a whole journey and predicts every link's rate at once.

    Stacked dilated convolutions reach about thirty links each way, so this
    member sees the trip context the trees only get through hand-cut +/-2
    neighbour features. On its own it is ~8% worse than the trees; what makes
    it worth carrying is that its errors correlate only 0.72 with theirs.
    """

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.inp = nn.Conv1d(n_features, CHANNELS, 1)
        self.blocks = nn.ModuleList(Block(d) for d in DILATIONS)
        self.out = nn.Conv1d(CHANNELS, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x)
        for block in self.blocks:
            h = block(h)
        return self.out(h).squeeze(1)


def journey_bounds(journey_id: np.ndarray) -> np.ndarray:
    """Start/end row of every journey, which is contiguous after the sort."""
    starts = np.flatnonzero(np.r_[True, journey_id[1:] != journey_id[:-1]])
    return np.c_[starts, np.r_[starts[1:], len(journey_id)]]


def pack_journeys(
    df: pd.DataFrame, x: np.ndarray, y: np.ndarray, w: np.ndarray, device: torch.device
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]]:
    """Batch journeys of similar length into [batch, features, links] tensors."""
    bounds = journey_bounds(df["journey_id"].to_numpy())
    lengths = bounds[:, 1] - bounds[:, 0]
    groups: list[list[int]] = []
    current: list[int] = []
    longest = 0
    for j in np.argsort(lengths, kind="stable"):
        longest = max(longest, int(lengths[j]))
        if current and longest * (len(current) + 1) > MAX_TOKENS_PER_BATCH:
            groups.append(current)
            current, longest = [], int(lengths[j])
        current.append(int(j))
    groups.append(current)

    packed = []
    for group in groups:
        spans = [(int(bounds[j, 0]), int(bounds[j, 1])) for j in group]
        width = max(b - a for a, b in spans)
        xb = np.zeros((len(spans), width, x.shape[1]), dtype=np.float32)
        yb = np.zeros((len(spans), width), dtype=np.float32)
        wb = np.zeros((len(spans), width), dtype=np.float32)
        rows = np.full((len(spans), width), -1, dtype=np.int64)
        for i, (a, b) in enumerate(spans):
            xb[i, : b - a] = x[a:b]
            yb[i, : b - a] = y[a:b]
            wb[i, : b - a] = w[a:b]
            rows[i, : b - a] = np.arange(a, b)
        packed.append(
            (
                torch.from_numpy(xb).to(device).transpose(1, 2),
                torch.from_numpy(yb).to(device),
                torch.from_numpy(wb).to(device),
                rows,
            )
        )
    return packed


def sequence_predictions(
    df: pd.DataFrame, train_rows: np.ndarray, run_started: float
) -> np.ndarray:
    """Fit the sequence member and predict every link of every journey.

    Test links are fed in as context but never scored -- the loss is masked to
    training links, exactly as the neighbour features already read a test link's
    speed without reading its target.

    The journey columns are withheld here on purpose. Their leave-one-out form
    differs between two links of the same journey by a term in those links' own
    targets, so a model that sees a whole journey at once can difference two of
    them and read a target straight out. A row-at-a-time tree cannot; this net
    can, and does -- it cut its training loss by a third and made its test error
    75% worse. Given the raw chain instead it matches what it scored with the
    journey columns present, because it works the trip context out itself.
    """
    y = df[TARGET].to_numpy(dtype=np.float32)
    w = np.zeros(len(df), dtype=np.float32)
    w[train_rows] = 1.0
    is_train = w > 0

    x = df[SEQUENCE_FEATURES].to_numpy(dtype=np.float32)
    x = (x - np.nanmean(x[is_train], axis=0)) / (np.nanstd(x[is_train], axis=0) + 1e-6)
    np.nan_to_num(x, copy=False)  # a missing neighbour becomes the mean link
    y_scale = float(y[is_train].std())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(52)
    packed = pack_journeys(df, x, y, w, device)
    net = LinkSequenceNet(x.shape[1]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=SEQ_LR, weight_decay=1e-4)

    # Whatever the trees left of the budget, capped at what this net can use
    # before it starts overfitting 15,247 journeys.
    budget = max(15.0, min(SEQ_SECONDS, run_started + SEQ_DEADLINE - time.time()))
    rng = np.random.default_rng(52)
    start = time.time()
    while time.time() - start < budget:
        for group in opt.param_groups:
            group["lr"] = (
                SEQ_LR * 0.5 * (1 + math.cos(math.pi * (time.time() - start) / budget))
            )
        for i in rng.permutation(len(packed)):
            xb, yb, wb, _ = packed[i]
            loss = ((net(xb) * y_scale - yb) ** 2 * wb).sum() / wb.sum().clamp(min=1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    out = np.zeros(len(df), dtype=np.float32)
    net.eval()
    with torch.no_grad():
        for xb, _, _, rows in packed:
            block = (net(xb) * y_scale).cpu().numpy()
            keep = rows >= 0
            out[rows[keep]] = block[keep]
    return out


class Ensemble:
    """Average several boosted models that each see part of the feature set.

    Exp12 showed the model sits just past its capacity optimum: extra rounds
    started fitting per-link noise. Averaging decorrelated fits removes variance
    without adding any, which is the one way left to spend the unused budget
    that does not also buy more overfitting. Feature subsampling is what makes
    the members differ -- the members are otherwise deterministic.
    """

    def __init__(self) -> None:
        self.members = [build_model(52 + i) for i in range(N_ENSEMBLE)]

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> "Ensemble":
        for member in self.members:
            member.fit(x, y)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.mean([member.predict(x) for member in self.members], axis=0)


def train_model() -> dict[str, float]:
    """Train and evaluate. Returns results dict."""
    t0 = time.time()

    df = load_data().reset_index(drop=True)
    # `_row` survives the split, so the sequence model can be handed the same
    # train/test mask in the journey-ordered frame it needs.
    df["_row"] = np.arange(len(df), dtype=np.int64)
    train_df, test_df = train_test_split(df, test_size=0.2, random_seed=42)
    add_journey_effect(train_df, test_df)

    y_train = train_df[TARGET].to_numpy(dtype=np.float32)
    y_test = test_df[TARGET].to_numpy(dtype=np.float32)
    journey_id_te = test_df["journey_id"].to_numpy()
    miles_te = test_df["miles"].to_numpy(dtype=np.float32)

    model = Ensemble()
    residual = out_of_fold_residual(train_df, y_train)
    model.fit(train_df[LINK_FEATURES], y_train)
    sequence = sequence_predictions(df, train_df["_row"].to_numpy(), t0)
    predicted = (
        (1 - BLEND) * model.predict(test_df[LINK_FEATURES])
        + BLEND * sequence[test_df["_row"].to_numpy()]
        + journey_offset(train_df, test_df, residual)
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

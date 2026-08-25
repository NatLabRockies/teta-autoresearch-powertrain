import math
import time

import numpy as np
import pandas as pd
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
MODEL_FAMILY = "CNN"

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

# --- model config ---
CHANNELS = 128
DILATIONS = (1, 2, 4, 8)
KERNEL = 5
LR = 3e-3
WEIGHT_DECAY = 1e-4
MAX_TOKENS_PER_BATCH = 32768
# Stop training here and spend what is left on inference and scoring. The
# harness kills the run at its own budget; this only reserves the tail.
TRAIN_SECONDS = 430.0


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


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    # sort by journey and time
    df = df.sort_values(["journey_id", "link_start_time"])
    return add_features(df).reset_index(drop=True)


class Block(nn.Module):
    """Residual dilated-convolution block over the link axis."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        pad = dilation * (KERNEL - 1) // 2
        self.conv1 = nn.Conv1d(
            channels, channels, KERNEL, padding=pad, dilation=dilation
        )
        self.conv2 = nn.Conv1d(channels, channels, 1)
        self.norm = nn.GroupNorm(1, channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.conv1(self.norm(x)))
        return x + self.conv2(h)


class LinkSequenceNet(nn.Module):
    """Reads a whole journey and predicts every link's energy rate at once.

    Every feature that has paid off so far is a hand-cut window onto the
    neighbouring links -- the entry speed, the gap before, the length of the
    link before. Stacked dilated convolutions are the general form of that: the
    receptive field here reaches about thirty links each way, and the model
    chooses what to read from it instead of being told.
    """

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.inp = nn.Conv1d(n_features, CHANNELS, 1)
        self.blocks = nn.ModuleList(Block(CHANNELS, d) for d in DILATIONS)
        self.out = nn.Conv1d(CHANNELS, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, features, links]
        h = self.inp(x)
        for block in self.blocks:
            h = block(h)
        return self.out(h).squeeze(1)


def build_model(n_features: int) -> LinkSequenceNet:
    return LinkSequenceNet(n_features)


def journey_bounds(journey_id: np.ndarray) -> np.ndarray:
    """Start/end row of every journey, which is contiguous after the sort."""
    starts = np.flatnonzero(np.r_[True, journey_id[1:] != journey_id[:-1]])
    return np.c_[starts, np.r_[starts[1:], len(journey_id)]]


def make_batches(bounds: np.ndarray) -> list[np.ndarray]:
    """Group journeys of similar length so padding stays cheap."""
    lengths = bounds[:, 1] - bounds[:, 0]
    order = np.argsort(lengths, kind="stable")
    batches: list[np.ndarray] = []
    current: list[int] = []
    longest = 0
    for j in order:
        longest = max(longest, int(lengths[j]))
        if current and longest * (len(current) + 1) > MAX_TOKENS_PER_BATCH:
            batches.append(np.array(current))
            current, longest = [], int(lengths[j])
        current.append(int(j))
    batches.append(np.array(current))
    return batches


def pad_batch(
    journeys: np.ndarray,
    bounds: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """Pack a set of journeys into [batch, features, links] with a weight mask."""
    spans = [(int(bounds[j, 0]), int(bounds[j, 1])) for j in journeys]
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
    return (
        torch.from_numpy(xb).to(device).transpose(1, 2),
        torch.from_numpy(yb).to(device),
        torch.from_numpy(wb).to(device),
        rows,
    )


def train_model() -> dict[str, float]:
    """Train and evaluate. Returns results dict."""
    t0 = time.time()

    df = load_data()
    # `_row` survives the split, so the harness's own mask can be recovered as
    # positions in the journey-ordered frame the sequence model needs.
    df["_row"] = np.arange(len(df), dtype=np.int64)
    train_df, test_df = train_test_split(df, test_size=0.2, random_seed=42)

    y_test = test_df[TARGET].to_numpy(dtype=np.float32)
    journey_id_te = test_df["journey_id"].to_numpy()
    miles_te = test_df["miles"].to_numpy(dtype=np.float32)

    y = df[TARGET].to_numpy(dtype=np.float32)
    weight = np.zeros(len(df), dtype=np.float32)
    weight[train_df["_row"].to_numpy()] = 1.0
    is_train = weight > 0

    # Standardise on the training links only. Test links are still fed to the
    # model as sequence context, exactly as the neighbour features already were,
    # but nothing about their targets touches the fit.
    x = df[LINK_FEATURES].to_numpy(dtype=np.float32)
    centre = np.nanmean(x[is_train], axis=0)
    scale = np.nanstd(x[is_train], axis=0) + 1e-6
    x = (x - centre) / scale
    np.nan_to_num(x, copy=False)  # a missing neighbour becomes the mean link
    y_scale = float(y[is_train].std())

    bounds = journey_bounds(df["journey_id"].to_numpy())
    batches = make_batches(bounds)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(52)
    model = build_model(len(LINK_FEATURES)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    packed = [pad_batch(b, bounds, x, y, weight, device) for b in batches]
    rng = np.random.default_rng(52)
    epoch = 0
    while time.time() - t0 < TRAIN_SECONDS:
        # Cosine decay over the time budget rather than over a fixed epoch count,
        # since how many epochs fit is a property of the machine, not the model.
        progress = min(1.0, (time.time() - t0) / TRAIN_SECONDS)
        for group in opt.param_groups:
            group["lr"] = LR * 0.5 * (1 + math.cos(math.pi * progress))
        for i in rng.permutation(len(packed)):
            xb, yb, wb, _ = packed[i]
            pred = model(xb) * y_scale
            loss = ((pred - yb) ** 2 * wb).sum() / wb.sum().clamp(min=1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        epoch += 1

    predicted_all = np.zeros(len(df), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for xb, _, _, rows in packed:
            out = (model(xb) * y_scale).cpu().numpy()
            keep = rows >= 0
            predicted_all[rows[keep]] = out[keep]
    predicted = predicted_all[test_df["_row"].to_numpy()]

    results = evaluate(y_test, predicted, journey_id=journey_id_te, miles=miles_te)

    print(f"epochs: {epoch}")
    report(
        results,
        model_family=MODEL_FAMILY,
        features=LINK_FEATURES,
        total_seconds=time.time() - t0,
    )

    return results


if __name__ == "__main__":
    run_with_budget(train_model)

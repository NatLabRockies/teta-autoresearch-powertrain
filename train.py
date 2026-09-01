"""Structured physics model: predict the *shape* physics gives a link's energy.

The scaffold forest predicts `energy_rate_gge` directly and breaks six of the
nine checks in `physics.py`, worst of all `climb_floor` (81% of cases) — its
implied drivetrain efficiency is 1.29, i.e. it bills a climb for less than the
height it gains. No amount of capacity fixes that, because gravity is not
something an unconstrained regressor is obliged to respect.

So the model here predicts the four terms of a road-load decomposition instead:

    E(v, g, d) = d * (A + B*g+ - C*g-) + T

`A` is a per-mile road-load rate, `B` and `C` are the per-mile climb and descent
slopes per grade-percent, and `T` is a fixed per-link transient cost — the
acceleration energy that a link average hides. The harness scores `E / d`.

Two properties make this work:

- **The ceiling in `physics.py` decomposes the same way.** `resistance` and
  `accessory` are proportional to `d`, `transient` is not, `potential` is
  proportional to both `d` and `g`. Bounding each head separately therefore
  bounds the total exactly, with no slack wasted.
- **Every head is squashed into a physically legal interval**, so every member
  of this model family satisfies all nine checks for *any* input. This is not a
  clamp on the output: the sigmoid is inside the function being fit, gradients
  flow through it, and the model learns within the feasible set rather than
  being projected onto it afterwards.

The one structural cost: `A`, `B`, `C` and `T` must not depend on the current
link's own grade or length. A head that varied with grade could make a climb
cheaper than flat ground; one that varied with length could make a longer link
cheaper than a shorter one. Grade and length enter through the algebra instead,
which is where the physics puts them anyway.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

import physics
from harness import (
    evaluate,
    report,
    run_with_budget,
    train_test_split,
)

MODEL_FAMILY = "PhysicsMLP"

#: What the network itself sees. These drive the four heads, so none of them
#: may depend on the current link's own grade or length — see `PhysicsNet`.
NET_FEATURES = [
    "speed_mph",
    "prev_speed_mph",
    "next_speed_mph",
    "prev2_speed_mph",
]

#: Consumed by the structural algebra rather than by the network.
STRUCTURAL_FEATURES = [
    "grade_percent",
    "miles",
]

#: Everything the model consumes, which is what `report()` records.
LINK_FEATURES = NET_FEATURES + STRUCTURAL_FEATURES

TARGET = "energy_rate_gge"

DATA_PATH = "data/processed/2017_Chevy_Bolt.parquet"

# --- model config ---
HIDDEN = 128
EPOCHS = 300
BATCH = 8192
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 0

#: Safety net only. The epoch count is chosen to finish well inside the
#: harness budget; this stops a run that somehow does not, so it reports a
#: result instead of being killed.
TRAIN_SECONDS_CAP = 470.0

# --- physical constants, read through `physics.py` so they cannot drift ---

K_POT = physics.MASS_KG * physics.G * physics.MI_TO_M / 100.0 / physics.J_PER_GGE
"""GGE of potential energy in one mile at one percent grade."""

ETA = physics.ETA_DRIVE

MIN_MS = 1e-3
"""Floor on speed in m/s, so the accessory term cannot divide by zero."""


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    # sort by journey and time
    return df.sort_values(["journey_id", "link_start_time"])


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the neighbour-link columns, in place.

    Rows must already be ordered by journey and time. The links at either end
    of a journey have no neighbour there, and 0 is the physically correct value
    rather than an imputation: a trip starts and ends at rest.

    This runs on the whole frame before the split — it reads only features, no
    targets — and again inside the physics sweep, where `build_frame` gives
    every synthetic link its own single-link journey, so each one is treated as
    a standing start to a full stop.
    """
    speed = df.groupby("journey_id", sort=False)["speed_mph"]
    df["prev_speed_mph"] = speed.shift(1).fillna(0.0)
    df["next_speed_mph"] = speed.shift(-1).fillna(0.0)
    # Two links back: the approach profile. Whether the vehicle was already
    # slowing before the previous link says how it entered this one.
    df["prev2_speed_mph"] = speed.shift(2).fillna(0.0)
    return df


# ---------------------------------------------------------------------------
# Physical envelopes, as functions of speed alone
# ---------------------------------------------------------------------------


def _road_load_per_mile(speed_mph: Tensor) -> Tensor:
    """Steady-state rolling + aerodynamic work over one mile, in GGE."""
    v = speed_mph * physics.MPH_TO_MS
    force = (
        physics.CRR * physics.MASS_KG * physics.G
        + 0.5 * physics.AIR_DENSITY * physics.CDA_M2 * v**2
    )
    return force * physics.MI_TO_M / physics.J_PER_GGE


def _accessory_per_mile(speed_mph: Tensor) -> Tensor:
    """Accessory draw over one mile, in GGE. Slower links spend more of it."""
    v = torch.clamp(speed_mph * physics.MPH_TO_MS, min=MIN_MS)
    return physics.ACCESSORY_WATTS * (physics.MI_TO_M / v) / physics.J_PER_GGE


def _transient_gge(speed_mph: Tensor) -> Tensor:
    """The acceleration energy a link average may hide, in GGE."""
    v = speed_mph * physics.MPH_TO_MS
    return (
        physics.TRANSIENT_KINETIC_MULTIPLE
        * 0.5
        * physics.EFFECTIVE_MASS_KG
        * v**2
        / physics.J_PER_GGE
    )


class PhysicsNet(nn.Module):
    """An MLP whose four outputs are the terms of a road-load decomposition.

    Each head is squashed into the interval physical law allows it:

    | head | range                                    | what it protects            |
    |------|------------------------------------------|-----------------------------|
    | `A`  | `[res/2, res/eta + acc]`                 | flat positivity, distance   |
    |      |                                          | growth, the ceiling         |
    | `B`  | `[k_pot, k_pot/eta]`                     | the climb floor, the ceiling|
    | `C`  | `[0, eta*k_pot]`                         | grade monotonicity, round   |
    |      |                                          | trips, regen, the floor     |
    | `T`  | `[0, transient/eta]`                     | the ceiling                 |

    `A >= res/2` is what `energy_grows_with_distance` needs: `physics.py` scales
    the real road load by `FLOOR_SAFETY_FACTOR = 0.5` to build that bound, and a
    per-mile rate at least that large means extending a link always pays for the
    road it adds. `B >= k_pot >= eta*k_pot >= C` gives grade monotonicity and
    round-trip convexity together, since a climb's marginal cost is then never
    below a descent's marginal saving.
    """

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(n_features, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
        )
        self.head = nn.Linear(HIDDEN, 4)
        with torch.no_grad():
            self.head.weight.mul_(0.1)
            # Start the transient head near zero: `T / d` is divided by a link
            # length as small as 0.002 mi, so a mid-range initialization would
            # start the fit orders of magnitude above the target.
            self.head.bias.copy_(torch.tensor([0.0, 0.0, 0.0, -3.0]))

    def forward(
        self, x: Tensor, speed_mph: Tensor, grade_percent: Tensor, miles: Tensor
    ) -> Tensor:
        # The heads sit exactly on their bounds when a sigmoid saturates, and
        # `physics.py` compares against float64 constants with a 1e-12
        # tolerance. In float32 the climb slope lands a part in 1e7 below
        # `K_POT` and the climb floor reads as violated by 4e-9 GGE. The
        # network stays in float32; the structural algebra is float64.
        raw = self.head(self.body(x)).double()
        a, b, c, t = raw.unbind(dim=-1)
        speed_mph = speed_mph.double()
        grade_percent = grade_percent.double()
        miles = miles.double()

        resistance = _road_load_per_mile(speed_mph)
        accessory = _accessory_per_mile(speed_mph)
        transient = _transient_gge(speed_mph)

        floor = 0.5 * resistance
        ceiling = resistance / ETA + accessory
        rate_flat = floor + (ceiling - floor) * torch.sigmoid(a)
        climb = K_POT * (1.0 + (1.0 / ETA - 1.0) * torch.sigmoid(b))
        descend = ETA * K_POT * torch.sigmoid(c)
        fixed = (transient / ETA) * torch.sigmoid(t)

        up = torch.clamp(grade_percent, min=0.0)
        down = torch.clamp(-grade_percent, min=0.0)
        return rate_flat + climb * up - descend * down + fixed / miles


def _columns(df: pd.DataFrame, device: torch.device) -> tuple[Tensor, ...]:
    """Network inputs and the three structural columns, as GPU tensors."""
    def column(name: str) -> Tensor:
        return torch.tensor(
            df[name].to_numpy(dtype=np.float32), device=device, dtype=torch.float32
        )

    x = torch.tensor(
        df[NET_FEATURES].to_numpy(dtype=np.float32), device=device
    )
    return x, column("speed_mph"), column("grade_percent"), column("miles")


def train_model() -> dict[str, float]:
    """Train and evaluate. Returns results dict."""
    t0 = time.time()
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = add_features(load_data())
    train_df, test_df = train_test_split(df, test_size=0.2, random_seed=42)

    y_train = train_df[TARGET].to_numpy(dtype=np.float32)
    y_test = test_df[TARGET].to_numpy(dtype=np.float32)
    journey_id_te = test_df["journey_id"].to_numpy()
    miles_te = test_df["miles"].to_numpy(dtype=np.float32)

    x_tr, speed_tr, grade_tr, miles_tr = _columns(train_df, device)
    y_tr = torch.as_tensor(y_train, device=device)

    # Standardize the network inputs only; the structural columns are consumed
    # in physical units and must stay in them.
    mean = x_tr.mean(dim=0, keepdim=True)
    std = x_tr.std(dim=0, keepdim=True).clamp(min=1e-6)
    x_tr = (x_tr - mean) / std
    # The target's own scale, so the loss is O(1) whatever the units.
    y_scale = float(y_tr.std())

    model = PhysicsNet(len(NET_FEATURES)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    n = x_tr.shape[0]
    generator = torch.Generator(device=device).manual_seed(SEED)
    for _ in range(EPOCHS):
        order = torch.randperm(n, device=device, generator=generator)
        for start in range(0, n, BATCH):
            idx = order[start : start + BATCH]
            pred = model(x_tr[idx], speed_tr[idx], grade_tr[idx], miles_tr[idx])
            loss = torch.mean(((pred - y_tr[idx]) / y_scale) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        schedule.step()
        if time.time() - t0 > TRAIN_SECONDS_CAP:
            break

    model.eval()

    @torch.no_grad()
    def score(frame: pd.DataFrame) -> np.ndarray:
        """Predict for a frame that already carries the derived columns."""
        x, speed, grade, miles = _columns(frame, device)
        out = model((x - mean) / std, speed, grade, miles)
        return out.cpu().numpy().astype(np.float64)

    def predict(frame: pd.DataFrame) -> np.ndarray:
        """Predict for a raw frame — the physics sweep's entry point."""
        return score(add_features(frame))

    # The test rows carry neighbours from the *unsplit* frame, so they must not
    # be re-derived here: their neighbours are training links, not the adjacent
    # test rows.
    predicted = score(test_df)

    results = evaluate(
        y_test,
        predicted,
        journey_id=journey_id_te,
        miles=miles_te,
        # How the physics sweep asks this model about a link it never saw.
        predict=predict,
    )

    report(
        results,
        model_family=MODEL_FAMILY,
        features=LINK_FEATURES,
        total_seconds=time.time() - t0,
    )

    return results


if __name__ == "__main__":
    run_with_budget(train_model)

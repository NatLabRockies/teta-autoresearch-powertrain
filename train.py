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

import sys
import time

import numpy as np
import pandas as pd
import shapely
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
    "prev_miles",
    "next_miles",
    "prev2_miles",
    "prev2_speed_mph",
    "next2_speed_mph",
    "prev_grade_percent",
    "entry_kinetic_gge",
    "exit_kinetic_gge",
    "sinuosity",
    "junction_turn_degrees",
    "prev_sinuosity",
    "next_junction_turn_degrees",
]

#: Consumed by the structural algebra rather than by the network. The
#: neighbour speeds appear here as well as in `NET_FEATURES`: the structure
#: needs them in physical units to size the kinetic energy a link releases.
STRUCTURAL_FEATURES = [
    "grade_percent",
    "miles",
]

#: The raw, unstandardized columns `PhysicsNet.forward` reads, in order.
RAW_COLUMNS = [
    "speed_mph",
    "grade_percent",
    "miles",
    "prev_speed_mph",
    "next_speed_mph",
    "next_junction_turn_degrees",
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

#: Safety net only. The epoch count is chosen to finish inside the harness
#: budget; this stops a run that somehow does not, so it reports a result
#: instead of being killed. Set well above the ~470s a full run takes at the
#: slow end of the machine's ~2.4x load swing: a run that truncates here is
#: not comparable to one that does not, so the cap must not bind in practice.
TRAIN_SECONDS_CAP = 540.0

# --- physical constants, read through `physics.py` so they cannot drift ---

K_POT = physics.MASS_KG * physics.G * physics.MI_TO_M / 100.0 / physics.J_PER_GGE
"""GGE of potential energy in one mile at one percent grade."""

ETA = physics.ETA_DRIVE

EARTH_RADIUS_M = 6371000.0
"""Mean Earth radius, m — for the straight-line distance between endpoints."""

SINUOSITY_CAP = 10.0
"""Ceiling on the winding ratio, for links whose endpoints nearly coincide."""

LENGTH_SCALE_MI = 0.05
"""Scale on the learned length constant of the transient's growth, in miles —
the dataset's median link is 0.040 mi, so this starts it in the right decade."""

STRAIGHT_ON_DEGREES = 180.0
"""A full reversal, the largest junction turn there is — the scale that maps a
turn onto the fraction of its kinetic energy a link sheds cornering."""

MIN_MS = 1e-3
"""Floor on speed in m/s, so the accessory term cannot divide by zero."""

GRADE_SLACK_REF = 1.0
"""Grade, in percent, at which the climb term may spend the full flat-rate
headroom. Below it the allowance is scaled down linearly to zero, which is what
keeps level ground exactly level."""


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    # sort by journey and time
    return df.sort_values(["journey_id", "link_start_time"])


def _sinuosity(df: pd.DataFrame, geometry: np.ndarray | None) -> np.ndarray:
    """Path length over straight-line endpoint distance, >= 1 for a real road.

    A winding link costs more than its length suggests: the vehicle corners,
    and cornering is speed the average hides. The geometry column is not
    physically determined by speed, grade and length, so `physics.build_frame`
    has no column for it — `physics.py` says a model consuming one must supply
    its own value inside `predict`, and 1.0 is the right one: a synthetic link
    is straight.
    """
    if geometry is None:
        return np.ones(len(df))
    start = shapely.get_point(geometry, 0)
    end = shapely.get_point(geometry, -1)
    lat = np.radians(0.5 * (shapely.get_y(start) + shapely.get_y(end)))
    dx = np.radians(shapely.get_x(end) - shapely.get_x(start)) * np.cos(lat)
    dy = np.radians(shapely.get_y(end) - shapely.get_y(start))
    chord_m = EARTH_RADIUS_M * np.hypot(dx, dy)
    length_m = df["miles"].to_numpy() * physics.MI_TO_M
    # A link whose endpoints coincide is a loop, not an infinitely winding
    # road; cap it rather than letting the ratio run away.
    return np.clip(length_m / np.maximum(chord_m, 1e-6), 1.0, SINUOSITY_CAP)


def _bearings(geometry: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compass bearing entering and leaving each link, in degrees.

    Taken from the first and last *segment* of the link's geometry rather than
    from its endpoints, so a link that bends in the middle still reports the
    direction it actually joins its neighbours at.
    """
    def bearing(tail: int, head: int) -> np.ndarray:
        a, b = shapely.get_point(geometry, tail), shapely.get_point(geometry, head)
        lat = np.radians(0.5 * (shapely.get_y(a) + shapely.get_y(b)))
        dx = np.radians(shapely.get_x(b) - shapely.get_x(a)) * np.cos(lat)
        dy = np.radians(shapely.get_y(b) - shapely.get_y(a))
        return np.degrees(np.arctan2(dx, dy))

    return bearing(0, 1), bearing(-2, -1)


def _kinetic_gge_numpy(speed_mph: np.ndarray) -> np.ndarray:
    """`_kinetic_gge` for feature construction, before anything is on the GPU."""
    v = speed_mph * physics.MPH_TO_MS
    return 0.5 * physics.EFFECTIVE_MASS_KG * v**2 / physics.J_PER_GGE


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
    grouped = df.groupby("journey_id", sort=False)["speed_mph"]
    df["prev_speed_mph"] = grouped.shift(1).fillna(0.0)
    df["next_speed_mph"] = grouped.shift(-1).fillna(0.0)
    # Two links back: whether the vehicle was already slowing on its approach.
    df["prev2_speed_mph"] = grouped.shift(2).fillna(0.0)
    # Two links forward: whether the braking continues past the next link.
    df["next2_speed_mph"] = grouped.shift(-2).fillna(0.0)
    # How far the previous link ran, which is how much to trust its average
    # speed as an estimate of the speed at its end. Zero at a trip start, where
    # there is no previous link to distrust.
    # The grade behind the vehicle. The link's own grade is structural, but the
    # previous one is not: arriving off a descent means arriving with momentum
    # the average speeds do not show.
    grade = df.groupby("journey_id", sort=False)["grade_percent"]
    df["prev_grade_percent"] = grade.shift(1).fillna(0.0)
    # The signed kinetic work of entering the link, in the target's own units.
    # The structure already divides a fixed energy by distance; what it cannot
    # do is square a speed, so hand the difference of squares over directly.
    df["entry_kinetic_gge"] = _kinetic_gge_numpy(
        df["speed_mph"].to_numpy()
    ) - _kinetic_gge_numpy(df["prev_speed_mph"].to_numpy())
    # The exit half: what the link hands on rather than what it was handed.
    df["exit_kinetic_gge"] = _kinetic_gge_numpy(
        df["next_speed_mph"].to_numpy()
    ) - _kinetic_gge_numpy(df["speed_mph"].to_numpy())
    # One WKB decode for the whole frame, shared by every geometry feature:
    # decoding 1.6M linestrings costs ~20s, and doing it twice was enough to
    # push the run into its training-time cap.
    geometry = (
        shapely.from_wkb(df["geometry"].to_numpy())
        if "geometry" in df.columns
        else None
    )
    df["sinuosity"] = _sinuosity(df, geometry)
    # The turn taken at the junction into this link: how far the vehicle had to
    # swing between leaving the previous link and joining this one. A sharp
    # turn is a deceleration the link averages hide. The sweep frame has no
    # geometry, so a synthetic link is entered straight on.
    if geometry is not None:
        entry, exit_ = _bearings(geometry)
        df["_exit_bearing"] = exit_
        previous = df.groupby("journey_id", sort=False)["_exit_bearing"].shift(1)
        turn = (entry - previous.to_numpy() + 180.0) % 360.0 - 180.0
        df["junction_turn_degrees"] = np.abs(np.nan_to_num(turn))
        df.drop(columns=["_exit_bearing"], inplace=True)
    else:
        df["junction_turn_degrees"] = np.zeros(len(df))
    # The turn waiting at the *end* of this link, which is the turn recorded
    # against the next one. A sharp turn ahead is braking that happens during
    # this link and is billed to it.
    df["next_junction_turn_degrees"] = (
        df.groupby("journey_id", sort=False)["junction_turn_degrees"]
        .shift(-1)
        .fillna(0.0)
    )
    # The shape of the road behind: a winding approach means the vehicle
    # arrives having already given up speed to corner.
    df["prev_sinuosity"] = (
        df.groupby("journey_id", sort=False)["sinuosity"].shift(1).fillna(1.0)
    )
    length = df.groupby("journey_id", sort=False)["miles"]
    df["prev_miles"] = length.shift(1).fillna(0.0)
    df["next_miles"] = length.shift(-1).fillna(0.0)
    # Reliability context for prev2_speed_mph, as prev_miles is for prev_speed.
    df["prev2_miles"] = length.shift(2).fillna(0.0)
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


def _kinetic_gge(speed_mph: Tensor) -> Tensor:
    """Kinetic energy at a given speed, in GGE.

    Effective mass, not curb mass — spinning the wheels up is work the
    simulator bills, and `physics.py` sizes its bounds the same way.
    """
    v = speed_mph * physics.MPH_TO_MS
    return 0.5 * physics.EFFECTIVE_MASS_KG * v**2 / physics.J_PER_GGE


def _transient_gge(speed_mph: Tensor) -> Tensor:
    """The acceleration energy a link average may hide, in GGE."""
    return physics.TRANSIENT_KINETIC_MULTIPLE * _kinetic_gge(speed_mph)


class PhysicsNet(nn.Module):
    """An MLP whose four outputs are the terms of a road-load decomposition.

    Each head is squashed into the interval physical law allows it:

    | head | range                                    | what it protects            |
    |------|------------------------------------------|-----------------------------|
    | `A`  | `[res/2, res/eta + acc]`                 | flat positivity, distance   |
    |      |                                          | growth, the ceiling         |
    | `B`  | `[k_pot, k_pot/eta + slack/g]`           | the climb floor, the ceiling|
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
        self.head = nn.Linear(HIDDEN, 8)
        with torch.no_grad():
            self.head.weight.mul_(0.1)
            # Start the transient head near zero: `T / d` is divided by a link
            # length as small as 0.002 mi, so a mid-range initialization would
            # start the fit orders of magnitude above the target.
            self.head.bias.copy_(
                torch.tensor([0.0, 0.0, 0.0, -3.0, -3.0, -3.0, 0.0, -3.0])
            )

    def forward(self, x: Tensor, raw: Tensor) -> Tensor:
        # The heads sit exactly on their bounds when a sigmoid saturates, and
        # `physics.py` compares against float64 constants with a 1e-12
        # tolerance. In float32 the climb slope lands a part in 1e7 below
        # `K_POT` and the climb floor reads as violated by 4e-9 GGE. The
        # network stays in float32; the structural algebra is float64.
        heads = self.head(self.body(x)).double()
        a, b, c, t, u, w, lam, corner = heads.unbind(dim=-1)
        (
            speed_mph,
            grade_percent,
            miles,
            prev_mph,
            next_mph,
            turn_ahead,
        ) = raw.double().unbind(dim=-1)

        resistance = _road_load_per_mile(speed_mph)
        accessory = _accessory_per_mile(speed_mph)
        transient = _transient_gge(speed_mph)

        floor = 0.5 * resistance
        ceiling = resistance / ETA + accessory
        rate_flat = floor + (ceiling - floor) * torch.sigmoid(a)

        up = torch.clamp(grade_percent, min=0.0)
        down = torch.clamp(-grade_percent, min=0.0)

        # The climb term is allowed to spend whatever ceiling headroom the flat
        # rate left unused, so the marginal cost of grade is not pinned to peak
        # drivetrain efficiency. A real climb runs the motor away from its peak,
        # so `mgh / ETA` is a floor on what it costs rather than a ceiling; the
        # `absolute_ceiling` check has room for that because it also allows a
        # transient and an accessory term the flat rate is not using.
        #
        # Gated by grade so the headroom vanishes on level ground, where there
        # is no climb to spend it on.
        slack = (ceiling - rate_flat) * torch.clamp(up / GRADE_SLACK_REF, max=1.0)
        climb = K_POT * up + torch.sigmoid(b) * (
            K_POT * up * (1.0 / ETA - 1.0) + slack
        )
        descend = ETA * K_POT * torch.sigmoid(c)

        # The fixed per-link cost, in two signed halves. `T+` is the
        # acceleration energy a link average hides; `T-` is the kinetic energy
        # the link gives back when it ends slower than it started, which is how
        # a flat link can come out negative at all.
        #
        # `T-` is gated by the energy actually released, so it vanishes at
        # steady state — which is what the physics sweep is, since every
        # synthetic link is its own journey and both neighbour speeds are equal
        # there. That is what keeps `flat_energy_positive` true by construction
        # while the model is still free to predict regen on real links. Its size
        # is capped by `eta * transient`, exactly the `absolute_floor` bound.
        # Two ways a link gives kinetic energy back. The first is the drop from
        # the speed it was entered at to the one it hands on. The second is
        # braking for the turn at its far end, which the neighbour speeds do
        # not show because the vehicle slows and then speeds back up inside the
        # link.
        #
        # The second is what exp10 tried to reach by widening the gate to the
        # link's own speed, and that broke the guarantee: under the 0-fill
        # convention a synthetic sweep link is indistinguishable from a real
        # launch-and-stop, so a gate built from speeds alone cannot vanish on
        # the sweep. A gate built from the *turn ahead* can — the sweep has no
        # geometry, so its turn is exactly zero, and the release with it.
        released = torch.clamp(
            _kinetic_gge(prev_mph) - _kinetic_gge(next_mph), min=0.0
        ) + torch.sigmoid(corner) * _kinetic_gge(speed_mph) * (
            turn_ahead / STRAIGHT_ON_DEGREES
        )
        # The transient may grow with link length. A half-mile link at 30 mph
        # average plausibly contains more acceleration events than a hundred-
        # foot one at the same average, and a fixed T says they contain the
        # same. `energy_grows_with_distance` permits any T that is
        # non-decreasing in distance, since the growth it demands is already
        # covered by `A >= res/2` - so a saturating factor is legal where a
        # free function of length would not be.
        #
        # At `growth = 0` this is exactly the length-independent T it replaces,
        # so the relaxation only adds freedom.
        scale = LENGTH_SCALE_MI * nn.functional.softplus(lam)
        growth = torch.sigmoid(w) * torch.exp(-miles / scale)
        fixed = (transient / ETA) * torch.sigmoid(t) * (
            1.0 - growth
        ) - torch.sigmoid(u) * torch.minimum(released, ETA * transient)

        return rate_flat + climb - descend * down + fixed / miles


def _columns(df: pd.DataFrame, device: torch.device) -> tuple[Tensor, Tensor]:
    """Network inputs and the raw structural columns, as GPU tensors."""
    x = torch.tensor(df[NET_FEATURES].to_numpy(dtype=np.float32), device=device)
    raw = torch.tensor(df[RAW_COLUMNS].to_numpy(dtype=np.float32), device=device)
    return x, raw


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

    x_tr, raw_tr = _columns(train_df, device)
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
    completed = 0
    generator = torch.Generator(device=device).manual_seed(SEED)
    for completed in range(1, EPOCHS + 1):
        order = torch.randperm(n, device=device, generator=generator)
        for start in range(0, n, BATCH):
            idx = order[start : start + BATCH]
            pred = model(x_tr[idx], raw_tr[idx])
            loss = torch.mean(((pred - y_tr[idx]) / y_scale) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        schedule.step()
        if time.time() - t0 > TRAIN_SECONDS_CAP:
            break

    # Whether the training-time cap bit. A truncated run is not comparable to
    # an untruncated one, and the metrics alone cannot tell the difference.
    print(f"epochs completed: {completed}/{EPOCHS}", file=sys.stderr)

    model.eval()

    @torch.no_grad()
    def score(frame: pd.DataFrame) -> np.ndarray:
        """Predict for a frame that already carries the derived columns."""
        x, raw = _columns(frame, device)
        out = model((x - mean) / std, raw)
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

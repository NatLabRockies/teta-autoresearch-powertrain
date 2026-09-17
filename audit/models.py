"""Each arm's final model, rebuilt from its final `train.py`.

Settings are copied from unguided `bc07466` (exp48) and domain-guided `e315384`
(exp50). Changing any of them breaks the reproduction check.

The unguided model has three parts, and the audit needs to fit them separately:

    UnguidedEnsemble        5 seeded HistGradientBoostingRegressors, averaged
    sequence_predictions    a small 1-D convolutional network over each journey
    journey_offset          a per-journey correction from training residuals

The domain-guided model is two small MLPs, averaged.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

TARGET_COL = "energy_rate_gge"

# --- unguided: 5-seed HistGradientBoosting ensemble (bc07466) --------------------

N_ENSEMBLE = 5
BASE_SEED = 52

UNGUIDED_PARAMS = {
    "max_iter": 2000,
    "learning_rate": 0.1,
    "max_leaf_nodes": 31,
    "max_features": 0.7,
    "early_stopping": False,
}


class UnguidedEnsemble:
    """Mean of five fits that differ only by random seed."""

    def __init__(self, n: int = N_ENSEMBLE) -> None:
        self.seeds = tuple(BASE_SEED + i for i in range(n))
        self.models: list[HistGradientBoostingRegressor] = []

    def fit(self, x, y: np.ndarray) -> "UnguidedEnsemble":
        self.models = [
            HistGradientBoostingRegressor(**UNGUIDED_PARAMS, random_state=s).fit(x, y)
            for s in self.seeds
        ]
        return self

    def predict(self, x) -> np.ndarray:
        return np.mean([m.predict(x) for m in self.models], axis=0)


# --- unguided: the journey offset ------------------------------------------------

JOURNEY_PRIOR_LINKS = 20.0


def out_of_fold_residual(
    train_df: pd.DataFrame, y_train: np.ndarray, features: list[str]
) -> np.ndarray:
    """Training residuals from a model that did not see the row it scores.

    From unguided/train.py. Two folds, each scored by the model fit on the other.
    This costs two more ensemble fits.
    """
    fold = np.arange(len(train_df)) % 2
    residual = np.empty(len(train_df), dtype=np.float64)
    for held_out in (0, 1):
        fit = fold != held_out
        model = UnguidedEnsemble()
        model.fit(train_df.loc[fit, features], y_train[fit])
        rows = ~fit
        residual[rows] = y_train[rows] - model.predict(train_df.loc[rows, features])
    return residual


def journey_offset(
    train_df: pd.DataFrame, test_df: pd.DataFrame, residual: np.ndarray
) -> np.ndarray:
    """Per-journey correction. From unguided/train.py.

    Distance-weighted mean training residual per journey, shrunk toward zero.
    It is built from the labels of the journey's own training links.
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


# --- unguided: the sequence model ------------------------------------------------

CHANNELS = 64
DILATIONS = (1, 2)
KERNEL = 5
SEQ_LR = 4.5e-3
#: The arm trains this for 60 seconds of wall clock, not a fixed number of steps.
SEQ_SECONDS = 60.0
MAX_TOKENS_PER_BATCH = 32768
BLEND = 0.30
SEQ_SEED = 52


def _build_seq(n_features: int):
    from torch import nn

    class Block(nn.Module):
        """Residual dilated convolution over the link axis, padded on both sides.

        Same padding on both sides means each output reads links after it as
        well as before it.
        """

        def __init__(self, dilation: int) -> None:
            super().__init__()
            self.conv1 = nn.Conv1d(
                CHANNELS,
                CHANNELS,
                KERNEL,
                padding=dilation * (KERNEL - 1) // 2,
                dilation=dilation,
            )
            self.conv2 = nn.Conv1d(CHANNELS, CHANNELS, 1)
            self.norm = nn.GroupNorm(1, CHANNELS)
            self.act = nn.GELU()

        def forward(self, x):
            return x + self.conv2(self.act(self.conv1(self.norm(x))))

    class LinkSequenceNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inp = nn.Conv1d(n_features, CHANNELS, 1)
            self.blocks = nn.ModuleList(Block(d) for d in DILATIONS)
            self.out = nn.Conv1d(CHANNELS, 1, 1)

        def forward(self, x):
            h = self.inp(x)
            for block in self.blocks:
                h = block(h)
            return self.out(h).squeeze(1)

    return LinkSequenceNet()


def _journey_bounds(journey_id: np.ndarray) -> np.ndarray:
    starts = np.flatnonzero(np.r_[True, journey_id[1:] != journey_id[:-1]])
    return np.c_[starts, np.r_[starts[1:], len(journey_id)]]


def _pack_journeys(df, x, y, w, device):
    """Batch journeys of similar length into [batch, features, links] tensors."""
    import torch

    bounds = _journey_bounds(df["journey_id"].to_numpy())
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
    df: pd.DataFrame,
    train_rows: np.ndarray,
    features: list[str],
    *,
    seconds: float = SEQ_SECONDS,
) -> np.ndarray:
    """Fit the sequence model and predict every link of every journey.

    From unguided/train.py `sequence_predictions`. Test links are fed in as
    context but never used in the loss, exactly as the arm did it.
    """
    import torch

    y = df[TARGET_COL].to_numpy(dtype=np.float32)
    w = np.zeros(len(df), dtype=np.float32)
    w[train_rows] = 1.0
    is_train = w > 0

    x = df[features].to_numpy(dtype=np.float32)
    x = (x - np.nanmean(x[is_train], axis=0)) / (np.nanstd(x[is_train], axis=0) + 1e-6)
    np.nan_to_num(x, copy=False)
    y_scale = float(y[is_train].std())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEQ_SEED)
    packed = _pack_journeys(df, x, y, w, device)
    net = _build_seq(x.shape[1]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=SEQ_LR, weight_decay=1e-4)

    rng = np.random.default_rng(SEQ_SEED)
    start = time.time()
    while time.time() - start < seconds:
        for group in opt.param_groups:
            group["lr"] = (
                SEQ_LR * 0.5 * (1 + math.cos(math.pi * (time.time() - start) / seconds))
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


# --- domain-guided: 2 x 128 MLP ensemble (e315384) -------------------------------

HIDDEN = 128
N_MODELS = 2
BATCH_SIZE = 8192
LEARNING_RATE = 1e-3
MAX_EPOCHS = 320
WEIGHT_DECAY = 1e-4
MLP_SEED = 52


def _build_net(n_features: int):
    from torch import nn

    return nn.Sequential(
        nn.Linear(n_features, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, 1),
    )


class DomainGuidedMLP:
    """Two 128-wide MLPs, averaged. Fixed epoch count, so it is deterministic."""

    def __init__(self, features: list[str]) -> None:
        self.features = features
        self.nets: list = []
        self.mu = self.sd = None
        self.y_mu = self.y_sd = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "DomainGuidedMLP":
        import torch
        from torch import nn

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.mu, self.sd = x.mean(axis=0), x.std(axis=0)
        self.sd[self.sd == 0] = 1.0
        self.y_mu, self.y_sd = float(y.mean()), float(y.std())

        xt = torch.from_numpy(((x - self.mu) / self.sd).astype(np.float32)).to(device)
        yt = (
            torch.from_numpy(((y - self.y_mu) / self.y_sd).astype(np.float32))
            .to(device)
            .unsqueeze(1)
        )

        loss_fn = nn.MSELoss()
        n = xt.shape[0]
        for member in range(N_MODELS):
            torch.manual_seed(MLP_SEED + member)
            net = _build_net(xt.shape[1]).to(device)
            opt = torch.optim.AdamW(
                net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
            )
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
            for _ in range(MAX_EPOCHS):
                order = torch.randperm(n, device=device)
                for start in range(0, n, BATCH_SIZE):
                    idx = order[start : start + BATCH_SIZE]
                    opt.zero_grad()
                    loss_fn(net(xt[idx]), yt[idx]).backward()
                    opt.step()
                sched.step()
            net.eval()
            self.nets.append(net)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        import torch

        device = next(self.nets[0].parameters()).device
        xe = torch.from_numpy(((x - self.mu) / self.sd).astype(np.float32)).to(device)
        outs = []
        with torch.no_grad():
            for net in self.nets:
                outs.append(
                    torch.cat(
                        [
                            net(xe[i : i + 65536]).squeeze(1)
                            for i in range(0, len(xe), 65536)
                        ]
                    )
                    .cpu()
                    .numpy()
                )
        return np.mean(outs, axis=0) * self.y_sd + self.y_mu

    def to_cpu_eval(self) -> list:
        """CPU copies in eval mode, for the inference benchmark."""
        import copy

        return [copy.deepcopy(n).cpu().eval() for n in self.nets]

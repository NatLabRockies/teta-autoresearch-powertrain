"""Reconstructions of each arm's final model, with fit/predict/persist.

Hyperparameters are transcribed from the arms' final `train.py` — unguided
`bc07466` (exp48), domain-guided `e315384` (exp50). Changing any of them invalidates
the reproduction check in `audit_accuracy.py`.

The unguided arm's model is three parts, and the audit needs to score them apart as
well as together, so each is a separate object here:

    UnguidedEnsemble    5 seeded HistGradientBoostingRegressors, averaged
    SequenceMember      the dilated 1-D CNN over each journey's link chain, in
                        both the arm's bidirectional form and a causal one
    journey_offset      the per-journey residual correction (in `features`-free
                        functions below, since it is arithmetic, not a fit)

One reproducibility note, which is a property of the arm rather than of this audit:
the sequence member trains for a **wall-clock budget**, not a fixed number of epochs.
On different hardware it sees a different number of updates, so `unguided/reported`
is reproducible only to within that drift. The domain-guided arm's network runs a
fixed 320 epochs and is deterministic.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

# --- unguided: 5-seed HistGradientBoosting ensemble (bc07466) -------------------

N_ENSEMBLE = 5
BASE_SEED = 52
FEATURE_SUBSAMPLE = 0.7

UNGUIDED_PARAMS = {
    "max_iter": 2000,
    "learning_rate": 0.1,
    "max_leaf_nodes": 31,
    "max_features": FEATURE_SUBSAMPLE,
    "early_stopping": False,
}


class UnguidedEnsemble:
    """Mean of five independently seeded fits.

    The members differ only by `random_state`, which drives the per-split feature
    subsampling — the fits are otherwise deterministic. No sample weighting: this
    arm dropped the mile-weighted loss its predecessor used.
    """

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

    @property
    def n_trees(self) -> int:
        return sum(len(m._predictors) for m in self.models)


# --- unguided: the journey offset ----------------------------------------------

JOURNEY_PRIOR_LINKS = 20.0


def out_of_fold_residual(
    train_df: pd.DataFrame, y_train: np.ndarray, features: list[str]
) -> np.ndarray:
    """Training residuals from a model that did not see the row it is scoring.

    Transcribed from unguided/train.py. Two folds, each scored by the model fitted
    on the other. This costs two more ensemble fits, which is most of why
    reproducing this arm is expensive.
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
    """Per-journey correction. Transcribed from unguided/train.py.

    Distance-weighted mean training residual per journey, shrunk toward zero. This
    is the `LABEL`-tier component: it is built from the targets of the journey's own
    training links, so it has no inference-time form.
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


# --- unguided: the sequence member ---------------------------------------------

CHANNELS = 64
DILATIONS = (1, 2)
KERNEL = 5
SEQ_LR = 4.5e-3
#: The arm caps its sequence training at 60s and, in a 504s run, actually got the
#: full 60. Fixed here rather than derived from a run deadline, so the audit does
#: not inherit the arm's wall-clock coupling.
SEQ_SECONDS = 60.0
MAX_TOKENS_PER_BATCH = 32768
BLEND = 0.30
SEQ_SEED = 52

#: Receptive field of the stacked dilated blocks, in links, each way.
#: 1 + sum(dilation * (KERNEL - 1)) // 2 per side.
SEQ_HALF_WIDTH = sum(d * (KERNEL - 1) for d in DILATIONS) // 2


def _build_seq(n_features: int, causal: bool):
    import torch
    from torch import nn

    class Block(nn.Module):
        """Residual dilated-convolution block over the link axis.

        `causal=True` pads on the left only, so output position i depends on inputs
        at positions <= i. That is the whole difference between a sequence member a
        Compass search could run and the one the arm actually fit.
        """

        def __init__(self, dilation: int) -> None:
            super().__init__()
            self.pad = dilation * (KERNEL - 1)
            self.causal = causal
            pad = 0 if causal else self.pad // 2
            self.conv1 = nn.Conv1d(
                CHANNELS, CHANNELS, KERNEL, padding=pad, dilation=dilation
            )
            self.conv2 = nn.Conv1d(CHANNELS, CHANNELS, 1)
            self.norm = nn.GroupNorm(1, CHANNELS)
            self.act = nn.GELU()

        def forward(self, x):
            h = self.norm(x)
            if self.causal:
                h = torch.nn.functional.pad(h, (self.pad, 0))
            return x + self.conv2(self.act(self.conv1(h)))

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
    causal: bool = False,
    seconds: float = SEQ_SECONDS,
) -> np.ndarray:
    """Fit the sequence member and predict every link of every journey.

    Transcribed from unguided/train.py `sequence_predictions`, with one added
    switch: `causal=True` masks the convolutions so the network reads only links
    already traversed. Everything else — standardization, masking of the loss to
    training rows, the cosine schedule, the seed — is the arm's.

    Test links are fed in as context but never scored, exactly as the arm did it.
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
    net = _build_seq(x.shape[1], causal).to(device)
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


#: Set by `audit_accuracy` so this module needs no import from `features`.
TARGET_COL = "energy_rate_gge"


# --- domain-guided: 2 x 128 MLP ensemble (e315384) -----------------------------

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
    """The domain-guided arm's model: two 128-wide members, averaged.

    Kept as an object so the inference benchmark can reuse the fitted nets instead
    of retraining for them. Members differ only by seed (init + shuffle order); the
    epoch count is fixed, so this is deterministic on a given device.
    """

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

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.nets[0].parameters()) * len(self.nets)

    def to_cpu_eval(self) -> list:
        """CPU copies in eval mode, for the inference benchmark."""
        import copy

        return [copy.deepcopy(n).cpu().eval() for n in self.nets]

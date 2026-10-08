"""Inference time per link for each model, on one CPU thread.

RouteE Compass calls the link model at every link it looks at during a route
search, on CPU, inside one search thread. So this pins everything to a single
thread and times small batches as well as large ones. A plain Dijkstra loop
scores one link at a time; a batched search scores a few hundred.

Three models are timed: the unguided model as run, the unguided model refit on
usable inputs, and the domain-guided model.

These are Python timings. Compass is written in Rust, so the absolute numbers
are pessimistic for every model. The ratio between models is what carries over.

Run as a subprocess so the thread-pinning variables are set before numpy,
sklearn and torch import.
"""

from __future__ import annotations

import os

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_v] = "1"

import json  # noqa: E402
import pickle  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

# The trees were fit on named columns. The benchmark feeds bare arrays on purpose,
# because a Compass link model has no pandas layer. Column order is the same, so
# only the name check fires.
warnings.filterwarnings(
    "ignore", message="X does not have valid feature names", category=UserWarning
)

import contract as C  # noqa: E402

CACHE = Path("cache")
RESULTS = Path("results")

BATCH_SIZES = [1, 64, 512, 4096]


def _time_predict(fn, x, batch: int, min_reps: int, min_seconds: float) -> float:
    """Median microseconds per link over repeated timed calls."""
    xb = x[:batch]
    for _ in range(3):
        fn(xb)
    times = []
    t_start = time.perf_counter()
    while len(times) < min_reps or (time.perf_counter() - t_start) < min_seconds:
        t0 = time.perf_counter()
        fn(xb)
        times.append((time.perf_counter() - t0) / batch * 1e6)
        if len(times) > 100_000:
            break
    return float(np.median(times))


def _load(name: str):
    with (CACHE / f"{name}.pkl").open("rb") as fh:
        return pickle.load(fh)


def run() -> dict:
    RESULTS.mkdir(exist_ok=True)

    ug = _load("unguided_as_run")
    ug_refit = _load("unguided_retrained_contract")
    dg = _load("domain_guided_as_run")

    import torch

    torch.set_num_threads(1)
    nets = dg.to_cpu_eval()

    rng = np.random.default_rng(0)
    big = max(BATCH_SIZES)
    x_ug = rng.normal(size=(big, len(C.UNGUIDED_FEATURES)))
    x_ugr = rng.normal(size=(big, len(C.usable(C.UNGUIDED_FEATURES))))
    x_dg = torch.from_numpy(
        rng.normal(size=(big, len(C.DOMAIN_GUIDED_FEATURES))).astype(np.float32)
    )

    def predict_dg(xb):
        with torch.no_grad():
            return sum(net(xb) for net in nets) / len(nets)

    models = {
        "unguided/as-run": (ug.predict, x_ug),
        "unguided/retrained": (ug_refit.predict, x_ugr),
        "domain-guided": (predict_dg, x_dg),
    }
    out: dict = {
        "conditions": {
            "device": "cpu",
            "threads": 1,
            "torch_version": torch.__version__,
        },
        "what": {
            "unguided/as-run": "10,000 boosted trees (5 seeds x 2,000 trees)",
            "unguided/retrained": "10,000 boosted trees (5 seeds x 2,000 trees)",
            "domain-guided": "2 MLPs, 11 -> 128 -> 128 -> 1",
        },
        "us_per_link": {},
    }

    print("measuring latency (single-threaded CPU) ...", flush=True)
    print(f"{'batch':>8} {'ug as-run':>12} {'ug retrained':>13} {'domain-guided':>15}")
    for b in BATCH_SIZES:
        min_reps, min_secs = (30, 2.0) if b <= 512 else (3, 1.0)
        row = {
            k: round(_time_predict(fn, x, b, min_reps, min_secs), 3)
            for k, (fn, x) in models.items()
        }
        out["us_per_link"][str(b)] = row
        print(
            f"{b:>8} {row['unguided/as-run']:>12.3f} "
            f"{row['unguided/retrained']:>13.3f} {row['domain-guided']:>15.3f}"
        )

    with (RESULTS / "inference.json").open("w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote results/inference.json")
    return out


if __name__ == "__main__":
    run()

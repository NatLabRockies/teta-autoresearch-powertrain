"""Inference cost of each arm's final model, under Compass-like conditions.

RouteE Compass evaluates a link-cost function at every link traversal in a
shortest-path search, on CPU, inside a search thread. So the benchmark pins both
models to a single thread and measures small batches as well as large ones — a naive
Dijkstra loop scores one link at a time; a frontier-batched implementation scores a
few hundred.

Three quantities are reported, in increasing order of how far they travel:

  measured latency   microseconds per link in this Python harness. Honest for a
                     Python deployment, pessimistic for Compass (Rust), and dominated
                     by call overhead at batch 1. Read the ratio, not the absolute.
  arithmetic cost    node visits / multiply-accumulates per prediction. Hardware- and
                     language-independent, so this is the number that transfers.
  label state        how many extra floats each model forces every search label to
                     carry. This is the memory penalty domain.md names, and it is the
                     cost that decides feasibility rather than speed.

Three models are measured, not two: the unguided arm as run, the unguided recipe
refit inside the contract, and the domain-guided model. The middle one matters
because the as-run model is not implementable at any price — comparing only against
it would price a thing nobody can buy.

Run as a subprocess so the thread-pinning environment variables take effect before
numpy/sklearn/torch import.
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

# The models were fitted on named frames; the benchmark feeds them bare arrays on
# purpose, because a Compass link-cost function has no pandas layer and the
# per-call overhead of one would be measured as if it were model cost. Column
# order is identical, so the predictions are the same — only the name check fires.
warnings.filterwarnings(
    "ignore", message="X does not have valid feature names", category=UserWarning
)

import contract as C  # noqa: E402
import models as M  # noqa: E402

CACHE = Path("cache")
RESULTS = Path("results")

BATCH_SIZES = [1, 8, 64, 512, 4096, 65536]

#: Link traversals expanded by a search, for the derived wall-clock estimates.
#: A metro-scale point-to-point query expands on the order of 1e5; a large or
#: multi-destination query on the order of 1e6.
SEARCH_SIZES = [100_000, 1_000_000]


def _time_predict(fn, x, batch: int, min_reps: int, min_seconds: float):
    """Median microseconds per link, over repeated timed passes."""
    xb = x[:batch]
    for _ in range(3):  # warm up
        fn(xb)
    times = []
    t_start = time.perf_counter()
    reps = 0
    while reps < min_reps or (time.perf_counter() - t_start) < min_seconds:
        t0 = time.perf_counter()
        fn(xb)
        times.append((time.perf_counter() - t0) / batch * 1e6)
        reps += 1
        if reps > 100_000:
            break
    return float(np.median(times)), reps


def _gbdt_arithmetic(model) -> dict:
    """Node visits per prediction for a seeded ensemble."""
    n_trees = 0
    total_leaves = 0
    max_depth = 0
    for m in model.models:
        for stage in m._predictors:
            for pred in stage:
                n_trees += 1
                total_leaves += int(pred.get_n_leaf_nodes())
                max_depth = max(max_depth, int(pred.get_max_depth()))
    mean_leaves = total_leaves / n_trees
    # A traversal visits one node per level. Leaf-wise growth makes trees deeper
    # than balanced, so the true mean path length sits between these bounds.
    return {
        "form": f"{n_trees:,} boosted trees ({len(model.models)} seeds averaged)",
        "n_trees": n_trees,
        "mean_leaves_per_tree": round(mean_leaves, 1),
        "max_tree_depth": max_depth,
        "node_visits_per_prediction_lower": int(n_trees * np.log2(mean_leaves)),
        "node_visits_per_prediction_upper": int(n_trees * max_depth),
    }


def _mlp_arithmetic(n_features: int, hidden: int, n_members: int) -> dict:
    per_member = n_features * hidden + hidden * hidden + hidden * 1
    macs = per_member * n_members
    params = (per_member + 2 * hidden + 1) * n_members
    return {
        "form": f"{n_members} x ({n_features} -> {hidden} -> {hidden} -> 1), ReLU",
        "n_params": params,
        "multiply_accumulates_per_prediction": macs,
        "flops_per_prediction": 2 * macs,
    }


def _seq_arithmetic() -> dict:
    """Per-link cost of the dilated CNN member, if it could be run at all."""
    c, k = M.CHANNELS, M.KERNEL
    # inp 1x1 conv + per block (kxk dilated conv + 1x1 conv) + out 1x1 conv
    n_in = len(C.UNGUIDED_SEQUENCE_FEATURES)
    macs = n_in * c + len(M.DILATIONS) * (c * c * k + c * c) + c
    return {
        "form": (
            f"2-block dilated 1-D CNN, {c} channels, kernel {k}, "
            f"dilations {M.DILATIONS}"
        ),
        "multiply_accumulates_per_link": macs,
        "receptive_half_width_links": M.SEQ_HALF_WIDTH,
    }


def _label_state() -> dict:
    """Extra per-label state each feature set forces on the Compass search.

    A one-link lookback is free: the previous edge id is already part of the path a
    search label describes, and every LOOKBACK feature is a static attribute of that
    edge or an arithmetic combination of it. Anything more is a distinct scalar the
    label must store, and labels differing in it can no longer be merged — the
    memory blow-up domain.md refuses.
    """
    ug_state = C.label_cost(C.UNGUIDED_FEATURES)
    contract_feats = C.deployable(C.UNGUIDED_FEATURES)
    return {
        "domain-guided": {
            "extra_floats_per_label": 0,
            "detail": (
                f"all {len(C.DOMAIN_GUIDED_FEATURES)} features derive from the "
                "current edge and the previous edge id, which the search label "
                "already carries"
            ),
        },
        "unguided/as-run": {
            "extra_floats_per_label": ug_state,
            "detail": (
                "not implementable at any memory cost. next_miles, next_gap_seconds "
                "and dv_out need links the search has not chosen; gap_seconds exists "
                "only in the simulated trace; journey_rate and journey_gge_per_mile "
                "are means of the target. The sequence member reads "
                f"{M.SEQ_HALF_WIDTH} links AHEAD as well as behind, and the journey "
                "offset needs labels."
            ),
        },
        "unguided/retrained-contract": {
            "extra_floats_per_label": 0,
            "detail": (
                f"the {len(contract_feats)} deployable features only "
                f"({', '.join(contract_feats)}) — same label cost as domain-guided"
            ),
        },
        "unguided/retrained-state": {
            "extra_floats_per_label": 1,
            "detail": (
                "adds prev2_speed: one extra float per label, and labels differing "
                "only in the speed two links back can no longer be merged"
            ),
        },
    }


def _load(name: str):
    with (CACHE / f"{name}.pkl").open("rb") as fh:
        return pickle.load(fh)


def run() -> dict:
    RESULTS.mkdir(exist_ok=True)

    ug = _load("unguided_as_run")
    ug_contract = _load("unguided_retrained_contract")
    dg = _load("domain_guided_as_run")

    import torch

    torch.set_num_threads(1)
    nets_cpu = dg.to_cpu_eval()

    n_ug = len(C.UNGUIDED_FEATURES)
    n_ugc = len(C.deployable(C.UNGUIDED_FEATURES))
    n_dg = len(C.DOMAIN_GUIDED_FEATURES)

    rng = np.random.default_rng(0)
    big = max(BATCH_SIZES)
    x_ug = rng.normal(size=(big, n_ug))
    x_ugc = rng.normal(size=(big, n_ugc))
    x_dg = rng.normal(size=(big, n_dg)).astype(np.float32)
    t_dg = torch.from_numpy(x_dg)

    def predict_dg(xb):
        with torch.no_grad():
            return sum(net(xb) for net in nets_cpu) / len(nets_cpu)

    out: dict = {
        "conditions": {
            "device": "cpu",
            "threads": 1,
            "note": (
                "single-threaded CPU, matching one Compass search thread. Timings are "
                "Python-harness timings; the arithmetic and label-state figures are "
                "the language-independent ones."
            ),
            "torch_version": torch.__version__,
        },
        "latency_us_per_link": {},
        "arithmetic": {
            "unguided/as-run": _gbdt_arithmetic(ug),
            "unguided/retrained-contract": _gbdt_arithmetic(ug_contract),
            "unguided/sequence-member": _seq_arithmetic(),
            "domain-guided": _mlp_arithmetic(n_dg, M.HIDDEN, M.N_MODELS),
        },
        "label_state": _label_state(),
        "model_size_bytes": {
            "unguided/as-run": len(pickle.dumps(ug)),
            "unguided/retrained-contract": len(pickle.dumps(ug_contract)),
            "domain-guided": len(pickle.dumps(nets_cpu)),
        },
    }

    print("measuring latency (single-threaded CPU) ...", flush=True)
    header = f"{'batch':>8} {'ug as-run':>12} {'ug contract':>13} {'domain-guided':>15}"
    print(header)
    for b in BATCH_SIZES:
        # Large GBDT batches are slow; cap the work at the big sizes.
        min_reps, min_secs = (30, 2.0) if b <= 512 else (3, 1.0)
        ug_us, _ = _time_predict(ug.predict, x_ug, b, min_reps, min_secs)
        ugc_us, _ = _time_predict(ug_contract.predict, x_ugc, b, min_reps, min_secs)
        dg_us, _ = _time_predict(predict_dg, t_dg, b, min_reps, min_secs)
        out["latency_us_per_link"][str(b)] = {
            "unguided/as-run": round(ug_us, 3),
            "unguided/retrained-contract": round(ugc_us, 3),
            "domain-guided": round(dg_us, 3),
            "ratio_contract_over_dg": round(ugc_us / dg_us, 1),
        }
        print(f"{b:>8} {ug_us:>12.3f} {ugc_us:>13.3f} {dg_us:>15.3f}")

    out["search_wall_clock_seconds"] = {}
    for n in SEARCH_SIZES:
        row = {}
        for label, b in [("unbatched", "1"), ("batched_512", "512")]:
            lat = out["latency_us_per_link"][b]
            row[label] = {
                k: round(lat[k] * n / 1e6, 3)
                for k in (
                    "unguided/as-run",
                    "unguided/retrained-contract",
                    "domain-guided",
                )
            }
        out["search_wall_clock_seconds"][str(n)] = row

    with (RESULTS / "inference.json").open("w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote results/inference.json")
    return out


if __name__ == "__main__":
    run()

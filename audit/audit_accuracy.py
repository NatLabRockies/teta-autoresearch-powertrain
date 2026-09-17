"""Link RMSE for both arms' final models.

Three numbers:

  unguided/reported    the full three-part blend, as the arm ran it. Must match
                       the number the arm reported, or nothing else counts. It
                       cannot run in Compass: 8 of its 13 features, its sequence
                       model and its journey offset all need inputs a route
                       search does not have.
  unguided/retrained   the same tree recipe, refit on usable inputs only. This is
                       the fair comparison: a model built for the rules.
  domain-guided        as the arm ran it. Every input is usable.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np

import contract as C
import features as F
import models as M
from harness import evaluate, train_test_split

CACHE = Path("cache")
RESULTS = Path("results")

#: Link RMSE each arm's harness printed at its final commit.
REPORTED = {
    "unguided": {"commit": "bc07466", "rmse": 0.004785},
    "domain-guided": {"commit": "e315384", "rmse": 0.006823},
}
#: The random-forest model both arms started from.
BASELINE_RMSE = 0.013345

#: How far a reproduced number may sit from the reported one. The unguided
#: sequence model trains for 60 seconds of wall clock, so a different machine
#: fits a slightly different model; it gets 5%. The domain-guided model runs a
#: fixed 320 epochs and gets 2%.
REPRO_TOL = {"unguided": 0.05, "domain-guided": 0.02}


def _cached(name: str, build):
    CACHE.mkdir(exist_ok=True)
    p = CACHE / f"{name}.pkl"
    if p.exists():
        with p.open("rb") as fh:
            return pickle.load(fh)
    obj = build()
    with p.open("wb") as fh:
        pickle.dump(obj, fh)
    return obj


def run() -> dict:
    t0 = time.time()
    RESULTS.mkdir(exist_ok=True)

    print("loading and building features ...", flush=True)
    df = F.build_all(F.load_sorted()).drop(columns=["geometry"])
    # Row index that survives the split, so the sequence model can be given the
    # train mask in the journey-ordered frame it needs.
    df["_row"] = np.arange(len(df), dtype=np.int64)

    train_df, test_df = train_test_split(df, test_size=0.2, random_seed=42)
    F.add_journey_effect(train_df, test_df)

    y_train = train_df[F.TARGET].to_numpy(dtype=np.float32)
    y_test = test_df[F.TARGET].to_numpy(dtype=np.float32)
    jid_te = test_df["journey_id"].to_numpy()
    miles_te = test_df["miles"].to_numpy(dtype=np.float32)
    tr_rows = train_df["_row"].to_numpy()
    te_rows = test_df["_row"].to_numpy()
    print(f"  train {len(train_df):,}  test {len(test_df):,}", flush=True)

    UG = C.UNGUIDED_FEATURES
    DG = C.DOMAIN_GUIDED_FEATURES
    out: dict = {
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "baseline_rmse": BASELINE_RMSE,
        "features": {"unguided": C.rows(UG), "domain-guided": C.rows(DG)},
        "components": C.COMPONENTS,
        "reproduction": {},
        "configs": {},
    }

    def rmse(pred: np.ndarray) -> float:
        return evaluate(y_test, pred, journey_id=jid_te, miles=miles_te)["rmse"]

    def record(key: str, value: float, note: str) -> None:
        out["configs"][key] = {
            "rmse": round(value, 6),
            "vs_baseline_pct": round(100 * (value / BASELINE_RMSE - 1), 1),
            "note": note,
        }
        print(f"  {key:22s} link rmse {value:.6f}", flush=True)

    # -- unguided, as run ------------------------------------------------------
    print("\n[1/3] unguided: fitting the model as run ...", flush=True)
    print("  (a) 5 x HistGBR on 13 features", flush=True)
    ens = _cached(
        "unguided_as_run", lambda: M.UnguidedEnsemble().fit(train_df[UG], y_train)
    )
    print("  (b) out-of-fold residual (2 more ensemble fits)", flush=True)
    resid = _cached(
        "unguided_oof_residual",
        lambda: M.out_of_fold_residual(train_df, y_train, UG),
    )
    print("  (c) sequence model (60s)", flush=True)
    seq = _cached(
        "unguided_sequence",
        lambda: M.sequence_predictions(df, tr_rows, C.UNGUIDED_SEQUENCE_FEATURES),
    )
    blend_te = (1 - M.BLEND) * ens.predict(test_df[UG]) + M.BLEND * seq[te_rows]
    blend_resid = (1 - M.BLEND) * resid + M.BLEND * (y_train - seq[tr_rows])
    offset = M.journey_offset(train_df, test_df, blend_resid)
    value = rmse(blend_te + offset)
    record(
        "unguided/reported",
        value,
        "all 13 features, as the arm ran it; cannot run in Compass",
    )
    out["reproduction"]["unguided"] = _repro_check("unguided", value)

    # -- unguided, retrained ---------------------------------------------------
    print("\n[2/3] unguided: refitting on usable inputs only ...", flush=True)
    keep = C.usable(UG)
    refit = _cached(
        "unguided_retrained_contract",
        lambda: M.UnguidedEnsemble().fit(train_df[keep], y_train),
    )
    record(
        "unguided/retrained",
        rmse(refit.predict(test_df[keep])),
        f"same recipe, refit on the {len(keep)} usable features",
    )

    # -- domain-guided ---------------------------------------------------------
    print("\n[3/3] domain-guided: fitting as run (2 x 128 MLP, 320 epochs) ...")
    xd_tr = F.domain_guided_matrix(train_df, DG)
    xd_te = F.domain_guided_matrix(test_df, DG)
    dg = _cached(
        "domain_guided_as_run", lambda: M.DomainGuidedMLP(DG).fit(xd_tr, y_train)
    )
    value = rmse(dg.predict(xd_te))
    record("domain-guided", value, "all 11 features usable, as the arm ran it")
    out["reproduction"]["domain-guided"] = _repro_check("domain-guided", value)
    assert C.usable(DG) == DG, "domain-guided uses an unusable feature"

    out["elapsed_seconds"] = round(time.time() - t0, 1)
    with (RESULTS / "accuracy.json").open("w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote results/accuracy.json in {out['elapsed_seconds']}s")
    return out


def _repro_check(arm: str, value: float) -> dict:
    reported = REPORTED[arm]["rmse"]
    pct = 100 * (value / reported - 1)
    ok = abs(pct) <= REPRO_TOL[arm] * 100
    print(
        f"  reproduction [{arm}]: {'OK' if ok else 'MISMATCH'} "
        f"(reported {reported:.6f}, got {value:.6f}, {pct:+.2f}%)"
    )
    return {
        "reported": reported,
        "reproduced": round(value, 6),
        "delta_pct": round(pct, 2),
        "tolerance_pct": round(REPRO_TOL[arm] * 100, 1),
        "ok": ok,
    }


if __name__ == "__main__":
    run()

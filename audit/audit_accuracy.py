"""Reported vs audited accuracy for both arms' final models.

`reported` is what the agent's own harness printed. `audited` is the same model
scored under the RouteE Compass deployment contract in `contract.py`. For an arm that
stayed inside the contract the two numbers are the same by construction; the gap for
the other arm is the finding.

The unguided arm's model is a three-part blend, so the audit scores the parts as well
as the whole. Configurations, chosen so a skeptical reader can attack the result from
either side:

  unguided/reported            the full blend, as run — reproduces the archived number
  unguided/as-run/gbdt-only    the tree ensemble alone, still on all 13 features
  unguided/as-run/no-offset    trees + sequence member, without the journey offset
  unguided/audited-mean        trees only, out-of-contract inputs absent (train mean)
  unguided/audited-contract    trees only, out-of-contract inputs replaced by the best
                               value LINK+LOOKBACK can supply. The Compass number.
  unguided/audited-state       trees only, plus the one out-of-contract feature that
                               is genuinely causal (prev2_speed), priced in label memory
  unguided/retrained-contract  the same GBDT recipe *refit* inside the contract. The
                               fair fight: a model built for the constraint, not a
                               contract-violating model with its inputs removed.
  unguided/retrained-state     refit with prev2_speed allowed
  unguided/retrained-causal-seq  refit blend with a CAUSAL sequence member — the one
                               configuration that asks whether the CNN itself was the
                               value, or only its view of the future

  domain-guided/reported       11 features, as run
  domain-guided/audited        identical — every feature is already in contract

The three `as-run/*` rows exist because a feature ledger alone would miss two of this
arm's three components. The sequence member reads six links ahead structurally, and
the journey offset is built from labels. Decomposing the blend is what shows how much
of the reported score depends on each.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

import contract as C
import features as F
import models as M
from harness import evaluate, train_test_split

CACHE = Path("cache")
RESULTS = Path("results")

#: What each arm's harness printed at its final commit.
REPORTED = {
    "unguided": {"commit": "bc07466", "rmse": 0.004785, "trip_rmse": 0.001231},
    "domain-guided": {"commit": "e315384", "rmse": 0.006823, "trip_rmse": 0.002239},
}
BASELINE = {"rmse": 0.013345, "trip_rmse": 0.003037}

#: Reproduction tolerance. The domain-guided arm is deterministic (fixed 320 epochs)
#: and is held to 2%. The unguided arm's sequence member trains for a wall-clock
#: budget rather than a fixed epoch count, so a different host does a different number
#: of updates; it gets 5%. That asymmetry is a property of the arms, not of the audit.
REPRO_TOL = {
    "unguided": {"rmse": 0.05, "trip_rmse": 0.05},
    "domain-guided": {"rmse": 0.02, "trip_rmse": 0.02},
}


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


# --- feature-set construction ---------------------------------------------------


def build_matrix(
    df: pd.DataFrame,
    features: list[str],
    mode: str,
    train_means: dict[str, float] | None = None,
) -> pd.DataFrame:
    """The 13 columns the unguided model expects, filled per `mode`.

    Returned as a named frame rather than a bare array so the estimator matches
    columns by name, exactly as it was fitted. Order is preserved either way —
    substitution happens in place rather than by dropping columns.

    `mode`:
      as_run    every column as the arm computed it
      mean      out-of-contract columns replaced by their training mean
      contract  out-of-contract columns replaced by a causal substitute where one
                needs no extra label state, otherwise by the training mean
      state     as `contract`, but substitutes that cost label memory are allowed
    """
    out = np.empty((len(df), len(features)), dtype=np.float64)
    for j, f in enumerate(features):
        v = C.verdict(f)
        if mode == "as_run" or v.deployable:
            out[:, j] = df[f].to_numpy(dtype=np.float64)
            continue
        sub = v.causal_substitute is not None
        allowed = sub and (
            mode == "state" or (mode == "contract" and v.label_floats == 0)
        )
        if allowed:
            out[:, j] = df[f + F.CAUSAL].to_numpy(dtype=np.float64)
        elif mode in ("mean", "contract", "state"):
            out[:, j] = train_means[f]
        else:
            raise ValueError(mode)
    return pd.DataFrame(out, columns=features)


def restricted_features(mode: str) -> list[str]:
    """Feature set for a *retrained* unguided model under `mode`.

    `dke_per_mile`'s causal substitute is arithmetically identical to `dke_in_link`,
    which the model already has, so it is not carried into a refit — a duplicate
    column would only perturb `max_features` sampling for no information.
    """
    keep = C.deployable(C.UNGUIDED_FEATURES)
    if mode == "contract":
        return keep
    if mode == "state":
        return keep + ["prev2_speed"]
    raise ValueError(mode)


def _train_means(train_df: pd.DataFrame, features: list[str]) -> dict[str, float]:
    return {f: float(train_df[f].mean()) for f in features}


# --- the audit ------------------------------------------------------------------


def run() -> dict:
    t0 = time.time()
    RESULTS.mkdir(exist_ok=True)

    print("loading and building features ...", flush=True)
    df = F.load_sorted()
    df = F.build_all(df)
    df = df.drop(columns=["geometry"])
    # Survives the split, so the sequence member can be handed the same train/test
    # mask in the journey-ordered frame it needs. The arm does exactly this.
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
    out: dict = {
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "baseline": BASELINE,
        "reported": REPORTED,
        "ledger": {
            "unguided": C.ledger_rows(UG),
            "domain-guided": C.ledger_rows(C.DOMAIN_GUIDED_FEATURES),
        },
        "components": C.component_rows(),
        "configs": {},
    }

    def record(key: str, metrics: dict, note: str) -> None:
        out["configs"][key] = {
            "rmse": round(metrics["rmse"], 6),
            "trip_rmse": round(metrics["trip_rmse"], 6),
            "vs_baseline_rmse_pct": round(
                100 * (metrics["rmse"] / BASELINE["rmse"] - 1), 2
            ),
            "vs_baseline_trip_pct": round(
                100 * (metrics["trip_rmse"] / BASELINE["trip_rmse"] - 1), 2
            ),
            "note": note,
        }
        print(
            f"  {key:36s} rmse {metrics['rmse']:.6f}  trip {metrics['trip_rmse']:.6f}",
            flush=True,
        )

    def score(pred: np.ndarray) -> dict:
        return evaluate(y_test, pred, journey_id=jid_te, miles=miles_te)

    # -- unguided: the three parts of the as-run blend --------------------------
    print("\n[1/5] unguided: fitting the as-run blend ...", flush=True)
    print("  (a) 5 x HistGBR on 13 features", flush=True)
    ens = _cached(
        "unguided_as_run",
        lambda: M.UnguidedEnsemble().fit(train_df[UG], y_train),
    )
    print("  (b) out-of-fold residual (2 more ensemble fits)", flush=True)
    resid = _cached(
        "unguided_oof_residual",
        lambda: M.out_of_fold_residual(train_df, y_train, UG),
    )
    print("  (c) bidirectional sequence member (60s)", flush=True)
    seq_feats = C.UNGUIDED_SEQUENCE_FEATURES
    seq = _cached(
        "unguided_sequence",
        lambda: M.sequence_predictions(df, tr_rows, seq_feats, causal=False),
    )

    gbdt_te = ens.predict(build_matrix(test_df, UG, "as_run"))
    seq_te = seq[te_rows]
    blend_te = (1 - M.BLEND) * gbdt_te + M.BLEND * seq_te

    # The arm measures the offset against the *blend's* residual, not the trees'.
    blend_resid = (1 - M.BLEND) * resid + M.BLEND * (y_train - seq[tr_rows])
    offset = M.journey_offset(train_df, test_df, blend_resid)

    m = score(blend_te + offset)
    record("unguided/reported", m, "the full blend, 13 features, as the arm ran it")
    out["reproduction"] = {"unguided": _repro_check("unguided", m)}

    record(
        "unguided/as-run/gbdt-only",
        score(gbdt_te),
        "tree ensemble alone; no sequence member, no journey offset",
    )
    record(
        "unguided/as-run/no-offset",
        score(blend_te),
        "trees + sequence member; no journey offset",
    )

    # -- unguided: audited under each fill mode --------------------------------
    print("\n[2/5] unguided: scoring under the deployment contract ...", flush=True)
    means = _train_means(train_df, UG)
    for mode, note in [
        ("mean", "as-trained trees; out-of-contract inputs absent (train mean)"),
        (
            "contract",
            "as-trained trees; strict LINK+LOOKBACK substitution — the Compass number",
        ),
        (
            "state",
            "as-trained trees; prev2_speed restored (costs 1 float per search label)",
        ),
    ]:
        xa = build_matrix(test_df, UG, mode, means)
        record(f"unguided/audited-{mode}", score(ens.predict(xa)), note)

    # -- unguided: retrained for the contract ----------------------------------
    print("\n[3/5] unguided: refitting the same recipe on restricted features ...")
    refits: dict[str, M.UnguidedEnsemble] = {}
    for mode, note in [
        ("contract", "refit on LINK+LOOKBACK features only — the fair fight"),
        ("state", "refit with prev2_speed allowed"),
    ]:
        feats = restricted_features(mode)
        mdl = _cached(
            f"unguided_retrained_{mode}",
            lambda f=feats: M.UnguidedEnsemble().fit(train_df[f], y_train),
        )
        refits[mode] = mdl
        record(
            f"unguided/retrained-{mode}",
            score(mdl.predict(test_df[feats])),
            f"{note} ({len(feats)} features)",
        )
        out["configs"][f"unguided/retrained-{mode}"]["features"] = feats

    # -- unguided: was the sequence member the value, or its view of the future? --
    print("\n[4/5] unguided: refitting the blend with a CAUSAL sequence member ...")
    causal_seq_feats = C.deployable(seq_feats)
    causal_seq = _cached(
        "unguided_causal_sequence",
        lambda: M.sequence_predictions(df, tr_rows, causal_seq_feats, causal=True),
    )
    contract_feats = restricted_features("contract")
    causal_blend = (1 - M.BLEND) * refits["contract"].predict(
        test_df[contract_feats]
    ) + M.BLEND * causal_seq[te_rows]
    record(
        "unguided/retrained-causal-seq",
        score(causal_blend),
        (
            f"contract trees + left-masked CNN over {len(causal_seq_feats)} deployable "
            "features; no journey offset"
        ),
    )
    out["causal_sequence"] = {
        "features": causal_seq_feats,
        "half_width_links": M.SEQ_HALF_WIDTH,
        "blend": M.BLEND,
    }

    # -- domain-guided ---------------------------------------------------------
    print("\n[5/5] domain-guided: fitting as-run model (2 x 128 MLP, 320 epochs) ...")
    DG = C.DOMAIN_GUIDED_FEATURES
    xd_tr = F.domain_guided_matrix(train_df, DG)
    xd_te = F.domain_guided_matrix(test_df, DG)
    dg = _cached(
        "domain_guided_as_run",
        lambda: M.DomainGuidedMLP(DG).fit(xd_tr, y_train),
    )
    m = score(dg.predict(xd_te))
    record("domain-guided/reported", m, "11 features, as the arm ran it")
    out["reproduction"]["domain-guided"] = _repro_check("domain-guided", m)

    # Every domain-guided feature is in contract, so the audited configuration is
    # the reported one. Assert it in code rather than claiming it in prose.
    undeployable = C.undeployable(DG)
    assert not undeployable, (
        f"domain-guided has out-of-contract features: {undeployable}"
    )
    record(
        "domain-guided/audited",
        m,
        "identical to reported: all 11 features are inside the contract",
    )

    out["elapsed_seconds"] = round(time.time() - t0, 1)
    with (RESULTS / "accuracy.json").open("w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote results/accuracy.json in {out['elapsed_seconds']}s")
    return out


def _repro_check(arm: str, m: dict) -> dict:
    rep = REPORTED[arm]
    tol = REPRO_TOL[arm]
    d = {}
    for k in ("rmse", "trip_rmse"):
        pct = 100 * (m[k] / rep[k] - 1)
        d[k] = {
            "archived": rep[k],
            "reproduced": round(m[k], 6),
            "delta_pct": round(pct, 3),
            "tolerance_pct": round(tol[k] * 100, 1),
            "within_tolerance": abs(pct) <= tol[k] * 100,
        }
    ok = all(v["within_tolerance"] for v in d.values())
    print(f"  reproduction [{arm}]: {'OK' if ok else 'MISMATCH'}")
    for k, v in d.items():
        print(
            f"    {k}: archived {v['archived']:.6f} -> {v['reproduced']:.6f} "
            f"({v['delta_pct']:+.2f}%, tol +/-{v['tolerance_pct']}%)"
        )
    return d


if __name__ == "__main__":
    run()

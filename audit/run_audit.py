"""Run both audits, write results/report.md, and draw the figures in ../img/.

    pixi run python run_audit.py             # everything
    pixi run python run_audit.py --accuracy  # link RMSE only
    pixi run python run_audit.py --inference # inference time only (needs cached models)
    pixi run python run_audit.py --report    # re-render the report from cached json

Trained models are cached in `cache/`. Delete it to force a refit.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

RESULTS = Path("results")

#: Link traversals in a large route search, for the wall-clock example.
SEARCH_LINKS = 1_000_000


def render_report(acc: dict, inf: dict | None) -> str:
    L: list[str] = []
    w = L.append
    cfg = acc["configs"]
    base = acc["baseline_rmse"]

    w("# Audit report")
    w("")
    w("Made by `run_audit.py`. Both models are scored on the same held-out links")
    w(f"({acc['n_test']:,} of {acc['n_train'] + acc['n_test']:,}), using the same")
    w("`harness.py` split and metric the arms used.")
    w("")

    # -- 1. reproduction ---------------------------------------------------------
    w("## 1. Does the audit reproduce the reported numbers?")
    w("")
    w("| arm | reported link RMSE | reproduced | difference | allowed |")
    w("| --- | --- | --- | --- | --- |")
    ok = True
    for arm, r in acc["reproduction"].items():
        ok &= r["ok"]
        w(
            f"| {arm} | {r['reported']:.6f} | {r['reproduced']:.6f} | "
            f"{r['delta_pct']:+.2f}% | ±{r['tolerance_pct']:.0f}% |"
        )
    w("")
    if ok:
        w("**Yes.** Both arms reproduce within tolerance.")
    else:
        w("**No.** At least one arm is outside tolerance. Treat everything below")
        w("as unverified.")
    w("")
    w("The unguided arm's sequence model trains for 60 seconds of wall clock, not a")
    w("fixed number of steps, so a different machine fits a slightly different")
    w("model. It is allowed 5%. The domain-guided model is deterministic and is")
    w("allowed 2%.")
    w("")

    # -- 2. usable inputs --------------------------------------------------------
    w("## 2. Which inputs can Compass supply?")
    w("")
    w("Compass scores one link at a time while it searches for a route. It knows")
    w("the current link (speed, grade, distance, geometry) and the one link before")
    w("it. It does not know which links come next. It has no energy labels.")
    w("")
    for arm in ("unguided", "domain-guided"):
        rows = acc["features"][arm]
        n_ok = sum(r["usable"] for r in rows)
        w(f"### {arm}: {n_ok} of {len(rows)} features usable")
        w("")
        w("| feature | usable | why |")
        w("| --- | --- | --- |")
        for r in rows:
            mark = "yes" if r["usable"] else "**no**"
            w(f"| `{r['feature']}` | {mark} | {r['reason']} |")
        w("")
        if arm == "unguided":
            w("The unguided model also has two parts that are not features. Neither")
            w("can run in Compass.")
            w("")
            for name, why in acc["components"].items():
                w(f"- **{name}**: {why}")
            w("")

    # -- 3. link rmse ------------------------------------------------------------
    w("## 3. Link RMSE")
    w("")
    w(f"Baseline is the random forest both arms started from ({base:.6f}).")
    w("Lower is better.")
    w("")
    w("| model | link RMSE | vs baseline | what it is |")
    w("| --- | --- | --- | --- |")
    for key, c in cfg.items():
        w(f"| `{key}` | {c['rmse']:.6f} | {c['vs_baseline_pct']:+.1f}% | {c['note']} |")
    w("")

    rep, ret, dg = (
        cfg["unguided/reported"],
        cfg["unguided/retrained"],
        cfg["domain-guided"],
    )

    def rel(a: dict, b: dict) -> float:
        return 100 * (a["rmse"] / b["rmse"] - 1)

    w("What this says:")
    w("")
    w(f"- The unguided model reported {rep['rmse']:.6f}, the best number here. But")
    w("  that model cannot run in Compass. It needs future links, simulation")
    w("  timestamps and energy labels, none of which a route search has.")
    w(f"- Refit on usable inputs only, the unguided recipe gives {ret['rmse']:.6f}.")
    w("  This is the fair comparison.")
    w(f"- The domain-guided model scores {dg['rmse']:.6f}. Every input it uses is")
    better = "better" if ret["rmse"] < dg["rmse"] else "worse"
    w(
        f"  available in Compass. The unguided refit is {abs(rel(ret, dg)):.0f}% {better}."
    )
    w("")

    # -- 4. inference ------------------------------------------------------------
    w("## 4. Inference time")
    w("")
    if inf is None:
        w("_Not run._")
        return "\n".join(L) + "\n"

    w("One CPU thread, like one Compass search thread. Microseconds per link,")
    w("median over repeated calls, at several batch sizes.")
    w("")
    w("| batch | unguided as run | unguided retrained | domain-guided |")
    w("| --- | --- | --- | --- |")
    for b, row in inf["us_per_link"].items():
        w(
            f"| {int(b):,} | {row['unguided/as-run']:,.1f} | "
            f"{row['unguided/retrained']:,.1f} | {row['domain-guided']:,.2f} |"
        )
    w("")
    w("| model | what it is |")
    w("| --- | --- |")
    for k, v in inf["what"].items():
        w(f"| `{k}` | {v} |")
    w("")
    r512 = inf["us_per_link"]["512"]
    ratio = r512["unguided/retrained"] / r512["domain-guided"]
    s_dg = r512["domain-guided"] * SEARCH_LINKS / 1e6
    s_ug = r512["unguided/retrained"] * SEARCH_LINKS / 1e6
    w("What this says:")
    w("")
    w(f"- At batch 512, the domain-guided model is **{ratio:.0f}x** faster per link")
    w("  than the unguided model refit on usable inputs.")
    w(f"- A route search that scores {SEARCH_LINKS:,} links would take about")
    w(f"  {s_dg:.1f} s with the domain-guided model and {s_ug:.0f} s with the")
    w("  unguided refit.")
    w("- The gap is a model choice, not a feature choice. The unguided arm built")
    w("  10,000 boosted trees. The domain-guided arm built two small networks,")
    w("  because `domain.md` told it inference cost mattered.")
    w("- These are Python timings. Compass is Rust, so every model would be faster")
    w("  there. The ratio is what carries over.")
    w("")
    return "\n".join(L) + "\n"


def main() -> int:
    args = set(sys.argv[1:])
    only_report = "--report" in args
    do_acc = not only_report and ("--inference" not in args or "--accuracy" in args)
    do_inf = not only_report and ("--accuracy" not in args or "--inference" in args)

    if do_acc:
        import audit_accuracy

        audit_accuracy.run()

    if do_inf:
        print("\n" + "=" * 72)
        print("inference benchmark (subprocess, thread-pinned)")
        print("=" * 72, flush=True)
        r = subprocess.run([sys.executable, "audit_inference.py"], check=False)
        if r.returncode != 0:
            print("inference benchmark failed", file=sys.stderr)

    acc_p, inf_p = RESULTS / "accuracy.json", RESULTS / "inference.json"
    if not acc_p.exists():
        print("no results/accuracy.json; run without --report first", file=sys.stderr)
        return 1
    acc = json.loads(acc_p.read_text())
    inf = json.loads(inf_p.read_text()) if inf_p.exists() else None

    report = render_report(acc, inf)
    (RESULTS / "report.md").write_text(report)
    print(f"\nwrote results/report.md ({len(report.splitlines())} lines)")

    import plot

    plot.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

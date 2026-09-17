"""Two figures from the audit results, written to ../img/.

    pixi run python plot.py

  audit-error.png       link RMSE of every model, grouped by whether it can run
                        in RouteE Compass
  audit-inference.png   time to score one million links, one CPU thread

Reads results/accuracy.json and results/inference.json. Run the audit first.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

RESULTS = Path("results")
IMG = Path("../img")

SEARCH_LINKS = 1_000_000
BATCH = "512"

#: Half of a 16:9 slide, so two figures sit side by side.
FIGSIZE = (7.2, 6.0)

# colour follows the arm; hollow means "cannot run in Compass"
BLUE = "#2a78d6"  # domain-guided
ORANGE = "#eb6834"  # unguided
GRAY = "#898781"  # baseline
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "xtick.color": MUTED,
        "ytick.color": INK,
        "axes.edgecolor": AXIS,
    }
)


def _hbar_figure(rows, *, title, subtitle, xlabel, fmt, legend):
    """Horizontal bars sized for half of a 16:9 slide, two of these side by side.

    Model names sit above their bars so the figure stays narrow. `rows` is a
    list of (group, label, value, colour, runnable). Runnable models are solid;
    the rest are hollow with a dashed edge.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=200)
    fig.subplots_adjust(left=0.06, right=0.96, top=0.80, bottom=0.17)

    y = 0.0
    last_group = None
    for group, label, value, colour, runnable in rows:
        if group != last_group:
            y -= 0.9
            ax.text(
                0,
                y,
                group,
                ha="left",
                va="center",
                fontsize=13,
                color=INK2,
                fontweight="bold",
            )
            y -= 0.7
            last_group = group
        y -= 0.55
        ax.text(0, y + 0.55, label, ha="left", va="center", fontsize=12, color=INK)
        if runnable:
            ax.barh(y, value, height=0.5, color=colour, edgecolor="none")
        else:
            ax.barh(
                y,
                value,
                height=0.5,
                facecolor="none",
                edgecolor=colour,
                linewidth=1.6,
                linestyle=(0, (3, 2)),
            )
        ax.text(
            value,
            y,
            "  " + fmt(value),
            va="center",
            ha="left",
            fontsize=12,
            color=INK,
        )
        y -= 0.75

    ax.set_yticks([])
    ax.set_ylim(y, 0)
    ax.set_xlim(0, max(r[2] for r in rows) * 1.45)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.tick_params(axis="x", labelsize=11)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)

    fig.text(0.06, 0.95, title, fontsize=17, fontweight="bold", va="top")
    fig.text(0.06, 0.875, subtitle, fontsize=12, color=INK2, va="top")
    fig.legend(
        handles=legend,
        loc="lower left",
        bbox_to_anchor=(0.04, 0.0),
        ncol=len(legend),
        frameon=False,
        fontsize=11,
        handlelength=1.4,
        columnspacing=1.2,
    )
    return fig


def error_figure(acc: dict):
    cfg = acc["configs"]
    base = acc["baseline_rmse"]

    def row(group, label, value, colour, runnable=True):
        return (group, label, value, colour, runnable)

    can = "Can run in Compass"
    cannot = "Cannot run in Compass"
    rows = [
        row(can, "domain-guided", cfg["domain-guided"]["rmse"], BLUE),
        row(
            can,
            "unguided, refit on usable inputs",
            cfg["unguided/retrained"]["rmse"],
            ORANGE,
        ),
        row(can, "baseline random forest", base, GRAY),
        row(
            cannot,
            "unguided, as reported",
            cfg["unguided/reported"]["rmse"],
            ORANGE,
            False,
        ),
    ]
    # best first inside each group
    rows.sort(key=lambda r: (r[0] != can, r[2]))

    def fmt(v):
        if v == base:
            return f"{v:.4f}"
        return f"{v:.4f}  ({100 * (v / base - 1):+.0f}%)"

    legend = [
        Patch(color=BLUE, label="domain-guided"),
        Patch(color=ORANGE, label="unguided"),
        Patch(color=GRAY, label="baseline"),
    ]
    return _hbar_figure(
        rows,
        title="Link RMSE on held-out links",
        subtitle=f"{acc['n_test']:,} links. Lower is better. Change is vs baseline.",
        xlabel="link RMSE (GGE per mile)",
        fmt=fmt,
        legend=legend,
    )


def inference_figure(inf: dict):
    lat = inf["us_per_link"][BATCH]

    def seconds(key):
        return lat[key] * SEARCH_LINKS / 1e6

    can = "Can run in Compass"
    cannot = "Cannot run in Compass"
    rows = [
        (can, "domain-guided", seconds("domain-guided"), BLUE, True),
        (
            can,
            "unguided, refit on usable inputs",
            seconds("unguided/retrained"),
            ORANGE,
            True,
        ),
        (cannot, "unguided, as reported", seconds("unguided/as-run"), ORANGE, False),
    ]

    def fmt(s):
        return f"{s:.1f} s" if s < 60 else f"{s:.0f} s  ({s / 60:.0f} min)"

    ratio = lat["unguided/retrained"] / lat["domain-guided"]

    legend = [
        Patch(color=BLUE, label="domain-guided"),
        Patch(color=ORANGE, label="unguided"),
    ]
    return _hbar_figure(
        rows,
        title=f"Time to score {SEARCH_LINKS:,} links",
        subtitle=f"One CPU thread, batches of {BATCH}. Domain-guided is {ratio:.0f}x faster.",
        xlabel="seconds",
        fmt=fmt,
        legend=legend,
    )


def main() -> None:
    IMG.mkdir(exist_ok=True)
    acc = json.loads((RESULTS / "accuracy.json").read_text())
    error_figure(acc).savefig(IMG / "audit-error.png")
    print(f"wrote {IMG / 'audit-error.png'}")

    inf_p = RESULTS / "inference.json"
    if inf_p.exists():
        inference_figure(json.loads(inf_p.read_text())).savefig(
            IMG / "audit-inference.png"
        )
        print(f"wrote {IMG / 'audit-inference.png'}")


if __name__ == "__main__":
    main()

"""The deployment contract, as data.

`domain.md` describes the RouteE Compass inference environment in prose. This module
restates it as a machine-readable ledger so that "N of M features are not deployable"
is a checkable assertion rather than an opinion in a paper.

Only the `domain-guided` arm was shown `domain.md`. Applying the contract to the
`unguided` arm is *not* an accusation that it broke a rule it was given — it was given
no such rule, and its `learnings.md` reasons carefully about the features it chose. It
is a measurement of the gap between what that arm optimised and what Compass can
actually execute.

Six availability tiers. The first two are inside the contract; the other four are
outside it for four *different* reasons, and keeping them apart is the point — each
failure mode has a different remedy, or none.

`LINK`     Static link attributes, or the current link's own averages. Free.
`LOOKBACK` Requires the single previously-traversed link. `domain.md` permits this
           explicitly and caps it here: one link, no more.
`STATE`    Causal — reads only links already traversed — but needs more than the one
           permitted link, so every extra scalar becomes part of the search label.
           Buyable, at a memory price this audit quantifies.
`FUTURE`   Requires links not yet traversed. Not obtainable at any price during a
           forward search: the route is the thing being computed.
`TRACE`    Exists only in the simulated 1 Hz drive-cycle trace the training data was
           aggregated from. There is no map-data counterpart at any price — this is
           not a search-engine limitation but an observability one.
`LABEL`    Derived from the *target* of other links. At inference there are no labels
           at all, so these are not features in any deployable sense.

Only `LINK` and `LOOKBACK` are inside the contract.

Why `TRACE` and `LABEL` are new this round: the previous pair of trees produced an
unguided model whose out-of-contract features were all aggregates of *observable*
quantities (speed, grade, geometry), so every one of them had a causal prefix
substitute and the interesting question was what that substitute cost. This arm's
model reaches further. `gap_seconds` is idle time read off the simulation trace, and
`journey_rate` is a leave-one-out mean of the target itself. Neither has a substitute
at any tier, and calling them `STATE` would imply a price exists.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- tiers ---------------------------------------------------------------------

LINK = "LINK"
LOOKBACK = "LOOKBACK"
STATE = "STATE"
FUTURE = "FUTURE"
TRACE = "TRACE"
LABEL = "LABEL"

TIERS = (LINK, LOOKBACK, STATE, FUTURE, TRACE, LABEL)

#: Tiers a Compass link-cost function may read under `domain.md` as written.
IN_CONTRACT = frozenset({LINK, LOOKBACK})

#: Tiers a forward search could compute if it paid for label memory.
CAUSAL_TIERS = frozenset({LINK, LOOKBACK, STATE})


@dataclass(frozen=True)
class FeatureVerdict:
    """One feature, its availability tier, and the clause that decides it."""

    tier: str
    reason: str
    #: The sentence of `domain.md` the verdict rests on, or "" for LINK/LOOKBACK
    #: features that the document affirmatively permits.
    clause: str = ""
    #: Best causal substitute if the engine carries running state. `None` means
    #: nothing better than the training mean is recoverable — which for `TRACE` and
    #: `LABEL` features is a statement about the world, not about Compass.
    causal_substitute: str | None = None
    #: Extra scalars a Compass search label must carry to supply the substitute.
    #: 0 for in-contract features; None where no substitute exists.
    label_floats: int | None = 0

    @property
    def deployable(self) -> bool:
        return self.tier in IN_CONTRACT

    @property
    def causally_recoverable(self) -> bool:
        """True if a forward search could compute *something* for this feature."""
        return self.tier in CAUSAL_TIERS or self.causal_substitute is not None


_PERMITTED = (
    "Our inference environment has the following features: Average Speed, "
    "Average Road Gradient, Link Distance, Geometry."
)
_ONE_LINK = (
    "If you're considering any kind of link sequencing, we only want to include a "
    "single one link lookback."
)
_NO_FUTURE = (
    "we will only have the context of the previous links that have been traversed "
    "and know nothing about the future links that might be traversed."
)
_MEMORY = (
    "our current search harness would incur a huge memory penalty for trying to "
    "enumerate new labels with that much previous context."
)
_LIMITED = (
    "when we're applying these models for inference, we only have limited data "
    "(which is why we're developing these models in the first place)."
)


# --- the ledger ----------------------------------------------------------------

#: Every feature appearing in either arm's final model.
LEDGER: dict[str, FeatureVerdict] = {
    # -- shared base columns, affirmatively permitted -------------------------
    "speed_mph": FeatureVerdict(
        LINK, "link-average speed; posted or probe", _PERMITTED
    ),
    "grade_percent": FeatureVerdict(LINK, "link-average gradient", _PERMITTED),
    "miles": FeatureVerdict(LINK, "link distance", _PERMITTED),
    # -- unguided arm ---------------------------------------------------------
    "dke_in_link": FeatureVerdict(
        LOOKBACK,
        "(speed_mph^2 - prev_speed^2) / miles — kinetic-energy change from the "
        "entry speed to this link's own speed, from one link of lookback. This is "
        "the same physics the domain-guided arm reached as ke_delta_per_mile.",
        _ONE_LINK,
    ),
    "prev_miles": FeatureVerdict(
        LOOKBACK, "distance of the single previous link", _ONE_LINK
    ),
    "dke_per_mile": FeatureVerdict(
        FUTURE,
        "(next_speed^2 - prev_speed^2) / miles. Spans the link from the PREVIOUS "
        "link's speed to the NEXT link's, so it inherits the next link — which is "
        "what the search is deciding. Its causal half is already in the model as "
        "dke_in_link.",
        _NO_FUTURE,
        causal_substitute="(speed_mph^2 - prev_speed^2) / miles  [= dke_in_link]",
        label_floats=0,
    ),
    "next_miles": FeatureVerdict(
        FUTURE,
        "distance of the FOLLOWING link. Not chosen yet when the current link is "
        "scored; no amount of label state recovers it.",
        _NO_FUTURE,
        causal_substitute=None,
        label_floats=None,
    ),
    "dv_out": FeatureVerdict(
        FUTURE,
        "next_speed - speed_mph, the exit speed change. Inherits the next link.",
        _NO_FUTURE,
        causal_substitute=None,
        label_floats=None,
    ),
    "prev2_speed": FeatureVerdict(
        STATE,
        "speed of the link TWO back. Causal — the vehicle has traversed it — but "
        "domain.md caps the lookback at one link, so this is one extra float every "
        "search label must carry, and labels differing in it cannot be merged. The "
        "one out-of-contract feature in this model that has an honest price.",
        _MEMORY,
        causal_substitute="prev2_speed (unchanged — it is already causal)",
        label_floats=1,
    ),
    "gap_seconds": FeatureVerdict(
        TRACE,
        "idle seconds between the previous link ending and this one starting, from "
        "link_start_time / link_end_time. Those columns are artifacts of the "
        "simulated drive-cycle trace; a map has no counterpart, so this is not "
        "recoverable at any memory price. The arm's own reasoning is sound — a stop "
        "means the entry speed was really 0 — but the signal is unobservable at "
        "inference.",
        _LIMITED,
        causal_substitute=None,
        label_floats=None,
    ),
    "next_gap_seconds": FeatureVerdict(
        TRACE,
        "the exit-side mirror of gap_seconds. Fails twice over: it is trace-only AND "
        "it reads the following link.",
        _LIMITED,
        causal_substitute=None,
        label_floats=None,
    ),
    "journey_rate": FeatureVerdict(
        LABEL,
        "leave-one-out mean of the TARGET over the journey's training links, shrunk "
        "to the global mean. This is not a feature at inference: there are no labels "
        "to average, and the average spans links the search has not chosen. It is "
        "also the model's single largest lever.",
        _NO_FUTURE,
        causal_substitute=None,
        label_floats=None,
    ),
    "journey_gge_per_mile": FeatureVerdict(
        LABEL,
        "the same journey offset weighted by distance — training energy_gge over "
        "training miles. Target-derived, journey-wide, same verdict.",
        _NO_FUTURE,
        causal_substitute=None,
        label_floats=None,
    ),
    # -- domain-guided arm ----------------------------------------------------
    "prev_speed_mph": FeatureVerdict(
        LOOKBACK, "speed of the single previously-traversed link", _ONE_LINK
    ),
    "ke_delta_per_mile": FeatureVerdict(
        LOOKBACK,
        "(speed_mph^2 - prev_speed_mph^2) / (2 * miles) — kinetic-energy change "
        "across the entry boundary, from one link of lookback",
        _ONE_LINK,
    ),
    "sinuosity": FeatureVerdict(
        LINK,
        "travelled length over endpoint crow-flight distance; a static road "
        "attribute precomputed offline from geometry",
        _PERMITTED,
    ),
    "vertices_per_mile": FeatureVerdict(
        LINK,
        "linestring vertex density; a static road attribute standing in for road class",
        _PERMITTED,
    ),
    "junction_turn_degrees": FeatureVerdict(
        LOOKBACK,
        "heading change from the previous link's exit bearing into this link's entry "
        "bearing; both bearings are static, the pairing needs one link of lookback",
        _ONE_LINK,
    ),
    "prev_grade_percent": FeatureVerdict(
        LOOKBACK, "gradient of the single previous link", _ONE_LINK
    ),
    "prev_sinuosity": FeatureVerdict(
        LOOKBACK, "sinuosity of the single previous link", _ONE_LINK
    ),
}


# --- model components beyond the feature list ----------------------------------

#: The unguided arm's model is not only a feature vector. Two of its three parts are
#: outside the contract for reasons no feature ledger would catch, so they are stated
#: here and scored separately in `audit_accuracy.py`.
COMPONENTS: dict[str, FeatureVerdict] = {
    "sequence_member": FeatureVerdict(
        FUTURE,
        "a 2-block dilated 1-D CNN over each journey's whole link chain. Kernel 5 at "
        "dilations 1 and 2 gives a receptive field of 13 links — SIX AHEAD as well as "
        "six behind. It reads the future structurally, not through a named feature, "
        "so no feature ledger catches it. A left-masked version of the same network "
        "IS causal, and this audit fits one to price the difference.",
        _NO_FUTURE,
        causal_substitute="the same CNN with causal (left-only) padding",
        label_floats=None,
    ),
    "journey_offset": FeatureVerdict(
        LABEL,
        "a distance-weighted per-journey correction built from the training "
        "residuals of that journey's own links. Needs labels, and needs the whole "
        "journey. Not deployable in a route search at any price.",
        _NO_FUTURE,
        causal_substitute=None,
        label_floats=None,
    ),
}


# --- the two arms' final feature sets ------------------------------------------

#: unguided, commit bc07466 (exp48) — train.py LINK_FEATURES, in order.
UNGUIDED_FEATURES: list[str] = [
    "speed_mph",
    "grade_percent",
    "miles",
    "dke_per_mile",
    "dke_in_link",
    "gap_seconds",
    "prev_miles",
    "next_miles",
    "next_gap_seconds",
    "prev2_speed",
    "dv_out",
    "journey_rate",
    "journey_gge_per_mile",
]

#: The features the unguided arm's sequence member sees — the per-link ones only.
#: Transcribed from `SEQUENCE_FEATURES` in its train.py.
UNGUIDED_SEQUENCE_FEATURES: list[str] = [
    f for f in UNGUIDED_FEATURES if not f.startswith("journey_")
]

#: domain-guided, commit e315384 (exp50) — train.py LINK_FEATURES, in order.
DOMAIN_GUIDED_FEATURES: list[str] = [
    "speed_mph",
    "grade_percent",
    "miles",
    "prev_speed_mph",
    "ke_delta_per_mile",
    "sinuosity",
    "junction_turn_degrees",
    "prev_grade_percent",
    "prev_miles",
    "prev_sinuosity",
    "vertices_per_mile",
]


def verdict(name: str) -> FeatureVerdict:
    if name not in LEDGER:
        raise KeyError(f"{name!r} has no contract verdict; add one to LEDGER")
    return LEDGER[name]


def deployable(features: list[str]) -> list[str]:
    """The subset of `features` a Compass link-cost function may read."""
    return [f for f in features if verdict(f).deployable]


def undeployable(features: list[str]) -> list[str]:
    return [f for f in features if not verdict(f).deployable]


def split_by_tier(features: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {t: [] for t in TIERS}
    for f in features:
        out[verdict(f).tier].append(f)
    return out


def label_cost(features: list[str]) -> int | None:
    """Extra floats per search label to supply `features` causally.

    `None` means at least one feature has no causal form at all, so the set is not
    implementable at any memory price.
    """
    total = 0
    for f in features:
        v = verdict(f)
        if v.deployable:
            continue
        if v.label_floats is None:
            return None
        total += v.label_floats
    return total


def ledger_rows(features: list[str]) -> list[dict[str, object]]:
    """The ledger for `features`, as plain dicts for JSON / table output."""
    rows = []
    for f in features:
        v = verdict(f)
        rows.append(
            {
                "feature": f,
                "tier": v.tier,
                "deployable": v.deployable,
                "reason": v.reason,
                "clause": v.clause,
                "causal_substitute": v.causal_substitute,
                "label_floats": v.label_floats,
            }
        )
    return rows


def component_rows() -> list[dict[str, object]]:
    rows = []
    for name, v in COMPONENTS.items():
        rows.append(
            {
                "component": name,
                "tier": v.tier,
                "deployable": v.deployable,
                "reason": v.reason,
                "causal_substitute": v.causal_substitute,
            }
        )
    return rows

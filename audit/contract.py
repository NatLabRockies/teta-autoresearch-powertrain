"""Which inputs a RouteE Compass route search can supply.

Compass scores one road link at a time while it searches for a route. When it
scores a link it knows:

- the current link: its speed, grade, distance and geometry
- the one link before it (`domain.md` allows exactly one link of lookback)

It does not know which links come next, because the search is still choosing
them. It has no energy labels, because energy is the thing being predicted.

Only the domain-guided arm was shown `domain.md`. Applying these rules to the
unguided arm is not a claim that it broke a rule. It was given no rule. The
audit measures the gap between what it optimised and what Compass can run.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Feature:
    usable: bool
    reason: str


#: Every feature in either arm's final model.
FEATURES: dict[str, Feature] = {
    # shared
    "speed_mph": Feature(True, "average speed of the current link"),
    "grade_percent": Feature(True, "average grade of the current link"),
    "miles": Feature(True, "length of the current link"),
    # unguided
    "dke_in_link": Feature(
        True, "kinetic energy change from the previous link's speed to this one"
    ),
    "prev_miles": Feature(True, "length of the previous link"),
    "dke_per_mile": Feature(
        False,
        "uses the NEXT link's speed; the search has not chosen that link yet",
    ),
    "next_miles": Feature(False, "length of the NEXT link"),
    "dv_out": Feature(False, "speed change into the NEXT link"),
    "prev2_speed": Feature(
        False, "speed two links back; only one link of lookback is allowed"
    ),
    "gap_seconds": Feature(
        False,
        "idle time before the link, read from the simulated drive trace; a road map has no such value",
    ),
    "next_gap_seconds": Feature(
        False, "idle time after the link; from the trace, and about the NEXT link"
    ),
    "journey_rate": Feature(
        False,
        "average energy of the journey's other links; there are no energy labels at inference",
    ),
    "journey_gge_per_mile": Feature(
        False, "same as journey_rate, weighted by distance"
    ),
    # domain-guided
    "prev_speed_mph": Feature(True, "speed of the previous link"),
    "ke_delta_per_mile": Feature(
        True, "kinetic energy change from the previous link's speed to this one"
    ),
    "sinuosity": Feature(True, "how curvy the link is; computed from its geometry"),
    "junction_turn_degrees": Feature(
        True, "turn angle from the previous link into this one; from geometry"
    ),
    "prev_grade_percent": Feature(True, "grade of the previous link"),
    "prev_sinuosity": Feature(True, "sinuosity of the previous link"),
    "vertices_per_mile": Feature(
        True, "how many geometry points per mile; a stand-in for road class"
    ),
}

#: The unguided model has two parts that are not features. Both are unusable.
COMPONENTS: dict[str, str] = {
    "sequence model": (
        "a small convolutional network over the whole journey. It reads 6 links "
        "ahead as well as 6 links behind."
    ),
    "journey offset": (
        "a per-journey correction built from the energy labels of that journey's "
        "training links. There are no labels at inference."
    ),
}

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

#: What the unguided sequence model sees: the per-link features only.
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


def usable(features: list[str]) -> list[str]:
    return [f for f in features if FEATURES[f].usable]


def rows(features: list[str]) -> list[dict[str, object]]:
    """The feature list as plain dicts, for JSON and the report."""
    return [
        {"feature": f, "usable": FEATURES[f].usable, "reason": FEATURES[f].reason}
        for f in features
    ]

"""
Machine-readable comparison metadata for algorithms, and the rules that decide
whether a given (algorithm, problem, objective, yard geometry) combination is a
fair comparison.

Three attributes are declared on every ``BaseAlgorithm`` subclass:

``geometry``
    The yard the published method reasons about. ``single-bay`` methods treat the
    yard as a flat list of ``num_bays * num_rows`` stacks, so gantry travel
    between bays is invisible to them; ``multi-bay`` methods model the bay axis.

``objectives``
    What the search itself drives toward. ``fixed-rule`` marks deterministic
    construction rules that optimise nothing -- the selected objective is only
    measured afterwards.

``fidelity``
    How close the implementation is to the published method.

Degradation is *derived*, never stored: a single-bay method is perfectly valid
under a relocation-count objective no matter how the yard is shaped, and only
becomes a weak baseline once crane time enters the objective on a yard that
really has two horizontal axes.

Note on what counts as a 3D yard: the platform stores every layout as
bays x rows x tiers, but the single-bay benchmarks map their S stacks onto
S *bays* with one row each (see ``core/benchmarks/caserta.py``). A flat line of
stacks is still 2D, so a yard is only multi-bay when both bays and rows exceed
one -- ``num_bays > 1`` on its own means nothing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

# ================================================================ #
#  Vocabularies                                                      #
# ================================================================ #

GEOMETRY_SINGLE_BAY = "single-bay"
GEOMETRY_MULTI_BAY = "multi-bay"
GEOMETRIES = (GEOMETRY_SINGLE_BAY, GEOMETRY_MULTI_BAY)

GEOMETRY_LABELS = {
    GEOMETRY_SINGLE_BAY: "Single-bay (2D)",
    GEOMETRY_MULTI_BAY: "Multi-bay (3D)",
}

# Objectives a search can drive toward.
OBJECTIVE_RELOCATIONS = "relocations"
OBJECTIVE_CRANE_TIME = "crane_time"
# Not an objective: a deterministic rule that optimises nothing.
OBJECTIVE_FIXED_RULE = "fixed-rule"
OBJECTIVE_VALUES = (
    OBJECTIVE_RELOCATIONS,
    OBJECTIVE_CRANE_TIME,
    OBJECTIVE_FIXED_RULE,
)

OBJECTIVE_LABELS = {
    OBJECTIVE_RELOCATIONS: "Relocation count",
    OBJECTIVE_CRANE_TIME: "Crane working time",
    OBJECTIVE_FIXED_RULE: "Fixed rule (objective measured only)",
}

FIDELITY_FAITHFUL = "faithful"
FIDELITY_ADAPTED = "adapted"
FIDELITY_DEGENERATE = "degenerate"
FIDELITIES = (FIDELITY_FAITHFUL, FIDELITY_ADAPTED, FIDELITY_DEGENERATE)

FIDELITY_LABELS = {
    FIDELITY_FAITHFUL: "Faithful reproduction",
    FIDELITY_ADAPTED: "Platform adaptation",
    FIDELITY_DEGENERATE: "Degenerate baseline",
}

# Assessment outcomes, ordered from best to worst.
STATUS_OK = "ok"
STATUS_DEGRADED = "degraded"
STATUS_INCOMPATIBLE = "incompatible"

# ``ObjectiveSpec.mode`` values under which crane time is part of the objective.
_TIME_AWARE_MODES = frozenset({"crane_time", "weighted"})


# ================================================================ #
#  Reading metadata off an algorithm class                           #
# ================================================================ #

def algorithm_meta(cls: type) -> Dict[str, Any]:
    """Return the comparison metadata of *cls*, falling back to the defaults."""
    return {
        "geometry": getattr(cls, "geometry", GEOMETRY_SINGLE_BAY),
        "objectives": list(
            getattr(cls, "objectives", None) or [OBJECTIVE_RELOCATIONS]
        ),
        "fidelity": getattr(cls, "fidelity", FIDELITY_FAITHFUL),
    }


def optimises(cls: type, objective_mode: str) -> bool:
    """True if the search of *cls* actually drives toward *objective_mode*."""
    objectives = algorithm_meta(cls)["objectives"]
    if OBJECTIVE_FIXED_RULE in objectives:
        return False
    if objective_mode in _TIME_AWARE_MODES:
        return OBJECTIVE_CRANE_TIME in objectives
    return OBJECTIVE_RELOCATIONS in objectives


def supports_multi_bay(cls: type) -> bool:
    """True if *cls* models the bay axis rather than flattening it away."""
    return algorithm_meta(cls)["geometry"] == GEOMETRY_MULTI_BAY


def yard_geometry(num_bays: int, num_rows: int) -> str:
    """
    Classify a layout. Both horizontal axes must exceed one to be multi-bay:
    the single-bay benchmarks store S stacks as S bays with one row each, which
    is still a flat line of stacks.
    """
    return (
        GEOMETRY_MULTI_BAY
        if int(num_bays) > 1 and int(num_rows) > 1
        else GEOMETRY_SINGLE_BAY
    )


# ================================================================ #
#  Fair-comparison assessment                                        #
# ================================================================ #

def assess_compatibility(
    cls: type,
    problem_name: str,
    objective_mode: str = OBJECTIVE_RELOCATIONS,
    num_bays: int = 1,
    num_rows: int = 1,
) -> Dict[str, Any]:
    """
    Judge one (algorithm, problem, objective, geometry) combination.

    Returns ``{"status", "notes"}`` where status is ``ok`` / ``degraded`` /
    ``incompatible`` and notes are human-readable reasons for the GUI.
    """
    compatible = list(getattr(cls, "compatible_problems", []) or [])
    if compatible and problem_name not in compatible:
        return {
            "status": STATUS_INCOMPATIBLE,
            "notes": [
                f"Not registered for {problem_name}; declared for "
                f"{', '.join(compatible)}."
            ],
        }

    meta = algorithm_meta(cls)
    notes: List[str] = []
    status = STATUS_OK

    time_aware_requested = objective_mode in _TIME_AWARE_MODES

    if not optimises(cls, objective_mode):
        status = STATUS_DEGRADED
        if OBJECTIVE_FIXED_RULE in meta["objectives"]:
            notes.append(
                "Deterministic rule: the selected objective is measured "
                "afterwards, never optimised. Use as a baseline only."
            )
        elif time_aware_requested:
            notes.append(
                "Minimises relocation count only; crane time is reported but "
                "not optimised."
            )
        else:
            notes.append(
                "Does not minimise relocation count; the reported value is a "
                "by-product of a different objective."
            )

    if (
        time_aware_requested
        and yard_geometry(num_bays, num_rows) == GEOMETRY_MULTI_BAY
        and meta["geometry"] == GEOMETRY_SINGLE_BAY
    ):
        status = STATUS_DEGRADED
        notes.append(
            f"Single-bay method on a {num_bays}x{num_rows} yard: stacks are "
            "flattened, so gantry travel between bays is not part of its "
            "decisions."
        )

    if meta["fidelity"] == FIDELITY_ADAPTED:
        notes.append(
            "Platform adaptation rather than a line-by-line reproduction; "
            "published numbers are not expected to match."
        )
    elif meta["fidelity"] == FIDELITY_DEGENERATE:
        status = STATUS_DEGRADED
        notes.append(
            "Degenerate baseline: does not represent the published method."
        )

    return {"status": status, "notes": notes}


def comparison_axes(
    problem_name: str,
    objective_mode: str = OBJECTIVE_RELOCATIONS,
    time_model: Optional[str] = None,
    num_bays: int = 1,
    num_rows: int = 1,
    instance_source: Optional[str] = None,
) -> Dict[str, Any]:
    """
    The four axes that must match for two runs to be comparable.

    Kept as one helper so the GUI, the result store and the docs cannot drift
    apart on what "same setting" means.
    """
    return {
        "problem": problem_name,
        "objective": (
            objective_mode
            if objective_mode not in _TIME_AWARE_MODES or not time_model
            else f"{objective_mode}/{time_model}"
        ),
        "geometry": yard_geometry(num_bays, num_rows),
        "instances": instance_source or "random",
    }


def validate_meta(cls: type) -> List[str]:
    """Return a list of problems with the metadata block of *cls* (empty = fine)."""
    meta = algorithm_meta(cls)
    problems: List[str] = []

    if meta["geometry"] not in GEOMETRIES:
        problems.append(
            f"geometry {meta['geometry']!r} not in {list(GEOMETRIES)}"
        )
    if meta["fidelity"] not in FIDELITIES:
        problems.append(
            f"fidelity {meta['fidelity']!r} not in {list(FIDELITIES)}"
        )

    objectives: Sequence[str] = meta["objectives"]
    if not objectives:
        problems.append("objectives must not be empty")
    for objective in objectives:
        if objective not in OBJECTIVE_VALUES:
            problems.append(
                f"objective {objective!r} not in {list(OBJECTIVE_VALUES)}"
            )
    if OBJECTIVE_FIXED_RULE in objectives and len(objectives) > 1:
        problems.append(
            "fixed-rule cannot be combined with other objectives"
        )

    return problems

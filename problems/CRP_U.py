"""CRP-U – unrestricted (free-order) container relocation."""

from __future__ import annotations

from problems.brp_nonfixed import BRPNonFixed


class CRP_U(BRPNonFixed):
    """Unrestricted retrieval order; same environment as BRP-NonFixed."""

    name = "CRP-U"
    description = (
        "Unrestricted CRP: free choice of retrieval order; minimise total relocations. "
        "Same dynamics as BRP-NonFixed."
    )
    tags = ["crp", "unrestricted", "non-fixed-order", "yard-only"]
    metric_names = ["relocations", "steps", "time"]
    hide_from_problem_list = False

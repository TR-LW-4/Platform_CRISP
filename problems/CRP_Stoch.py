"""CRP-Stoch – stochastic CRP variant (extends CRP-Time until dedicated dynamics)."""

from __future__ import annotations

from problems.CRP_Time import CRP_Time


class CRP_Stoch(CRP_Time):
    """Same kinematics as CRP-Time; extend reset/extra for stochasticity as needed."""

    name = "CRP-Stoch"
    description = (
        "Stochastic container retrieval (CRP). "
        "Currently uses CRP-Time dynamics; plug in stochastic arrivals/uncertainty here."
    )
    tags = ["crp", "stochastic", "time", "yard-only", "fixed-order"]
    metric_names = ["time", "crane_time", "relocations", "steps"]

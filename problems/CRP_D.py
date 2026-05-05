"""CRP-D – duplicate-group / duplicate-container stowage (placeholder on CRP-Stow)."""

from __future__ import annotations

from problems.CRP_Stow import CRP_Stow


class CRP_D(CRP_Stow):
    """Stowage with duplicate groups; extend episode build for duplicate containers."""

    name = "CRP-D"
    description = (
        "Duplicate-group CRP (stowage). "
        "Currently uses CRP-Stow dynamics; specialise instance generation for duplicates."
    )
    tags = ["crp", "duplicate", "stowage", "vessel", "group-matching"]
    metric_names = ["shifters", "time", "vessel_utilisation"]

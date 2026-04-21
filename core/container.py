"""
Container data model with extensible attributes.
All problems in the platform use Container as the fundamental unit.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Optional


class ContainerSize(IntEnum):
    TEU = 1    # 20-foot standard
    FEU = 2    # 40-foot standard
    CUSTOM = 3


class ContainerType(IntEnum):
    STANDARD  = 0
    REEFER    = 1   # refrigerated – requires power plug slot
    HAZMAT    = 2   # dangerous goods – segregation rules apply
    OVERSIZE  = 3   # exceeds standard envelope


class WeightClass(IntEnum):
    LIGHT  = 0   # < 10 t
    MEDIUM = 1   # 10 – 20 t
    HEAVY  = 2   # > 20 t

    @classmethod
    def from_tons(cls, tons: float) -> "WeightClass":
        if tons < 10:
            return cls.LIGHT
        if tons < 20:
            return cls.MEDIUM
        return cls.HEAVY


@dataclass
class Container:
    """
    Represents one physical container.

    Core fields (always present):
        id        – unique integer ID within an episode
        group     – destination-port group (0-indexed)
        priority  – retrieval order for BRP-Fixed (1 = retrieve first)
        weight    – gross weight in metric tons
        size      – TEU / FEU / CUSTOM
        ctype     – STANDARD / REEFER / HAZMAT / OVERSIZE

    Extensible:
        attrs     – arbitrary key-value store for future attributes
                    e.g. attrs["temperature_setpoint"] = -18  (reefer)
                         attrs["imdg_class"] = "3"            (hazmat)
                         attrs["custom_length_ft"] = 45       (custom)
    """
    id:       int
    group:    int            = 0
    priority: int            = 0
    weight:   float          = 10.0
    size:     ContainerSize  = ContainerSize.TEU
    ctype:    ContainerType  = ContainerType.STANDARD
    attrs:    Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Convenience helpers                                                   #
    # ------------------------------------------------------------------ #

    @property
    def weight_class(self) -> WeightClass:
        return WeightClass.from_tons(self.weight)

    @property
    def slot_count(self) -> int:
        """Number of TEU-slots occupied (1 for TEU, 2 for FEU, variable for CUSTOM)."""
        if self.size == ContainerSize.FEU:
            return 2
        if self.size == ContainerSize.CUSTOM:
            return int(self.attrs.get("slot_count", 1))
        return 1

    @property
    def is_reefer(self) -> bool:
        return self.ctype == ContainerType.REEFER

    @property
    def is_hazmat(self) -> bool:
        return self.ctype == ContainerType.HAZMAT

    def __repr__(self) -> str:
        return (
            f"C(id={self.id}, grp={self.group}, pri={self.priority}, "
            f"w={self.weight:.1f}t, {self.size.name}/{self.ctype.name})"
        )

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Container):
            return self.id == other.id
        return NotImplemented


# ------------------------------------------------------------------ #
# Factory helpers                                                       #
# ------------------------------------------------------------------ #

def make_containers(
    n: int,
    num_groups: int = 1,
    seed: int = 0,
    enable_weight: bool = False,
    enable_size: bool = False,
    enable_type: bool = False,
    group_assignment: str = "random",   # "random" | "sequential"
    priority_assignment: str = "sequential",  # "sequential" | "random"
) -> list[Container]:
    """
    Create *n* Container objects with configurable attributes.

    Parameters
    ----------
    n                   : total number of containers
    num_groups          : number of destination-port groups
    seed                : random seed
    enable_weight       : assign realistic weight values
    enable_size         : assign TEU/FEU mix
    enable_type         : assign container types (reefer/hazmat)
    group_assignment    : how to assign groups
    priority_assignment : how to assign retrieval priorities
    """
    import numpy as np
    rng = np.random.RandomState(seed)

    containers = []
    for i in range(n):
        # Group
        if group_assignment == "sequential":
            group = i % num_groups
        else:
            group = int(rng.randint(0, num_groups))

        # Priority (for BRP-Fixed): 1..n
        if priority_assignment == "sequential":
            priority = i + 1
        else:
            priority = int(rng.permutation(n)[i]) + 1

        # Weight
        weight = float(rng.uniform(5, 30)) if enable_weight else 10.0

        # Size
        if enable_size:
            size = ContainerSize.FEU if rng.rand() < 0.3 else ContainerSize.TEU
        else:
            size = ContainerSize.TEU

        # Type
        if enable_type:
            r = rng.rand()
            if r < 0.05:
                ctype = ContainerType.HAZMAT
            elif r < 0.15:
                ctype = ContainerType.REEFER
            else:
                ctype = ContainerType.STANDARD
        else:
            ctype = ContainerType.STANDARD

        containers.append(Container(
            id=i,
            group=group,
            priority=priority,
            weight=weight,
            size=size,
            ctype=ctype,
        ))

    return containers

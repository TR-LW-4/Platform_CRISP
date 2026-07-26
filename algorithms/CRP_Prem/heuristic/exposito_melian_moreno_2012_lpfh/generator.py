from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

Stacks = Dict[int, List[int]]


@dataclass
class GeneratorConfig:
    stacks: int
    tiers: int
    occupancy: float
    priorities: Sequence[int]
    priority_share: Dict[int, float]
    groups: List[List[int]]
    levels: List[Tuple[int, int]]
    group_level_share: List[List[float]]
    seed: int = 0


def generate_instance(cfg: GeneratorConfig) -> Stacks:
    """
    Paper-inspired constructive instance generator.
    Returns bay layout as stack->priorities (bottom..top).
    """
    rng = random.Random(cfg.seed)
    n_slots = cfg.stacks * cfg.tiers
    n_cont = int(round(max(0.0, min(1.0, cfg.occupancy)) * n_slots))

    # Priority counts.
    p_counts: Dict[int, int] = {}
    rem = n_cont
    prios = list(cfg.priorities)
    for i, p in enumerate(prios):
        if i == len(prios) - 1:
            c = rem
        else:
            c = int(round(cfg.priority_share.get(p, 0.0) * n_cont))
            c = max(0, min(rem, c))
        p_counts[p] = c
        rem -= c
    if rem > 0:
        p_counts[prios[-1]] += rem

    # Containers to place per group-level pair.
    gl_counts: List[List[int]] = []
    for gi, g in enumerate(cfg.groups):
        g_total = sum(p_counts.get(p, 0) for p in g)
        shares = cfg.group_level_share[gi]
        row = []
        rem_g = g_total
        for lj, sh in enumerate(shares):
            if lj == len(shares) - 1:
                c = rem_g
            else:
                c = int(round(sh * g_total))
                c = max(0, min(rem_g, c))
            row.append(c)
            rem_g -= c
        gl_counts.append(row)

    # Build stack storage with empty slots represented by None.
    slots: Dict[int, List[int]] = {s: [] for s in range(cfg.stacks)}

    for lj, (l0, l1) in enumerate(cfg.levels):
        level_capacity = l1 - l0 + 1
        level_pool: List[int] = []
        for gi, g in enumerate(cfg.groups):
            need = gl_counts[gi][lj]
            candidates = [p for p in g for _ in range(p_counts.get(p, 0))]
            rng.shuffle(candidates)
            take = candidates[:need]
            for p in take:
                p_counts[p] -= 1
            level_pool.extend(take)
        rng.shuffle(level_pool)

        for p in level_pool:
            feasible = [s for s in range(cfg.stacks) if len(slots[s]) < (l1 + 1)]
            if not feasible:
                feasible = [s for s in range(cfg.stacks) if len(slots[s]) < cfg.tiers]
            if not feasible:
                break
            s = rng.choice(feasible)
            slots[s].append(p)

    return slots

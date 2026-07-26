"""
Lower-bound utilities ported from:

- BlockingLowerBound.m
- RollingLowerBound.m
- boundsDifference.m
"""

from __future__ import annotations

import math

import numpy as np


def blocking_lower_bound(bay: np.ndarray) -> float:
    """Expected blocking lower bound from `BlockingLowerBound.m`."""
    tiers, stacks = bay.shape
    del tiers
    lower_bound = 0.0
    height = np.sum(bay != 0, axis=0)
    for s in range(stacks):
        if int(height[s]) > 1:
            for t in range(2, int(height[s]) + 1):
                block = int(bay[bay.shape[0] - t, s])
                sure_blocking = False
                potential_blocking = 0
                for u in range(1, t):
                    below = int(bay[bay.shape[0] - u, s])
                    if block > below:
                        sure_blocking = True
                    elif block == below:
                        potential_blocking += 1
                if sure_blocking:
                    lower_bound += 1.0
                elif potential_blocking > 0:
                    lower_bound += potential_blocking / (potential_blocking + 1.0)
    return float(lower_bound)


def rolling_lower_bound(bay: np.ndarray, n_rolling: int) -> float:
    """Port of `RollingLowerBound.m`."""
    if n_rolling == 0:
        return 0.0
    tiers, stacks = bay.shape
    lower_bound = 0.0
    height = np.sum(bay != 0, axis=0)
    maxmin_vector = np.zeros(stacks, dtype=int)
    for s in range(stacks):
        for st in range(stacks):
            if st == s:
                continue
            if int(height[st]) == 0:
                maxmin_vector[s] = tiers * stacks
            else:
                sta = bay[:, st]
                nz = sta[sta > 0]
                if len(nz):
                    maxmin_vector[s] = max(maxmin_vector[s], int(np.min(nz)))
    if int(np.max(maxmin_vector)) == tiers * stacks:
        return 0.0

    current_min = int(np.min(bay[bay != 0]))
    target_tiers, target_stacks = np.where(bay == current_min)
    for tier_idx, stack_idx in zip(target_tiers.tolist(), target_stacks.tolist()):
        if int(height[stack_idx]) > bay.shape[0] - int(tier_idx):
            for t in range(bay.shape[0] - int(height[stack_idx]), int(tier_idx)):
                if int(bay[t, stack_idx]) > int(maxmin_vector[stack_idx]):
                    lower_bound += 1.0 / len(target_tiers)
        new_bay = bay.copy()
        new_bay[: int(tier_idx) + 1, stack_idx] = 0
        lower_bound += (1.0 / len(target_tiers)) * rolling_lower_bound(
            new_bay, n_rolling - 1
        )
    return float(lower_bound)


def bounds_difference(bay: np.ndarray) -> float:
    """Port of `boundsDifference.m` used by PBFSA sampling."""
    tiers, stacks = bay.shape
    min_lower_bound = 0.0
    max_lower_bound = 0.0
    for s in range(stacks):
        for t in range(2, tiers + 1):
            block = int(bay[bay.shape[0] - t, s])
            if block == 0:
                continue
            is_blocking_min = False
            is_blocking_max = False
            for tloc in range(1, t):
                below = int(bay[bay.shape[0] - tloc, s])
                if block > below:
                    is_blocking_min = True
                if block >= below:
                    is_blocking_max = True
                if is_blocking_min and is_blocking_max:
                    break
            if is_blocking_min:
                min_lower_bound += 1.0
                max_lower_bound += 1.0
            elif is_blocking_max:
                max_lower_bound += 1.0
    n_containers = int(np.sum(bay > 0))
    max_lower_bound = (2 * math.ceil(n_containers / stacks) - 1) * max_lower_bound
    max_lower_bound = min(max_lower_bound, (n_containers - stacks) * (tiers - 1) + stacks)
    return float(max_lower_bound - min_lower_bound)

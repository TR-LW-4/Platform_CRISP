"""
Heuristic policies ported from:

- heuristic.m
- heuristic_Online.m
- retrieveEG.m
- retrieveEM.m
- retrieveERI.m
- retrieveL.m
- retrieveRand.m
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

from .bounds import blocking_lower_bound
from .bay_utils import unveil_containers, unveil_containers_online


HEURISTIC_IDS: Dict[str, int] = {
    "EG": 1,
    "EM": 2,
    "ERI": 3,
    "L": 4,
    "Rand": 5,
}


def _height(bay: np.ndarray) -> np.ndarray:
    return np.sum(bay != 0, axis=0).astype(int)


def retrieve_l(bay: np.ndarray, t_retrieve: int, s_retrieve: int) -> tuple[np.ndarray, int]:
    bay = bay.copy()
    t_size = bay.shape[0]
    stack = int(s_retrieve)
    tier = t_size - int(t_retrieve)
    height = _height(bay)
    n_reloc = int(height[stack] - tier)
    while tier < int(height[stack]):
        height_loc = height.copy()
        height_loc[stack] = t_size + 1
        target_stack = int(np.where(height_loc == np.min(height_loc))[0][0])
        target_tier = int(height[target_stack] + 1)
        bay[t_size - target_tier, target_stack] = bay[t_size - int(height[stack]), stack]
        height[target_stack] += 1
        bay[t_size - int(height[stack]), stack] = 0
        height[stack] -= 1
    bay[t_size - tier, stack] = 0
    return bay, n_reloc


def retrieve_rand(
    bay: np.ndarray,
    t_retrieve: int,
    s_retrieve: int,
    rng: Optional[np.random.RandomState] = None,
) -> tuple[np.ndarray, int]:
    bay = bay.copy()
    rng = rng if rng is not None else np.random.RandomState()
    t_size, s_size = bay.shape
    stack = int(s_retrieve)
    tier = t_size - int(t_retrieve)
    height = _height(bay)
    n_reloc = int(height[stack] - tier)
    while tier < int(height[stack]):
        height_loc = height.copy()
        height_loc[stack] = t_size + 1
        candidates = [s for s in range(s_size) if int(height_loc[s]) < t_size]
        target_stack = int(candidates[int(rng.randint(0, len(candidates)))])
        target_tier = int(height[target_stack] + 1)
        bay[t_size - target_tier, target_stack] = bay[t_size - int(height[stack]), stack]
        height[target_stack] += 1
        bay[t_size - int(height[stack]), stack] = 0
        height[stack] -= 1
    bay[t_size - tier, stack] = 0
    return bay, n_reloc


def retrieve_eri(bay: np.ndarray, t_retrieve: int, s_retrieve: int) -> tuple[np.ndarray, int]:
    bay = bay.copy()
    t_size, s_size = bay.shape
    stack = int(s_retrieve)
    tier = t_size - int(t_retrieve)
    height = _height(bay)
    n_reloc = int(height[stack] - tier)
    while tier < int(height[stack]):
        blocking_container = int(bay[t_size - int(height[stack]), stack])
        best_eri = float("inf")
        target_stack = None
        for s in range(s_size):
            if s != stack and int(height[s]) < t_size:
                eri = 0.0
                for t in range(1, int(height[s]) + 1):
                    val = int(bay[t_size - t, s])
                    if val < blocking_container:
                        eri += 1.0
                    elif val == blocking_container:
                        eri += 0.5
                if eri < best_eri or (
                    abs(eri - best_eri) < 1e-12
                    and target_stack is not None
                    and int(height[target_stack]) < int(height[s])
                ):
                    target_stack = s
                    best_eri = eri
                elif eri < best_eri and target_stack is None:
                    target_stack = s
                    best_eri = eri
        assert target_stack is not None
        target_tier = int(height[target_stack] + 1)
        bay[t_size - target_tier, target_stack] = blocking_container
        height[target_stack] += 1
        bay[t_size - int(height[stack]), stack] = 0
        height[stack] -= 1
    bay[t_size - tier, stack] = 0
    return bay, n_reloc


def retrieve_em(bay: np.ndarray, t_retrieve: int, s_retrieve: int) -> tuple[np.ndarray, int]:
    bay = bay.copy()
    t_size, s_size = bay.shape
    z_max = int(np.max(bay[bay != 0])) if np.any(bay != 0) else 0
    stack = int(s_retrieve)
    tier = t_size - int(t_retrieve)
    height = _height(bay)
    n_reloc = int(height[stack] - tier)
    min_vector = np.zeros(s_size, dtype=int)
    for s in range(s_size):
        if s == stack:
            continue
        stack_considered = bay[:, s]
        nz = stack_considered[stack_considered != 0]
        if len(nz) == 0:
            min_vector[s] = z_max + 1
        elif int(height[s]) == t_size:
            min_vector[s] = 0
        else:
            min_vector[s] = int(np.min(nz))
    while tier < int(height[stack]):
        blocking_container = int(bay[t_size - int(height[stack]), stack])
        target_stack = None
        if int(np.max(min_vector)) > blocking_container:
            target_min = int(np.min(min_vector[min_vector > blocking_container]))
            target_height = -1
            for s in range(s_size):
                if s != stack and int(height[s]) < t_size:
                    if int(min_vector[s]) == target_min and int(height[s]) > target_height:
                        target_height = int(height[s])
                        target_stack = s
        else:
            target_min = int(np.max(min_vector))
            target_height = -1
            n_max_min = t_size
            for s in range(s_size):
                if s != stack and int(height[s]) < t_size:
                    stack_considered = bay[:, s]
                    n_min = (
                        int(np.sum(stack_considered == target_min)) / max(int(np.max(min_vector)), 1)
                    )
                    if int(min_vector[s]) == target_min and (
                        n_min < n_max_min or (n_min == n_max_min and int(height[s]) > target_height)
                    ):
                        n_max_min = n_min
                        target_stack = s
                        target_height = int(height[s])
        assert target_stack is not None
        target_tier = int(height[target_stack] + 1)
        bay[t_size - target_tier, target_stack] = bay[t_size - int(height[stack]), stack]
        if int(min_vector[target_stack]) != 0:
            if target_tier != t_size:
                min_vector[target_stack] = min(int(min_vector[target_stack]), blocking_container)
            else:
                min_vector[target_stack] = 0
        else:
            min_vector[target_stack] = blocking_container
        height[target_stack] += 1
        bay[t_size - int(height[stack]), stack] = 0
        height[stack] -= 1
    bay[t_size - tier, stack] = 0
    return bay, n_reloc


def retrieve_eg(bay: np.ndarray, t_retrieve: int, s_retrieve: int) -> tuple[np.ndarray, int]:
    bay = bay.copy()
    t_size, s_size = bay.shape
    stack = int(s_retrieve)
    tier = t_size - int(t_retrieve)
    height = _height(bay)
    virtual_min_vector = np.zeros(s_size, dtype=int)
    n_reloc = int(height[stack] - tier)
    for s in range(s_size):
        if s == stack:
            continue
        stack_considered = bay[:, s]
        nz = stack_considered[stack_considered != 0]
        if len(nz) == 0:
            virtual_min_vector[s] = int(np.max(bay[bay != 0])) + 1
        elif int(height[s]) == t_size:
            virtual_min_vector[s] = 0
        else:
            virtual_min_vector[s] = int(np.min(nz))

    if n_reloc > 0:
        blocking_containers = bay[t_size - int(height[stack]) : int(t_retrieve), stack].astype(int)
        positions: Dict[int, list[int]] = {}
        for idx, val in enumerate(blocking_containers.tolist(), start=1):
            positions.setdefault(int(val), []).append(idx)
        index_relocated = np.zeros((t_size, s_size), dtype=int)
        add_height = np.zeros(s_size, dtype=int)

        first_phase_blocking = sorted(blocking_containers.tolist(), reverse=True)
        first_phase_relocated = np.zeros(len(first_phase_blocking), dtype=int)
        for c_idx, block in enumerate(first_phase_blocking):
            if block < int(np.max(virtual_min_vector)):
                target_stack = None
                target_min = float("inf")
                for s in range(s_size):
                    if (
                        int(virtual_min_vector[s]) > block
                        and int(virtual_min_vector[s]) < target_min
                        and int(np.max(index_relocated[:, s])) < positions[block][0]
                    ):
                        target_min = int(virtual_min_vector[s])
                        target_stack = s
                if target_stack is not None:
                    first_phase_relocated[c_idx] = 1
                    add_height[target_stack] += 1
                    index_relocated[add_height[target_stack] - 1, target_stack] = positions[block][0]
                    positions[block] = positions[block][1:]
                    if int(height[target_stack] + add_height[target_stack]) == t_size:
                        virtual_min_vector[target_stack] = 0
                    else:
                        virtual_min_vector[target_stack] = block

        for s in range(s_size):
            if s != stack:
                if int(add_height[s]) > 1 or int(height[s] + add_height[s]) == t_size:
                    virtual_min_vector[s] = 0

        if int(np.sum(add_height)) < len(first_phase_blocking):
            second_phase_blocking = [
                first_phase_blocking[i]
                for i in range(len(first_phase_blocking))
                if first_phase_relocated[i] == 0
            ]
            second_phase_blocking = sorted(second_phase_blocking)
            for block in second_phase_blocking:
                diff_min = virtual_min_vector.copy()
                target_stack = None
                for s in range(s_size):
                    if s == stack or int(height[s] + add_height[s]) == t_size:
                        diff_min[s] = -(t_size * s_size)
                    elif int(add_height[s]) == 1:
                        diff_min[s] = block - int(virtual_min_vector[s])
                    else:
                        diff_min[s] = int(virtual_min_vector[s]) - block
                if int(np.max(diff_min)) > 0:
                    target_min = int(np.min(diff_min[diff_min > 0]))
                    target_height = -1
                    for s in range(s_size):
                        if s != stack and int(height[s] + add_height[s]) < t_size:
                            if int(diff_min[s]) == target_min and int(height[s] + add_height[s]) > target_height:
                                target_height = int(height[s] + add_height[s])
                                target_stack = s
                else:
                    target_min = int(np.max(diff_min))
                    target_height = -1
                    n_max_min = t_size
                    for s in range(s_size):
                        if s == stack or int(height[s] + add_height[s]) >= t_size:
                            continue
                        stack_considered = bay[:, s]
                        nz = stack_considered[stack_considered > 0]
                        n_min = 0
                        if len(nz) > 0:
                            n_min = int(np.sum(stack_considered == int(np.min(nz))))
                        relocated_vals = [
                            int(blocking_containers[pos - 1]) for pos in index_relocated[:, s].tolist() if pos > 0
                        ]
                        if relocated_vals:
                            relocated_min = min(relocated_vals)
                            if n_min == 0 or relocated_min < int(np.min(nz)) if len(nz) else True:
                                n_min = sum(1 for v in relocated_vals if v == relocated_min)
                        if int(diff_min[s]) == target_min and (
                            n_min < n_max_min or (n_min == n_max_min and int(height[s] + add_height[s]) > target_height)
                        ):
                            n_max_min = n_min
                            target_stack = s
                            target_height = int(height[s] + add_height[s])
                if target_stack is None:
                    raise RuntimeError("EG target stack not found")
                add_height[target_stack] += 1
                index_relocated[add_height[target_stack] - 1, target_stack] = positions[block][0]
                positions[block] = positions[block][1:]
                if int(height[target_stack] + add_height[target_stack]) == t_size:
                    virtual_min_vector[target_stack] = 0

        for s in range(s_size):
            new_containers = sorted([int(x) for x in index_relocated[:, s].tolist() if x > 0], reverse=True)
            if new_containers:
                for idx in reversed(new_containers):
                    height[s] += 1
                    bay[t_size - int(height[s]), s] = int(blocking_containers[idx - 1])
    bay[t_size - int(height[stack]) : int(t_retrieve) + 1, stack] = 0
    return bay, n_reloc


def _apply_heuristic(
    bay: np.ndarray,
    heuristic_number: int,
    t_retrieve: int,
    s_retrieve: int,
    rng: Optional[np.random.RandomState] = None,
) -> tuple[np.ndarray, int]:
    if heuristic_number == 1:
        return retrieve_eg(bay, t_retrieve, s_retrieve)
    if heuristic_number == 2:
        return retrieve_em(bay, t_retrieve, s_retrieve)
    if heuristic_number == 3:
        return retrieve_eri(bay, t_retrieve, s_retrieve)
    if heuristic_number == 4:
        return retrieve_l(bay, t_retrieve, s_retrieve)
    if heuristic_number == 5:
        return retrieve_rand(bay, t_retrieve, s_retrieve, rng=rng)
    raise ValueError(f"Unknown heuristic_number={heuristic_number}")


def run_heuristic_batch(
    bay: np.ndarray,
    heuristic_number: int,
    n_samples: int,
    rng: Optional[np.random.RandomState] = None,
) -> float:
    """Port of `heuristic.m`."""
    rng = rng if rng is not None else np.random.RandomState()
    obj = 0.0
    n_containers = int(np.sum(bay != 0))
    for _ in range(n_samples):
        temp_bay = bay.copy()
        total_reloc = 0
        n_left = n_containers
        while n_left > bay.shape[1] or (heuristic_number == 5 and n_left > 0):
            current_min = int(np.min(temp_bay[temp_bay != 0]))
            target_tiers, target_stacks = np.where(temp_bay == current_min)
            if len(target_tiers) > 1:
                temp_bay = unveil_containers(
                    temp_bay, target_tiers, target_stacks, current_min, rng=rng
                )
                next_tiers, next_stacks = np.where(temp_bay == int(np.min(temp_bay[temp_bay != 0])))
                t_retrieve, s_retrieve = int(next_tiers[0]), int(next_stacks[0])
            else:
                t_retrieve, s_retrieve = int(target_tiers[0]), int(target_stacks[0])
            temp_bay, n_reloc = _apply_heuristic(
                temp_bay, heuristic_number, t_retrieve, s_retrieve, rng=rng
            )
            total_reloc += n_reloc
            n_left -= 1
        if heuristic_number != 5 and np.any(temp_bay != 0):
            total_reloc += blocking_lower_bound(temp_bay)
        obj += total_reloc / n_samples
    return float(obj)


def run_heuristic_online(
    bay: np.ndarray,
    heuristic_number: int,
    n_samples: int,
    rng: Optional[np.random.RandomState] = None,
) -> float:
    """Port of `heuristic_Online.m`."""
    rng = rng if rng is not None else np.random.RandomState()
    obj = 0.0
    n_containers = int(np.sum(bay != 0))
    for _ in range(n_samples):
        temp_bay = bay.copy()
        total_reloc = 0
        n_left = n_containers
        while n_left > bay.shape[1] or (heuristic_number == 5 and n_left > 0):
            current_min = int(np.min(temp_bay[temp_bay != 0]))
            target_tiers, target_stacks = np.where(temp_bay == current_min)
            if len(target_tiers) > 1:
                temp_bay = unveil_containers_online(temp_bay, target_tiers, target_stacks, rng=rng)
                next_tiers, next_stacks = np.where(temp_bay == int(np.min(temp_bay[temp_bay != 0])))
                t_retrieve, s_retrieve = int(next_tiers[0]), int(next_stacks[0])
            else:
                t_retrieve, s_retrieve = int(target_tiers[0]), int(target_stacks[0])
            temp_bay, n_reloc = _apply_heuristic(
                temp_bay, heuristic_number, t_retrieve, s_retrieve, rng=rng
            )
            total_reloc += n_reloc
            n_left -= 1
        if heuristic_number != 5 and np.any(temp_bay != 0):
            total_reloc += blocking_lower_bound(temp_bay)
        obj += total_reloc / n_samples
    return float(obj)

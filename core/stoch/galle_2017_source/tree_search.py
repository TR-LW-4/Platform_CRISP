"""
Tree-search algorithms ported from:

- PBFSA.m
- PBFSA_Rec.m
- PBFSA_Decision.m
- PBFSA_Chance.m
- PBFSA_AllDec.m
- PBFSA_AllCha.m
- PBFS_Online.m
- PBFS_Rec_Online.m
- PBFS_Decision_Online.m
- PBFS_Chance_Online.m
- PBFS_AllDec_Online.m
- PBFS_AllCha_Online.m

Implementation note
-------------------
The original MATLAB code stores a mutable explicit tree and per-level node
tables.  The Python port keeps the same algorithmic behavior but uses a
memoized recursive solver keyed by `(type, level, abstracted_bay)`, which
is simpler and still preserves the abstraction / duplicate-merging effect.
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .astar import astar
from .bay_utils import abstract_bay
from .bounds import blocking_lower_bound, bounds_difference, rolling_lower_bound


@dataclass
class SearchStats:
    nodes_expanded: int = 0
    cache_hits: int = 0


def _n_positive_labels(bay: np.ndarray) -> int:
    vals = {int(v) for v in bay.flatten().tolist() if int(v) > 0}
    return len(vals)


def _n_nonzero_labels(bay: np.ndarray) -> int:
    vals = {int(v) for v in bay.flatten().tolist() if int(v) != 0}
    return len(vals)


def _counts_by_positive_label(bay: np.ndarray) -> List[int]:
    vals = sorted({int(v) for v in bay.flatten().tolist() if int(v) > 0})
    return [int(np.sum(bay == v)) for v in vals]


def _target_position_batch(bay: np.ndarray) -> Tuple[int, int]:
    current_min = int(np.min(bay[bay > 0]))
    tiers, stacks = np.where(bay == current_min)
    if len(tiers) != 1:
        raise ValueError("Batch decision node expects a unique revealed target")
    return int(tiers[0]), int(stacks[0])


def _target_position_online(bay: np.ndarray) -> Tuple[int, int]:
    val = int(np.min(bay[bay != 0]))
    tiers, stacks = np.where(bay == val)
    if len(tiers) != 1:
        raise ValueError("Online decision node expects a unique revealed target")
    return int(tiers[0]), int(stacks[0])


def _enumerate_relocation_outcomes(
    bay: np.ndarray,
    target_tier: int,
    target_stack: int,
) -> List[np.ndarray]:
    """
    Enumerate all end configurations obtained by relocating every blocker
    above the target to any non-full non-source stack, then retrieving the
    target.  The returned configurations are *not* abstracted.
    """
    t_size, s_size = bay.shape
    height = np.sum(bay != 0, axis=0).astype(int)
    target_level = t_size - target_tier
    n_reloc = int(height[target_stack] - target_level)
    out: List[np.ndarray] = []

    def _dfs(cur_bay: np.ndarray, cur_height: np.ndarray, remaining: int) -> None:
        if remaining == 0:
            leaf = cur_bay.copy()
            leaf[target_tier, target_stack] = 0
            out.append(leaf)
            return
        src_row = t_size - int(cur_height[target_stack])
        moving = int(cur_bay[src_row, target_stack])
        for s in range(s_size):
            if s == target_stack or int(cur_height[s]) >= t_size:
                continue
            next_bay = cur_bay.copy()
            next_height = cur_height.copy()
            dst_row = t_size - int(next_height[s]) - 1
            next_bay[dst_row, s] = moving
            next_height[s] += 1
            next_bay[src_row, target_stack] = 0
            next_height[target_stack] -= 1
            _dfs(next_bay, next_height, remaining - 1)

    _dfs(bay.copy(), height.copy(), n_reloc)
    return out


def _pbfsa_all_dec(bay: np.ndarray) -> Tuple[List[np.ndarray], int]:
    target_tier, target_stack = _target_position_batch(bay)
    t_size = bay.shape[0]
    height = np.sum(bay != 0, axis=0).astype(int)
    target_level = t_size - target_tier
    n_reloc = int(height[target_stack] - target_level)
    leaves = _enumerate_relocation_outcomes(bay, target_tier, target_stack)
    unique: Dict[bytes, np.ndarray] = {}
    for leaf in leaves:
        proj = abstract_bay(leaf)
        unique.setdefault(proj.tobytes(), proj)
    return list(unique.values()), n_reloc


def _pbfs_all_dec_online(bay: np.ndarray) -> List[Tuple[np.ndarray, int]]:
    target_tier, target_stack = _target_position_online(bay)
    leaves = _enumerate_relocation_outcomes(bay, target_tier, target_stack)
    unique: Dict[bytes, Tuple[np.ndarray, int]] = {}
    for leaf in leaves:
        reloc = 0  # filled below
        min_val = int(np.min(leaf[leaf != 0])) if np.any(leaf != 0) else 0
        if min_val < 0:
            leaf = leaf.copy()
            for t in range(leaf.shape[0]):
                for s in range(leaf.shape[1]):
                    if leaf[t, s] != 0:
                        leaf[t, s] = int(leaf[t, s] - min_val + 1)
        proj = abstract_bay(leaf)
        key = proj.tobytes()
        # Recompute reloc as number of blockers above the online target in parent.
        # We compare by counting differences from the original source stack.
        # The source implementation stores this during generation; here we can
        # derive it from the parent configuration once.
        reloc = int(np.sum(bay[:, target_stack] != 0) - (bay.shape[0] - target_tier))
        prev = unique.get(key)
        if prev is None or reloc < prev[1]:
            unique[key] = (proj, reloc)
    return list(unique.values())


def _pbfsa_all_cha(
    bay: np.ndarray,
    error_gap: float,
    rng: np.random.RandomState,
) -> List[Tuple[np.ndarray, float]]:
    t_size, s_size = bay.shape
    del t_size, s_size
    min_time_window = int(np.min(bay[bay > 0]))
    target_tiers, target_stacks = np.where(bay == min_time_window)
    n_targets = len(target_tiers)
    if error_gap > 0:
        n_samples = int(math.ceil((math.pi / 2.0) * (bounds_difference(bay) / error_gap) ** 2))
    else:
        n_samples = math.inf
    if n_samples <= math.factorial(n_targets):
        perms = [tuple((rng.permutation(n_targets) + 1).tolist()) for _ in range(int(n_samples))]
        total = float(n_samples)
    else:
        perms = list(itertools.permutations(range(1, n_targets + 1)))
        total = float(len(perms))
    out: Dict[bytes, Tuple[np.ndarray, float]] = {}
    for perm in perms:
        loc_bay = bay.copy()
        for target in range(n_targets):
            loc_bay[int(target_tiers[target]), int(target_stacks[target])] = (
                min_time_window + int(perm[target]) - 1
            )
        loc_bay = abstract_bay(loc_bay)
        key = loc_bay.tobytes()
        if key not in out:
            out[key] = (loc_bay, 1.0 / total)
        else:
            out[key] = (loc_bay, out[key][1] + 1.0 / total)
    return list(out.values())


def _pbfs_all_cha_online(bay: np.ndarray) -> List[Tuple[np.ndarray, float]]:
    min_time_window = int(np.min(bay[bay > 0]))
    target_tiers, target_stacks = np.where(bay == min_time_window)
    n_targets = len(target_tiers)
    out: Dict[bytes, Tuple[np.ndarray, float]] = {}
    for idx in range(n_targets):
        loc_bay = bay.copy()
        loc_bay[int(target_tiers[idx]), int(target_stacks[idx])] = -1
        loc_bay = abstract_bay(loc_bay)
        key = loc_bay.tobytes()
        if key not in out:
            out[key] = (loc_bay, 1.0 / n_targets)
        else:
            out[key] = (loc_bay, out[key][1] + 1.0 / n_targets)
    return list(out.values())


def _next_batch_node_type(bay: np.ndarray) -> str:
    current_min = int(np.min(bay[bay > 0]))
    tiers, _stacks = np.where(bay == current_min)
    return "D" if len(tiers) == 1 else "C"


def _decision_successors_batch(
    bay: np.ndarray,
    lower_bound_type: int,
) -> List[Tuple[np.ndarray, float]]:
    subtrees, n_reloc = _pbfsa_all_dec(bay)
    out = []
    for cfg in subtrees:
        lb = n_reloc + blocking_lower_bound(cfg) + rolling_lower_bound(cfg, lower_bound_type)
        out.append((cfg, float(lb)))
    out.sort(key=lambda x: x[1])
    return out


def _decision_successors_online(
    bay: np.ndarray,
    lower_bound_type: int,
) -> List[Tuple[np.ndarray, float, int]]:
    subtrees = _pbfs_all_dec_online(bay)
    out = []
    for cfg, reloc in subtrees:
        if lower_bound_type > 0:
            lb = reloc + blocking_lower_bound(cfg) + rolling_lower_bound(cfg, lower_bound_type)
        else:
            lb = reloc + blocking_lower_bound(cfg)
        out.append((cfg, float(lb), reloc))
    out.sort(key=lambda x: x[1])
    return out


def pbfsa(
    bay: np.ndarray,
    lower_bound_type: int = 1,
    error_gap: float = 0.0,
    time_limit_s: float = 3600.0,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[float, Dict[str, int], float]:
    """
    Port of `PBFSA.m`.

    Setting `error_gap=0` reproduces the exact batch-model solver used as
    the "optimal algorithm" in the original experiments.
    """
    rng = rng if rng is not None else np.random.RandomState()
    bay = abstract_bay(bay.copy())
    start = time.perf_counter()
    stats = SearchStats()
    counts = _counts_by_positive_label(bay)
    c_total = int(np.sum(bay != 0))
    n_time_windows_exp = 0
    n_loc = c_total
    while n_time_windows_exp < len(counts) and n_loc >= bay.shape[1]:
        n_time_windows_exp += 1
        n_loc -= counts[n_time_windows_exp - 1]
    error_gap_loc = float(error_gap) / max(n_time_windows_exp, 1)
    memo: Dict[Tuple[str, int, bytes], float] = {}

    def _rec(cur_bay: np.ndarray, node_type: str, level: int) -> float:
        if time.perf_counter() - start > time_limit_s:
            return float("inf")
        key = (node_type, level, cur_bay.tobytes())
        if key in memo:
            stats.cache_hits += 1
            return memo[key]
        stats.nodes_expanded += 1
        if level <= cur_bay.shape[1]:
            val = blocking_lower_bound(cur_bay)
        elif node_type == "C":
            val = 0.0
            for child_bay, prob in _pbfsa_all_cha(cur_bay, error_gap_loc, rng):
                val += prob * _rec(child_bay, "D", level)
        else:
            if int(np.sum(cur_bay != 0)) == _n_nonzero_labels(cur_bay) - (1 if -1 in cur_bay else 0):
                val = astar(cur_bay, lower_bound_type=lower_bound_type, upper_bound_type=2, rng=rng)
            else:
                successors = _decision_successors_batch(cur_bay, lower_bound_type)
                min_cost = float("inf")
                target_tier, target_stack = _target_position_batch(cur_bay)
                target_level = cur_bay.shape[0] - target_tier
                n_reloc = int(np.sum(cur_bay[:, target_stack] != 0) - target_level)
                for child_bay, lb in successors:
                    if lb < min_cost:
                        child_type = _next_batch_node_type(child_bay)
                        child_val = _rec(child_bay, child_type, level - 1)
                        min_cost = min(min_cost, n_reloc + child_val)
                val = min_cost
        memo[key] = float(val)
        return float(val)

    obj = _rec(bay, "C", c_total)
    elapsed = time.perf_counter() - start
    if elapsed > time_limit_s:
        obj = float("inf")
    return obj, stats.__dict__, elapsed


def pbfs_online(
    bay: np.ndarray,
    lower_bound_type: int = 1,
    time_limit_s: float = 3600.0,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[float, Dict[str, int], float]:
    """Port of `PBFS_Online.m`."""
    rng = rng if rng is not None else np.random.RandomState()
    bay = abstract_bay(bay.copy())
    start = time.perf_counter()
    stats = SearchStats()
    c_total = int(np.sum(bay != 0))
    memo: Dict[Tuple[str, int, bytes], float] = {}

    def _rec(cur_bay: np.ndarray, node_type: str, level: int) -> float:
        if time.perf_counter() - start > time_limit_s:
            return float("inf")
        key = (node_type, level, cur_bay.tobytes())
        if key in memo:
            stats.cache_hits += 1
            return memo[key]
        stats.nodes_expanded += 1
        if level <= cur_bay.shape[1]:
            val = blocking_lower_bound(cur_bay)
        elif node_type == "C":
            val = 0.0
            for child_bay, prob in _pbfs_all_cha_online(cur_bay):
                val += prob * _rec(child_bay, "D", level)
        else:
            successors = _decision_successors_online(cur_bay, lower_bound_type)
            min_cost = float("inf")
            for child_bay, lb, reloc in successors:
                if lb < min_cost:
                    child_val = _rec(child_bay, "C", level - 1)
                    min_cost = min(min_cost, reloc + child_val)
            val = min_cost
        memo[key] = float(val)
        return float(val)

    obj = _rec(bay, "C" if not np.any(bay < 0) else "D", c_total)
    elapsed = time.perf_counter() - start
    if elapsed > time_limit_s:
        obj = float("inf")
    return obj, stats.__dict__, elapsed

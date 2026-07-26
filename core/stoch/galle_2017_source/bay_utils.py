"""
Utilities ported from the following MATLAB files in
`/data/liuw2/StochasticCRP-master`:

- GenerateIncompleteConfig.m
- Pre_Processing.m
- UnvielContainers.m
- UnvielContainers_Online.m
- readInputFile.m

Plus adapter helpers to convert the current `CRP_Stoch` environment into
the source repository's matrix representation.

Matrix convention
-----------------
The source repository represents a bay as a `T x S` integer matrix:

- rows    : tiers from TOP (row 0) to BOTTOM (row T-1)
- columns : stacks from LEFT to RIGHT
- 0       : empty slot
- >0      : batch/order label
- -1      : in the online model, the currently revealed target container
            before its true absolute rank is shifted back to positive.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def abstract_bay(bay: np.ndarray) -> np.ndarray:
    """
    Source-compatible projection / abstraction:
    sort columns lexicographically top-to-bottom.

    This mirrors MATLAB's `sortrows(B')'`.
    """
    cols = [tuple(int(x) for x in bay[:, s]) for s in range(bay.shape[1])]
    cols_sorted = sorted(cols)
    arr = np.array(cols_sorted, dtype=int).T
    return arr.copy()


def batch_start_labels(batches: Sequence[Sequence[int]]) -> Dict[int, int]:
    """
    Return the source-style label for each priority.

    If batches are `[ [1,2], [3,4,5], [6] ]`, then priorities map to
    labels `{1:1, 2:1, 3:3, 4:3, 5:3, 6:6}`.
    """
    out: Dict[int, int] = {}
    start = 1
    for batch in batches:
        for p in batch:
            out[int(p)] = start
        start += len(batch)
    return out


def env_to_source_bay(env, mode: str = "batch") -> np.ndarray:
    """
    Convert a `CRP_Stoch` environment to the source repository matrix.

    Parameters
    ----------
    mode : "batch" or "online"
        In batch mode, containers of the same batch share the same positive
        start label. In online mode we use the same initial encoding; target
        revelation is handled later by `unveil_containers_online()`.
    """
    del mode  # kept for future divergence; current source encoding matches both
    max_tiers = int(env.config.max_tiers)
    stacks = env.yard_snapshot_priorities()
    stack_keys = sorted(stacks.keys())
    labels = batch_start_labels(env.get_batches())
    bay = np.zeros((max_tiers, len(stack_keys)), dtype=int)
    for s_idx, key in enumerate(stack_keys):
        priorities = list(stacks[key])  # bottom -> top
        for offset, pri in enumerate(reversed(priorities), start=1):
            row = max_tiers - len(priorities) + offset - 1
            # Reversed iteration places topmost priority in the first occupied row
            # and bottom-most in the last occupied row.
            bay[row, s_idx] = labels[int(pri)]
    return bay


def read_input_file(
    root: str | Path,
    stacks: int,
    tiers: int,
    instance: int,
    fill_rate: float,
) -> np.ndarray:
    """
    Port of `readInputFile.m`.

    Reads an instance from the Ku & Arthanari / Galle dataset under
    `crptw_instance/`.
    """
    root = Path(root)
    if stacks < 10:
        foldername = root / "crptw_instance" / f"0{stacks}0{tiers}"
        stem = f"0{stacks}0{tiers}"
    else:
        foldername = root / "crptw_instance" / f"{stacks}0{tiers}"
        stem = f"{stacks}0{tiers}"
    prefix = "T271014" if abs(fill_rate - 0.5) < 1e-9 else "T281014"
    filename = foldername / f"{prefix}_{stem}_{instance:03d}.txt"
    text = filename.read_text().strip().splitlines()
    rows = []
    for ln in text[1:]:
        toks = [int(x) for x in ln.split()]
        rows.append(toks)
    raw = np.array(rows, dtype=int)
    height = raw[:, 2]
    bay = np.zeros((tiers, stacks), dtype=int)
    for s in range(stacks):
        for t in range(int(height[s])):
            bay[tiers - t - 1, s] = raw[s, 2 * (t + 2) - 1]
    return bay


def generate_incomplete_config(
    stacks: int,
    tiers: int,
    n_batches: int,
    containers_per_batch: int,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Port of `GenerateIncompleteConfig.m`."""
    rng = rng if rng is not None else np.random.RandomState()
    n_containers = n_batches * containers_per_batch
    perm = rng.permutation(stacks * tiers) + 1
    dense = np.zeros((tiers, stacks), dtype=int)
    for idx, p in enumerate(perm):
        if p <= n_containers:
            dense.flat[idx] = int(p)
    # Apply gravity within each stack.
    gravity = np.zeros((tiers, stacks), dtype=int)
    for s in range(stacks):
        col = dense[:, s]
        non_zero = col[col != 0]
        for k in range(len(non_zero)):
            gravity[tiers - k - 1, s] = int(non_zero[len(non_zero) - k - 1])
    bay = np.zeros((tiers, stacks), dtype=int)
    for s in range(stacks):
        for t in range(tiers):
            if gravity[t, s] > 0:
                bay[t, s] = int(np.ceil(gravity[t, s] / containers_per_batch))
    return bay


def preprocess_bay(bay: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Port of `Pre_Processing.m`.

    Repeatedly retrieves unique visible targets until either the bay has at
    most S containers left or the next target is not unique / not directly
    accessible.
    """
    bay = bay.copy()
    t_size, s_size = bay.shape
    height = np.sum(bay != 0, axis=0)
    n_containers = int(np.sum(height))
    can_retrieve = True
    target_stacks = np.array([], dtype=int)
    while can_retrieve and np.any(bay != 0):
        current_min = int(np.min(bay[bay != 0]))
        tiers, stacks = np.where(bay == current_min)
        if (
            n_containers >= s_size
            and len(tiers) == 1
            and int(height[stacks[0]]) == t_size - int(tiers[0])
        ):
            bay[tiers[0], stacks[0]] = 0
            height[stacks[0]] -= 1
            n_containers -= 1
            target_stacks = stacks
        else:
            can_retrieve = False
            target_stacks = stacks
    return bay, target_stacks


def unveil_containers(
    bay: np.ndarray,
    target_tiers: Sequence[int],
    target_stacks: Sequence[int],
    current_min: int,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Port of `UnvielContainers.m`."""
    rng = rng if rng is not None else np.random.RandomState()
    target_tiers = list(target_tiers)
    target_stacks = list(target_stacks)
    sample_perm = rng.permutation(len(target_tiers))
    new_bay = np.zeros_like(bay)
    for s in range(bay.shape[1]):
        for t in range(bay.shape[0]):
            if bay[t, s] != 0 and bay[t, s] != current_min:
                new_bay[t, s] = bay[t, s] + len(sample_perm)
    for i, idx in enumerate(sample_perm, start=1):
        new_bay[target_tiers[idx], target_stacks[idx]] = i
    return new_bay


def unveil_containers_online(
    bay: np.ndarray,
    target_tiers: Sequence[int],
    target_stacks: Sequence[int],
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Port of `UnvielContainers_Online.m`."""
    rng = rng if rng is not None else np.random.RandomState()
    bay = bay.copy()
    idx = int(rng.randint(0, len(target_tiers)))
    bay[int(target_tiers[idx]), int(target_stacks[idx])] = -1
    return bay

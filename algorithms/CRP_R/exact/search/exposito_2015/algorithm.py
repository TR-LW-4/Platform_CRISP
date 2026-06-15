"""
Expósito-Izquierdo et al. (2015) Branch-and-Bound for Restricted BRP.

Reference
---------
C. Expósito-Izquierdo, B. Melián-Batista, J.M. Moreno-Vega,
"An exact approach for the Blocks Relocation Problem",
Expert Systems with Applications 42 (2015) 6408–6422.

Algorithm overview (Section 5)
--------------------------------
Best-first B&B on a tree of height N+1.
  - Level i  = all blocks with priority ≤ i already retrieved.
  - Each *descending node* represents the yard state reached after
    relocating ALL blockers O(c*) in some assignment and then
    auto-retrieving the target c* plus any further accessible targets.
  - Scoring: f(n) = g(n) + h(n)   (Eq. 17)
      g(n)  = relocations from root to n
      h(n)  = admissible lower bound           (Eq. 18)
  - α-restriction: at most α descending nodes per parent, ranked by
    attractiveness w(n, w)                     (Eq. 19)
    α = 0 (unlimited) guarantees optimality.
  - Initial upper bound: greedy run with α = 1 (Section 5.6).
"""

from __future__ import annotations

import heapq
import itertools
import multiprocessing as mp
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig
from core.yard import Yard, Stack


# ================================================================ #
#  B&B node                                                          #
# ================================================================ #

class _BBNode:
    """Lightweight node in the B&B search tree."""

    __slots__ = ("yard", "g", "priority", "f")

    def __init__(self, yard: Yard, g: int, priority: int, h: int) -> None:
        self.yard     = yard
        self.g        = g
        self.priority = priority  # next priority to retrieve (1-indexed)
        self.f        = g + h

    def __lt__(self, other: "_BBNode") -> bool:  # for heapq tie-breaking
        return self.f < other.f


# ================================================================ #
#  Yard / stack helpers (priority-based, no object-identity reliance)
# ================================================================ #

def _find_stack_by_priority(yard: Yard, priority: int) -> Optional[Stack]:
    """Return the stack containing the container with *priority*, or None."""
    for stack in yard.all_stacks():
        for c in stack.containers:
            if c.priority == priority:
                return stack
    return None


def _stack_max_priority(stack: Stack, N: int) -> int:
    """
    max(s) per Eq. 2: min priority NUMBER in stack (= most-urgent block).
    Returns N+1 for an empty stack.
    """
    if stack.is_empty:
        return N + 1
    return min(c.priority for c in stack.containers)


def _compute_nonlocated_count(yard: Yard) -> int:
    """
    |X|: count of non-located blocks in the yard (Eq. 4).

    Block c is non-located iff ∃ c' in same stack with
      tier(c') < tier(c)  AND  priority(c') < priority(c).
    """
    count = 0
    for stack in yard.all_stacks():
        containers = stack.containers  # bottom→top
        for i in range(1, len(containers)):
            c_pri = containers[i].priority
            for j in range(i):
                if containers[j].priority < c_pri:
                    count += 1
                    break  # c is non-located; count once
    return count


# ================================================================ #
#  Lower bound  h(n)   Eq. 18                                        #
# ================================================================ #

def _compute_h(yard: Yard, target_priority: int, N: int) -> int:
    """
    Admissible lower bound h(n).

    h(n) = |X|
         + |{ c ∈ O(c*) : c has no stack in which it can be well-located }|

    A block c (priority b_pri) is *well-located* in stack s iff
    all existing blocks in s have priority number > b_pri
    (i.e., they are retrieved AFTER c, so c on top is not non-located).
    Equivalently: min{ p(c') | c' ∈ s } > b_pri.
    """
    src_stack = _find_stack_by_priority(yard, target_priority)
    if src_stack is None:
        return 0  # target already retrieved

    src_key = (src_stack.bay, src_stack.row)

    # Locate target's tier index in src_stack
    target_idx = next(
        (i for i, c in enumerate(src_stack.containers)
         if c.priority == target_priority),
        None,
    )
    if target_idx is None:
        return 0

    blockers = src_stack.containers[target_idx + 1:]  # bottom→top above target

    # |X|
    X = _compute_nonlocated_count(yard)

    # Second term: O(c*) blocks with no well-located destination
    nowhere = 0
    for blocker in blockers:
        b_pri = blocker.priority
        can_place_well = False
        for key, stack in yard.stacks.items():
            if key == src_key:
                continue
            if stack.is_full:
                continue
            if stack.is_empty:
                can_place_well = True
                break
            min_pri = min(c.priority for c in stack.containers)
            if min_pri > b_pri:
                # All blocks in this stack are retrieved after b → b is well-located
                can_place_well = True
                break
        if not can_place_well:
            nowhere += 1

    return X + nowhere


# ================================================================ #
#  Auto-retrieval                                                     #
# ================================================================ #

def _auto_retrieve_yard(yard: Yard, N: int, start_priority: int) -> int:
    """
    Retrieve consecutive accessible targets starting at *start_priority*.
    Modifies *yard* in-place.
    Returns the next priority that is blocked (or N+1 if yard is cleared).
    """
    priority = start_priority
    while priority <= N:
        src_stack = _find_stack_by_priority(yard, priority)
        if src_stack is None:
            # Already removed – skip
            priority += 1
            continue
        if src_stack.top is None or src_stack.top.priority != priority:
            break  # blocked by another container
        src_stack.pop()
        priority += 1
    return priority


# ================================================================ #
#  Block-assignment enumeration   Section 5.2                        #
# ================================================================ #

def _enumerate_assignments(
    yard: Yard,
    n_blockers: int,
    src_key: Tuple[int, int],
    results: List[Yard],
) -> None:
    """
    Enumerate all feasible block assignments W(O(c*)).

    Relocates the top *n_blockers* containers of *src_key* one-by-one
    (LIFO) to every non-full, non-source stack.  Modifies *yard* in-place
    with undo; appends deep-copied yard states to *results*.
    """
    _enumerate_recursive(yard, n_blockers, src_key, 0, results)


def _enumerate_recursive(
    yard: Yard,
    n_blockers: int,
    src_key: Tuple[int, int],
    idx: int,
    results: List[Yard],
) -> None:
    if idx == n_blockers:
        results.append(yard.clone())
        return

    src_stack = yard.stacks[src_key]
    if src_stack.is_empty:
        results.append(yard.clone())
        return

    c = src_stack.top  # next blocker in LIFO order

    for dst_key, dst_stack in yard.stacks.items():
        if dst_key == src_key:
            continue
        if dst_stack.is_full:
            continue

        # Relocate c from src → dst
        src_stack.pop()
        dst_stack.push(c)

        _enumerate_recursive(yard, n_blockers, src_key, idx + 1, results)

        # Undo
        dst_stack.pop()
        src_stack.push(c)


# ================================================================ #
#  Attractiveness   w(n, w)   Eq. 19                                 #
# ================================================================ #

def _compute_attractiveness(
    initial_yard: Yard,
    n_blockers: int,
    src_key: Tuple[int, int],
    assign_yard: Yard,
    h_n: int,
    N: int,
) -> float:
    """
    w(n, w) = h(n) + 1 / (1 + Σ_{c ∈ O(c*)} max(s̃(c)))

    max(s̃(c)) uses the INITIAL stack maximum (before assignment),
    matching the paper's worked example (Fig. 6).
    """
    # Blocker priorities: top-n_blockers of src_key in initial_yard (top first)
    initial_src = initial_yard.stacks[src_key]
    n_total = len(initial_src.containers)
    # The top n_blockers entries (highest tiers) are the blockers
    blocker_pris = [
        initial_src.containers[i].priority
        for i in range(n_total - n_blockers, n_total)
    ]

    sum_max = 0
    for b_pri in blocker_pris:
        # Find destination stack in assign_yard by priority
        target_key: Optional[Tuple[int, int]] = None
        for key, stack in assign_yard.stacks.items():
            if key == src_key:
                continue
            for cont in stack.containers:
                if cont.priority == b_pri:
                    target_key = key
                    break
            if target_key is not None:
                break

        if target_key is None:
            sum_max += N + 1  # fallback (shouldn't occur)
        else:
            sum_max += _stack_max_priority(initial_yard.stacks[target_key], N)

    return h_n + 1.0 / (1.0 + sum_max)


# ================================================================ #
#  Greedy upper bound (Section 5.6)                                  #
# ================================================================ #

def _greedy_solve(yard: Yard, N: int, alpha_greedy: int = 1) -> int:
    """
    Fast upper-bound heuristic: run the B&B logic with α = 1
    (always pick the single most attractive block assignment).
    Returns total relocations.
    """
    current_yard = yard
    priority = _auto_retrieve_yard(current_yard, N, 1)
    total_reloc = 0

    while priority <= N:
        src_stack = _find_stack_by_priority(current_yard, priority)
        if src_stack is None:
            priority += 1
            continue

        src_key = (src_stack.bay, src_stack.row)

        # Find target's tier
        target_idx = next(
            (i for i, c in enumerate(src_stack.containers)
             if c.priority == priority),
            None,
        )
        if target_idx is None:
            priority += 1
            continue

        n_blockers = len(src_stack.containers) - target_idx - 1

        if n_blockers == 0:
            # Target on top → retrieve
            src_stack.pop()
            priority = _auto_retrieve_yard(current_yard, N, priority + 1)
            continue

        h_n = _compute_h(current_yard, priority, N)

        assignments: List[Yard] = []
        _enumerate_assignments(current_yard, n_blockers, src_key, assignments)

        if not assignments:
            break  # no valid assignment (degenerate instance)

        # Select most attractive assignment
        best_w = float("inf")
        best_assign: Optional[Yard] = None
        for ayard in assignments:
            w = _compute_attractiveness(
                current_yard, n_blockers, src_key, ayard, h_n, N
            )
            if w < best_w:
                best_w = w
                best_assign = ayard

        # Apply chosen assignment
        current_yard = best_assign  # type: ignore[assignment]
        total_reloc += n_blockers

        # Retrieve target (now on top of src)
        current_yard.stacks[src_key].pop()
        priority = _auto_retrieve_yard(current_yard, N, priority + 1)

    return total_reloc


# ================================================================ #
#  Main B&B routine                                                   #
# ================================================================ #

def _run_bb(
    initial_yard: Yard,
    N: int,
    alpha,          # int or float("inf")
    stop_event: mp.Event,
    time_limit_s: float,
    t_start: float,
) -> Tuple[int, bool, int]:
    """
    Algorithm 1 from the paper.
    Returns (best_relocations, optimal_proven, nodes_explored).
    """
    # ── Initial upper bound (Section 5.6): greedy with α=1 ──────── #
    ub_yard = initial_yard.clone()
    ub = _greedy_solve(ub_yard, N, alpha_greedy=1)

    # ── Root node ─────────────────────────────────────────────────── #
    root_yard = initial_yard.clone()
    root_priority = _auto_retrieve_yard(root_yard, N, 1)

    if root_priority > N:
        return 0, True, 1  # already solved (sorted yard)

    h0 = _compute_h(root_yard, root_priority, N)

    if h0 >= ub:
        return ub, (alpha == float("inf")), 1

    counter = itertools.count()
    root_node = _BBNode(root_yard, 0, root_priority, h0)
    heap: List[Tuple] = [(root_node.f, next(counter), root_node)]

    best_reloc = ub
    nodes_explored = 0

    while heap:
        if stop_event.is_set():
            break
        if time.perf_counter() - t_start > time_limit_s:
            break

        f_val, _, node = heapq.heappop(heap)

        # Pruning (line 8 of Algorithm 1)
        if f_val >= best_reloc:
            continue

        nodes_explored += 1

        priority = node.priority
        yard     = node.yard
        g        = node.g

        # ── All blocks retrieved ─────────────────────────────────── #
        if priority > N:
            if g < best_reloc:
                best_reloc = g
            continue

        # ── Find target in yard ──────────────────────────────────── #
        src_stack = _find_stack_by_priority(yard, priority)
        if src_stack is None:
            # Priority already gone (shouldn't happen with correct logic)
            continue

        src_key    = (src_stack.bay, src_stack.row)
        target_idx = next(
            (i for i, c in enumerate(src_stack.containers)
             if c.priority == priority),
            None,
        )
        if target_idx is None:
            continue

        n_blockers = len(src_stack.containers) - target_idx - 1

        # ── Case A: target already on top (lines 9-16) ──────────── #
        if n_blockers == 0:
            new_yard = yard.clone()
            new_yard.stacks[src_key].pop()  # retrieve target
            new_priority = _auto_retrieve_yard(new_yard, N, priority + 1)

            if new_priority > N:
                if g < best_reloc:
                    best_reloc = g
                continue

            h_new = _compute_h(new_yard, new_priority, N)
            if g + h_new < best_reloc:
                new_node = _BBNode(new_yard, g, new_priority, h_new)
                heapq.heappush(heap, (new_node.f, next(counter), new_node))

        # ── Case B: target is buried (lines 17-20) ──────────────── #
        else:
            h_n = _compute_h(yard, priority, N)

            assignments: List[Yard] = []
            _enumerate_assignments(yard, n_blockers, src_key, assignments)

            if not assignments:
                continue  # degenerate – no valid move

            # Score by attractiveness (Eq. 19)
            scored: List[Tuple[float, Yard]] = []
            for ayard in assignments:
                w = _compute_attractiveness(
                    yard, n_blockers, src_key, ayard, h_n, N
                )
                scored.append((w, ayard))

            scored.sort(key=lambda x: x[0])

            # α-restriction (line 19): keep at most α most attractive
            if alpha != float("inf"):
                scored = scored[: int(alpha)]

            new_g = g + n_blockers

            for _, ayard in scored:
                # Retrieve target (now on top of src in ayard)
                assign_src = ayard.stacks[src_key]
                if assign_src.is_empty or assign_src.top.priority != priority:
                    continue  # shouldn't occur

                assign_src.pop()  # retrieve c*
                new_priority = _auto_retrieve_yard(ayard, N, priority + 1)

                if new_priority > N:
                    if new_g < best_reloc:
                        best_reloc = new_g
                    continue

                h_new = _compute_h(ayard, new_priority, N)
                if new_g + h_new < best_reloc:
                    new_node = _BBNode(ayard, new_g, new_priority, h_new)
                    heapq.heappush(heap, (new_node.f, next(counter), new_node))

    time_ok    = (time.perf_counter() - t_start) <= time_limit_s
    stopped    = stop_event.is_set()
    optimal    = (not stopped) and time_ok and (len(heap) == 0 or alpha == float("inf"))
    return best_reloc, optimal, nodes_explored


# ================================================================ #
#  BaseAlgorithm subclass                                             #
# ================================================================ #

class ExposioBB2015(BaseAlgorithm):
    """
    Branch-and-Bound exact solver for Restricted BRP.

    Reference: Expósito-Izquierdo, Melián-Batista, Moreno-Vega (2015).

    Parameters
    ----------
    alpha : int
        Maximum descending nodes per B&B tree node.
        0 = unlimited (guarantees optimality).
        Smaller values trade optimality for speed.
    time_limit_s : float
        Per-seed wall-clock time limit in seconds (default 600).
    """

    name                = "Expósito-Izquierdo (2015) B&B"
    category            = "Exact"
    description         = (
        "Best-first branch-and-bound for the Restricted BRP "
        "(Expósito-Izquierdo et al., ESWA 2015). "
        "Admissible lower bound h(n) + α-restricted branching. "
        "α=0 (unlimited) guarantees optimality; α=1..5 gives fast near-optimal results."
    )
    compatible_problems = ["CRP-R"]
    step_label          = "Seed"

    def __init__(self, config: Optional[AlgorithmConfig] = None) -> None:
        super().__init__(config)

    # ---------------------------------------------------------------- #
    # Train                                                              #
    # ---------------------------------------------------------------- #

    def train(
        self,
        problem_factory: Callable,
        result_queue:    mp.Queue,
        stop_event:      mp.Event,
    ) -> None:
        cfg   = self.config
        extra = cfg.extra or {}

        raw_alpha  = extra.get("alpha", 0)
        alpha: object = float("inf") if (raw_alpha == 0 or raw_alpha == "inf") else int(raw_alpha)
        time_limit = float(extra.get("time_limit_s", 600))
        n_seeds    = max(1, cfg.num_eval_seeds)
        all_metrics: List[Dict] = []

        for seed in range(n_seeds):
            if stop_event.is_set():
                break

            t_start = time.perf_counter()

            env = problem_factory()
            env.config.seed = seed
            env.reset(seed=seed, options={"skip_auto_retrieve": True})

            yard0 = env.yard.clone()
            N     = len(env.containers)

            n_reloc, optimal, nodes = _run_bb(
                yard0, N, alpha, stop_event, time_limit, t_start
            )

            t_elapsed = time.perf_counter() - t_start

            metrics: Dict = {
                "relocations":    float(n_reloc),
                "steps":          float(n_reloc),
                "time":           float(n_reloc),
                "nodes_explored": float(nodes),
                "optimal_proven": 1.0 if optimal else 0.0,
                "solve_time_s":   round(t_elapsed, 4),
            }
            all_metrics.append(metrics)
            primary = float(n_reloc)

            if primary < self._best_metric:
                self._best_metric   = primary
                self._best_solution = []  # move list not tracked

            self._push(
                result_queue,
                step     = seed + 1,
                metric   = primary,
                metrics  = metrics,
                progress = (seed + 1) / n_seeds,
                snapshot = env.get_state_snapshot(),
            )

            print(
                f"[ExposioBB2015] seed={seed}  relocations={n_reloc}  "
                f"optimal={optimal}  nodes={nodes}  α={alpha}  t={t_elapsed:.3f}s",
                file=sys.stderr,
                flush=True,
            )

        if all_metrics:
            agg = {
                k: float(np.mean([m[k] for m in all_metrics if k in m]))
                for k in all_metrics[0]
            }
            self._push(
                result_queue,
                step     = n_seeds,
                metric   = self._best_metric,
                metrics  = agg,
                progress = 1.0,
            )

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution

    # ---------------------------------------------------------------- #
    # Config schema (for GUI / API)                                      #
    # ---------------------------------------------------------------- #

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update({
            "num_eval_seeds": {
                "type": "int", "default": 1, "min": 1, "max": 20,
                "label": "Seeds",
                "help": "Number of independent seeds to evaluate.",
            },
            "alpha": {
                "type": "int", "default": 0, "min": 0, "max": 100,
                "label": "α (0 = unlimited / optimal)",
                "help": (
                    "Max descending B&B nodes per parent. "
                    "0 = unlimited (optimal guaranteed). "
                    "1–5 = fast near-optimal (paper recommends ≥4 for full optimality)."
                ),
            },
            "time_limit_s": {
                "type": "float", "default": 600.0, "min": 1.0, "max": 86400.0,
                "label": "Time limit (s)",
                "help": "Per-seed wall-clock time limit in seconds.",
            },
        })
        return base

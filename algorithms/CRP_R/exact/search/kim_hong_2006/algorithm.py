"""
Kim & Hong (2006) exact Branch-and-Bound for the Restricted BRP (CRP-R).

Reference
---------
K.H. Kim, G.-P. Hong, "A heuristic rule for relocating blocks",
Computers & Operations Research 33 (2006) 940-954, Section 2
(case with precedence relationships among *individual* blocks,
Section 2.1 -- this is the case that matches CRP-R's single, fully
serial priority chain 1..N).

Algorithm overview (Section 2 / 2.1)
-------------------------------------
This is a *combinatorial* branch-and-bound: it searches directly over
bay-configuration states, with no IP/LP formulation and no LP-relaxation
bound underneath it (unlike the ``exact/solver`` algorithms in this
platform, e.g. ``wan_2009`` or ``caserta_2012_brp2``).

  - State S_k       = yard configuration after k blocks have been
                       retrieved.
  - Action a_k       = relocate every blocker above the k-th target
                       (one at a time, LIFO order, to some *other* stack
                       -- Assumption 3 of the paper), then retrieve the
                       target (and auto-retrieve any further already
                       accessible targets).
  - F(S_k)           = minimum number of additional relocations needed
                       to clear the remaining N-k blocks;
                       F(S_0) = min_{a1} { h(a1|S0) + F(S1) }.
  - Node selection   = depth-first + backtracking.  Among the children
                       generated at a node, the paper explores the
                       unexplored node with the *minimum lower bound*
                       first (Section 2.1).
  - Lower bound      = (relocations already realised, root -> node)
                       + "confirmed relocations" of the node, where a
                       confirmed relocation is a container sitting above
                       another container of *smaller* priority number in
                       the same stack -- it is certain to require at
                       least one relocation eventually, no matter what
                       is decided from here on.
  - Dominance rules  = explicitly *not* used (the paper found the extra
                       bookkeeping not worth the small amount of
                       pruning it bought -- see Section 2.1).

Independence
------------
This module only depends on the shared platform infrastructure
(``core.base_algorithm``, ``core.yard``) and defines all of its own
helper functions privately (leading underscore).  It does not import
anything from any other algorithm package (``heuristic.kim_hong``,
``exact.search.exposito_2015``, ``exact.solver.*``, ...), so it can be
added, changed or removed without ever touching -- or being coupled
to -- any other algorithm already registered in the platform.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig
from core.yard import Yard, Stack


# ================================================================ #
#  Yard helpers (private -- self-contained)                         #
# ================================================================ #

def _find_stack_by_priority(yard: Yard, priority: int) -> Optional[Stack]:
    """Return the stack containing the container with *priority*, or None."""
    for stack in yard.all_stacks():
        for c in stack.containers:
            if c.priority == priority:
                return stack
    return None


def _auto_retrieve(yard: Yard, n_total: int, start_priority: int) -> int:
    """
    Retrieve consecutive already-accessible targets, starting at
    *start_priority*.  Modifies *yard* in place.
    Returns the next priority that is blocked (or n_total+1 once the
    yard has been fully cleared).
    """
    priority = start_priority
    while priority <= n_total:
        stack = _find_stack_by_priority(yard, priority)
        if stack is None:
            priority += 1
            continue
        if stack.top is None or stack.top.priority != priority:
            break  # blocked -- wait for a relocation decision
        stack.pop()
        priority += 1
    return priority


# ================================================================ #
#  Lower bound: "confirmed relocations" (Section 2.1)               #
# ================================================================ #

def _confirmed_relocations(yard: Yard) -> int:
    """
    Count containers that sit above another container of *smaller*
    priority number in the same stack.  Each such container is
    "confirmed" to require at least one relocation eventually, no
    matter which future decisions are made -- this is exactly the
    quantity the paper adds to the already-realised relocation count
    to obtain the lower bound of a B&B node (Section 2.1).
    """
    count = 0
    for stack in yard.all_stacks():
        min_below: Optional[int] = None
        for c in stack.containers:  # bottom -> top
            if min_below is not None and c.priority > min_below:
                count += 1
            min_below = c.priority if min_below is None else min(min_below, c.priority)
    return count


# ================================================================ #
#  Branching: enumerate every feasible destination assignment       #
#  for the blockers above the current target (LIFO, "other" stacks  #
#  only -- Assumption 3)                                            #
# ================================================================ #

def _enumerate_assignments(
    yard: Yard,
    n_blockers: int,
    src_key: Tuple[int, int],
) -> List[Yard]:
    """
    Enumerate every feasible way of relocating the top *n_blockers*
    containers of ``yard.stacks[src_key]`` to *other* stacks, one at a
    time in LIFO order.  These are exactly "the branches that follow
    directly" a node, as illustrated in the paper's worked example
    (Section 2.1).

    Returns a list of yard *clones*; in every returned clone the
    target container is now on top of ``src_key``, ready to be
    retrieved by the caller.  *yard* itself is left unmodified.
    """
    results: List[Yard] = []
    _enumerate_recursive(yard, n_blockers, src_key, 0, results)
    return results


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
    blocker = src_stack.top  # next blocker to move, LIFO order

    for dst_key, dst_stack in yard.stacks.items():
        if dst_key == src_key or dst_stack.is_full:
            continue

        src_stack.pop()
        dst_stack.push(blocker)

        _enumerate_recursive(yard, n_blockers, src_key, idx + 1, results)

        # Undo, so *yard* is restored before trying the next destination.
        dst_stack.pop()
        src_stack.push(blocker)


# ================================================================ #
#  Depth-first branch-and-bound (Section 2 / 2.1)                   #
# ================================================================ #

class _SearchContext:
    """Mutable state shared across recursive calls of one B&B run."""

    __slots__ = ("n_total", "best", "nodes", "stop_event", "deadline", "aborted")

    def __init__(self, n_total: int, stop_event: mp.Event, deadline: float) -> None:
        self.n_total    = n_total
        self.best        = float("inf")
        self.nodes       = 0
        self.stop_event  = stop_event
        self.deadline    = deadline
        self.aborted     = False


def _should_abort(ctx: _SearchContext) -> bool:
    if ctx.aborted:
        return True
    if ctx.nodes % 256 == 0:
        if ctx.stop_event.is_set() or time.perf_counter() > ctx.deadline:
            ctx.aborted = True
            return True
    return False


def _dfs(yard: Yard, g: int, priority: int, ctx: _SearchContext) -> None:
    """
    Depth-first branch-and-bound over bay states (Algorithm of
    Section 2 / 2.1).  Updates ``ctx.best`` in place; no return value.
    """
    ctx.nodes += 1
    if _should_abort(ctx):
        return

    # ── All blocks retrieved: candidate (possibly optimal) solution ── #
    if priority > ctx.n_total:
        if g < ctx.best:
            ctx.best = g
        return

    # ── Lower-bound pruning at this node ─────────────────────────── #
    lb = g + _confirmed_relocations(yard)
    if lb >= ctx.best:
        return

    src_stack = _find_stack_by_priority(yard, priority)
    if src_stack is None:
        return  # defensive: should not happen

    src_key    = (src_stack.bay, src_stack.row)
    target_idx = next(
        (i for i, c in enumerate(src_stack.containers) if c.priority == priority),
        None,
    )
    if target_idx is None:
        return  # defensive: should not happen

    n_blockers = len(src_stack.containers) - target_idx - 1

    # ── Target already accessible: no branching needed ─────────────── #
    if n_blockers == 0:
        new_yard = yard.clone()
        new_yard.stacks[src_key].pop()
        new_priority = _auto_retrieve(new_yard, ctx.n_total, priority + 1)
        _dfs(new_yard, g, new_priority, ctx)
        return

    # ── Branch: enumerate all feasible relocation assignments ─────── #
    children = _enumerate_assignments(yard, n_blockers, src_key)
    if not children:
        return  # degenerate instance: no feasible relocation

    new_g = g + n_blockers

    # Rank children by their own lower bound: the paper selects "the
    # unexplored node with the minimum lower bound" among ties at the
    # deepest level (Section 2.1) -- exploring in ascending order of
    # child lower bound (depth-first) reproduces this rule.
    scored: List[Tuple[float, Yard]] = [
        (new_g + _confirmed_relocations(child), child) for child in children
    ]
    scored.sort(key=lambda t: t[0])

    for child_lb, child in scored:
        if ctx.aborted:
            return
        if child_lb >= ctx.best:
            continue  # prune before descending

        child.stacks[src_key].pop()  # retrieve target (now on top)
        new_priority = _auto_retrieve(child, ctx.n_total, priority + 1)
        _dfs(child, new_g, new_priority, ctx)


def _run_bb(
    initial_yard: Yard,
    n_total:      int,
    stop_event:   mp.Event,
    time_limit_s: float,
) -> Tuple[int, bool, int]:
    """
    Run the Kim & Hong (2006) exact B&B on *initial_yard*.
    Returns (best_relocations, optimal_proven, nodes_explored).
    """
    yard0          = initial_yard.clone()
    start_priority = _auto_retrieve(yard0, n_total, 1)

    if start_priority > n_total:
        return 0, True, 1  # already fully sorted

    ctx = _SearchContext(
        n_total    = n_total,
        stop_event = stop_event,
        deadline   = time.perf_counter() + time_limit_s,
    )
    _dfs(yard0, 0, start_priority, ctx)

    best    = int(ctx.best) if ctx.best < float("inf") else -1
    optimal = (not ctx.aborted) and ctx.best < float("inf")
    return best, optimal, ctx.nodes


# ================================================================ #
#  BaseAlgorithm subclass                                            #
# ================================================================ #

class KimHong2006BB(BaseAlgorithm):
    """
    Exact branch-and-bound for the Restricted BRP / CRP-R.

    Reference: Kim & Hong (2006), Computers & Operations Research 33,
    Section 2 (individual-precedence case, Section 2.1).

    Unlike the exact/solver algorithms in this platform, this is *not*
    an integer program: it is a hand-crafted combinatorial search
    directly over bay-configuration states, using a problem-specific
    lower bound ("confirmed relocations") and depth-first backtracking,
    exactly as described in the original 1998-2006-era literature.

    Parameters
    ----------
    time_limit_s : float
        Per-seed wall-clock time limit in seconds (default 300).
        If the search is stopped early, the best solution found so far
        is reported and ``optimal_proven`` is set to 0.0.
    """

    name                = "Kim & Hong (2006) B&B"
    category            = "Exact"
    description         = (
        "Depth-first branch-and-bound directly over bay-configuration "
        "states (Kim & Hong, COR 2006, Section 2.1) -- no IP formulation. "
        "Lower bound = realised relocations + 'confirmed relocations' "
        "(containers stacked above a smaller-priority container). "
        "Guarantees the optimal number of relocations if it completes "
        "within the time limit."
    )
    compatible_problems = ["CRP-R"]
    step_label           = "Seed"

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
        cfg        = self.config
        extra      = cfg.extra or {}
        time_limit = float(extra.get("time_limit_s", 300.0))
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
            n_total = len(env.containers)

            n_reloc, optimal, nodes = _run_bb(
                yard0, n_total, stop_event, time_limit
            )
            if n_reloc < 0:
                # No complete solution found within the time limit.
                n_reloc = int(env.get_metrics().get("relocations", 0.0))

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
                self._best_solution = []  # move list not tracked (state-space search)

            self._push(
                result_queue,
                step     = seed + 1,
                metric   = primary,
                metrics  = metrics,
                progress = (seed + 1) / n_seeds,
                snapshot = env.get_state_snapshot(),
            )

            print(
                f"[KimHong2006BB] seed={seed}  relocations={n_reloc}  "
                f"optimal={optimal}  nodes={nodes}  t={t_elapsed:.3f}s",
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
            "time_limit_s": {
                "type": "float", "default": 300.0, "min": 1.0, "max": 86400.0,
                "label": "Time limit (s)",
                "help": (
                    "Per-seed wall-clock time limit in seconds. The original "
                    "1998 depth-first B&B has no node/branch restriction, so "
                    "runtime can grow very quickly with stack size -- the "
                    "paper itself reports multi-minute solves for bays as "
                    "small as 5 stacks x 6 tiers."
                ),
            },
        })
        return base

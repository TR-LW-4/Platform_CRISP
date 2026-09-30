"""
Step 2 of López-Plata et al. (2019): completing the partial solutions with Expansion
(Sec. 6.2, p. 444) inside the loop of Algorithm 2, lines 3-14 (p. 443).
Nodes are Bay objects carrying phi' and g (operating_cost.py); stacks are numbered 1..n_S.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

from .operating_cost import Bay, Move

MQ = (1.0, 2.0, 1.0, 0.5)   # mq_m, mq_b, mq_c, mq_r (p. 446)


def q_m(bay: Bay, s: int, d: int) -> float:
    """Eq. 27: carry c+ from s to the new top of d, then return empty to the new top of s."""
    n_s, n_d = len(bay.stacks[s - 1]), len(bay.stacks[d - 1])
    return bay.cost.m_loaded(s, n_s, d, n_d + 1) + bay.cost.m(d, n_d + 1, s, n_s - 1)


def alphas(bay: Bay, n_C: int, mq: Sequence[float] = MQ) -> List[Tuple[float, int]]:
    """Eq. 28 for relocating c+ from s(c*) to every other non-full stack d: (alpha, d).
    q_b, q_c and q_r are those of c+ on d after the move (O11, Fig. 4); maxMov is the largest
    q_m over these same destinations; maxDist = n_S."""
    s = bay.target_stack()
    c, n_S = bay.stacks[s - 1][-1], len(bay.stacks)
    dests = [d for d in range(1, n_S + 1) if d != s and len(bay.stacks[d - 1]) < bay.cost.n_T]
    moves = [q_m(bay, s, d) for d in dests]
    max_mov = max(moves, default=1.0)
    out = []
    for d, move in zip(dests, moves):
        dest = bay.stacks[d - 1]
        q_b = 1 if dest and min(dest) < c else 0                        # eq. 23
        q_c = c - min(min(dest), c) if dest else c - bay.target          # eq. 24 (O10 (a) if c is alone)
        out.append((move / max_mov * mq[0] + q_b * mq[1] + q_c / n_C * mq[2] + d / n_S * mq[3], d))
    return out


def children(bay: Bay, tau: int, n_C: int, mq: Sequence[float] = MQ) -> List[Tuple[float, int]]:
    """The tau destinations with the lowest alpha, ties to the lowest stack index (O08)."""
    return sorted(alphas(bay, n_C, mq))[:tau]


def sequences(node: Bay, kappa: int, tau: int, n_C: int,
              mq: Sequence[float] = MQ) -> List[Tuple[Bay, float]]:
    """The subtree of Expansion from node: c* is retrieved when it is on top, otherwise c+ goes to
    each of the tau children; a sequence ends right after min(kappa, blocks left) retrievals.
    Returns (end node, sum of alpha over its relocations) in generation order."""
    bay, out = node.copy(), []
    goal = min(kappa, sum(len(st) for st in bay.stacks))

    def grow(retrieved: int, total: float) -> None:
        if retrieved == goal:
            out.append((bay.copy(), total))
            return
        s = bay.target_stack()
        before = bay.crane, bay.target, bay.g
        if bay.stacks[s - 1][-1] == bay.target:
            bay.apply((s, 0))
            grow(retrieved + 1, total)
            bay.undo((s, 0), *before)
            return
        for a, d in children(bay, tau, n_C, mq):
            bay.apply((s, d))
            grow(retrieved, total + a)
            bay.undo((s, d), *before)

    grow(0, 0.0)
    return out


def roulette(sums: Sequence[float]) -> List[float]:
    """Selection probabilities, fitness-proportional with fitness 1 / (sum of alpha) (O09b)."""
    inverse = [1.0 / a for a in sums]
    total = sum(inverse)
    return [x / total for x in inverse]


def draw(ranked: List[Tuple[object, float]], rng: random.Random) -> object:
    """Roulette wheel over (item, sum of alpha); no draw when there is no choice."""
    if len(ranked) == 1:
        return ranked[0][0]
    assert all(a > 0 for _, a in ranked), "every sequence with a choice contains a relocation"
    return rng.choices([item for item, _ in ranked], weights=roulette([a for _, a in ranked]))[0]


def expand(node: Bay, rng: random.Random, kappa: int, tau: int, mu: int, n_C: int,
           mq: Sequence[float] = MQ, cache: Optional[Dict[tuple, list]] = None) -> Bay:
    """Expansion (p. 444): of the sequences from node, the mu with the lowest sum of alpha (O09a,
    ties in generation order) enter a roulette wheel; the drawn moves are applied to node.
    The sequences and their sums of alpha depend on the configuration only (alpha does not use the
    crane position), so with a cache they are computed once per configuration of a run."""
    key = tuple(map(tuple, node.stacks))
    best = cache.get(key) if cache is not None else None
    if best is None:
        start = len(node.phi)
        best = [(tuple(end.phi[start:]), a)
                for end, a in sorted(sequences(node, kappa, tau, n_C, mq), key=lambda seq: seq[1])[:mu]]
        if not best:
            raise ValueError("Expansion: no feasible sequence")
        if cache is not None:
            cache[key] = best
    end = node.copy()
    for move in draw(best, rng):
        end.apply(move)
    return end


def algorithm2_loop(partials: List[Bay], phi_best: List[Move], f_best: float, rng: random.Random,
                    rho: int, kappa: int, tau: int, mu: int, n_C: int,
                    mq: Sequence[float] = MQ) -> Tuple[List[Move], float]:
    """Algorithm 2, lines 3-14: rho iterations (O15), each starting from the same N; every node is
    expanded on its own until it is complete, and a complete solution replaces phi_best only if its
    f is strictly lower (line 10)."""
    cache: Dict[tuple, list] = {}
    for _ in range(rho):
        for node in partials:
            done = node
            while done.target is not None:
                done = expand(done, rng, kappa, tau, mu, n_C, mq, cache)
            if done.g < f_best:
                phi_best, f_best = list(done.phi), done.g
    return phi_best, f_best

"""
JovanovicACO_uBRP
<2019> <heuristic> <unrestricted> <single-bay> <CRP-U>
ACO-C with Gre-C extended candidate list
n_iterations --- 1000 --- Colony iterations per layout
n_ants --- 10 --- Ants per iteration
q0 --- 0.9 --- Exploitation rate

------------------------------- Reference --------------------------------
R. Jovanović, M. Tuba, S. Voß,
"An efficient ant colony optimization algorithm for the blocks relocation
 problem",
European Journal of Operational Research 274 (2019) 78–90.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from algorithms.ALGORITHM import Algorithm
from algorithms.fixed.CRP_U.move import Move
from .scoring import compute_lb, dif, run_greedy_ubrp


class ACO(Algorithm):

    def main(self):
        n_iters   = self.parameter("n_iterations", 1000, int)
        n_ants    = self.parameter("n_ants", 10, int)
        q0        = self.parameter("q0", 0.9, float)
        rho       = self.parameter("rho", 0.1, float)
        phi       = self.parameter("phi", 0.9, float)
        max_moves = self.parameter("max_moves", 10, int)
        max_const = self.parameter("max_const_iter", 100, int)
        np.random.seed(self.parameter("seed", 0, int))

        n_total   = int(self.problem.num_containers)
        max_tiers = int(self.problem.max_tiers)
        n_stacks  = len(self.problem.stacks)

        all_keys = list(range(n_stacks))
        key_to_1based_idx = {index: index + 1 for index in all_keys}
        stacks_init = {index: list(stack) for index, stack in enumerate(self.problem.stacks)}

        # ── Gre-C warm-start + initial LB ────────────────────────── #
        S_best, best_cost, best_path = run_greedy_ubrp(
            stacks_init, all_keys, n_total, max_tiers, key_to_1based_idx
        )
        lb_init = compute_lb(stacks_init)

        # ── Pheromone initialisation (Eq. 37) ────────────────────── #
        val_greedy = 1.0 / max(best_cost - lb_init + 1, 1)
        tau_0      = val_greedy / n_stacks
        tau_min    = tau_0 / n_stacks

        pheromone = np.full(
            (n_total, n_total + n_stacks, max_moves + 1, n_total),
            tau_0, dtype=np.float32,
        )

        # ── Opt: precompute location array, smin, empty_d ────────── #
        # loc_init[c]       = stack_key containing container c (1..N)
        # stack_min_init[k] = min priority in stack k (N+1 if empty)
        # empty_d[k]        = dd*(empty stack k) for pheromone indexing
        loc_init:       List[Any]        = [None] * (n_total + 1)
        stack_min_init: Dict[Any, int]   = {}
        empty_d:        Dict[Any, int]   = {}

        for key, prios in stacks_init.items():
            stack_min_init[key] = min(prios) if prios else (n_total + 1)
            empty_d[key]        = n_total + key_to_1based_idx[key]
            for p in prios:
                loc_init[p] = key

        iters_no_improve = 0

        # ── Main ACO loop ─────────────────────────────────────────── #
        for iteration in range(n_iters):
            iter_improved = False

            # ── Colony: each ant builds one solution ──────────────── #
            for _ant in range(n_ants):
                # Reset per-ant state  (O(N + W) copies)
                stacks: Dict[Any, List[int]] = {
                    k: list(v) for k, v in stacks_init.items()
                }
                M         = [0] * (n_total + 1)      # relocation counts
                S: List[Tuple[int, int, int, int]] = []
                path: List[Tuple[int, Any, Any]] = []
                valid     = True
                loc       = list(loc_init)            # O(N) copy
                smin      = dict(stack_min_init)      # O(W) copy
                lb_curr   = lb_init

                for target in range(1, n_total + 1):
                    if not valid:
                        break

                    src_key = loc[target]
                    if src_key is None:
                        continue

                    # ── Relocate until target is on top ───────────── #
                    safety = 0
                    while (
                        valid
                        and stacks[src_key]
                        and stacks[src_key][-1] != target
                    ):
                        safety += 1
                        if safety > n_total * n_total:
                            valid = False
                            break

                        # ── Build candidate list Ĉ (Eqs. 9–23) ────── #
                        top_c = stacks[src_key][-1]

                        # T: top blocker → any non-full non-source stack
                        T_dsts: List[Any] = [
                            dst for dst in all_keys
                            if dst != src_key and len(stacks[dst]) < max_tiers
                        ]

                        # Tn: non-WL tops not in src_key (Eq. 10)
                        Tn: List[int] = []
                        for key in all_keys:
                            if key == src_key or not stacks[key]:
                                continue
                            c_cand = stacks[key][-1]
                            if smin[key] < c_cand:
                                Tn.append(c_cand)

                        # Or: Tn containers → destinations where WL (Eq. 12)
                        Or_cands: List[Tuple[int, Any, Any]] = []
                        for c_cand in Tn:
                            c_key_cand = loc[c_cand]
                            for dst in all_keys:
                                if (
                                    dst != c_key_cand
                                    and len(stacks[dst]) < max_tiers
                                    and smin[dst] > c_cand
                                ):
                                    Or_cands.append((c_cand, c_key_cand, dst))

                        # NR: can top_c or any Tn member be well-located?
                        NR_exists = len(Or_cands) > 0
                        if not NR_exists:
                            for dst in T_dsts:
                                if smin[dst] > top_c:
                                    NR_exists = True
                                    break

                        # Build combined candidate list
                        # Entries: (c, c_src_key, dst_key)
                        all_cands: List[Tuple[int, Any, Any]] = [
                            (top_c, src_key, dst) for dst in T_dsts
                        ]

                        if NR_exists:
                            # Cr = T ∪ Or (Eq. 13)
                            all_cands.extend(Or_cands)
                        else:
                            # Ow moves (Eqs. 17–20)
                            max_n  = max(Tn) if Tn else 0
                            Tn_set = set(Tn)
                            for key in all_keys:
                                if key == src_key or not stacks[key]:
                                    continue
                                c_cand = stacks[key][-1]
                                if c_cand in Tn_set or c_cand == target:
                                    continue
                                if smin[key] < c_cand:
                                    continue    # not WL
                                below = stacks[key][:-1]
                                dd_below = (
                                    min(below) if below else (n_total + 1)
                                )
                                if dd_below <= max_n:
                                    continue
                                for dst in all_keys:
                                    if dst != key and len(stacks[dst]) < max_tiers:
                                        all_cands.append((c_cand, key, dst))

                        if not all_cands:
                            valid = False
                            break

                        # ── Compute g(α) for each candidate ──────── #
                        best_g    = -1.0
                        best_c    = -1
                        best_csrc: Any = None
                        best_dst:  Any = None
                        best_d    = 0

                        cands_g:   List[float] = []
                        cands_c:   List[int]   = []
                        cands_src: List[Any]   = []
                        cands_dst: List[Any]   = []
                        cands_d:   List[int]   = []

                        t_i = target - 1

                        for c_cand, c_src, dst in all_cands:
                            c_i   = c_cand - 1
                            mc_cl = min(M[c_cand], max_moves)

                            sm    = smin[dst]
                            d_val = sm if sm <= n_total else empty_d[dst]

                            # dife heuristic (Eq. 21): use dd(S) for scoring
                            d_dd     = sm if sm <= n_total else (n_total + 1)
                            base_dif = (
                                (d_dd - c_cand)
                                if d_dd > c_cand
                                else (2 * n_total + 1 - d_dd)
                            )
                            if smin[c_src] == c_cand:   # c is WL in src
                                below    = stacks[c_src][:-1]
                                dd_below = (
                                    min(below) if below else (n_total + 1)
                                )
                                dife_val = n_total + base_dif - (dd_below - c_cand)
                            else:
                                dife_val = base_dif

                            f_val = 1.0 / max(1e-9, 1.0 + dife_val)
                            tau   = float(pheromone[c_i, d_val - 1, mc_cl, t_i])
                            g     = f_val * tau

                            cands_g.append(g)
                            cands_c.append(c_cand)
                            cands_src.append(c_src)
                            cands_dst.append(dst)
                            cands_d.append(d_val)

                            if g > best_g:
                                best_g    = g
                                best_c    = c_cand
                                best_csrc = c_src
                                best_dst  = dst
                                best_d    = d_val

                        # ── Transition rule (Eqs. 31–32) ─────────── #
                        if (
                            np.random.uniform() < q0
                            or len(all_cands) == 1
                        ):
                            chosen_c   = best_c
                            chosen_src = best_csrc
                            chosen_dst = best_dst
                            chosen_d   = best_d
                        else:
                            g_arr  = np.asarray(cands_g, dtype=np.float64)
                            g_sum  = g_arr.sum()
                            probs  = (
                                g_arr / g_sum
                                if g_sum > 0.0
                                else np.ones(len(cands_g)) / len(cands_g)
                            )
                            idx        = np.random.choice(len(cands_g), p=probs)
                            chosen_c   = cands_c[idx]
                            chosen_src = cands_src[idx]
                            chosen_dst = cands_dst[idx]
                            chosen_d   = cands_d[idx]

                        # ── Incremental LB (Opt-2) ────────────────── #
                        was_non_wl  = smin[chosen_src] < chosen_c
                        will_non_wl = smin[chosen_dst] < chosen_c

                        # ── Apply relocation ──────────────────────── #
                        mc_rec = min(M[chosen_c], max_moves)
                        S.append((chosen_c, chosen_d, mc_rec, target))
                        path.append((chosen_c, chosen_src, chosen_dst))

                        stacks[chosen_src].pop()
                        if smin[chosen_src] == chosen_c:
                            smin[chosen_src] = (
                                min(stacks[chosen_src])
                                if stacks[chosen_src]
                                else (n_total + 1)
                            )

                        stacks[chosen_dst].append(chosen_c)
                        if chosen_c < smin[chosen_dst]:
                            smin[chosen_dst] = chosen_c

                        loc[chosen_c]  = chosen_dst
                        M[chosen_c]   += 1

                        if was_non_wl:
                            lb_curr -= 1
                        if will_non_wl:
                            lb_curr += 1

                        # ── Early termination ─────────────────────── #
                        if len(S) + lb_curr >= best_cost:
                            valid = False

                    # Retrieve target
                    if (
                        valid
                        and stacks[src_key]
                        and stacks[src_key][-1] == target
                    ):
                        path.append((target, src_key, None))
                        stacks[src_key].pop()
                        if smin[src_key] == target:
                            smin[src_key] = (
                                min(stacks[src_key])
                                if stacks[src_key]
                                else (n_total + 1)
                            )
                        loc[target] = None

                # ── Local pheromone update (Eq. 36) ──────────────── #
                for c, d_val, mc_rec, t in S:
                    c_i = c - 1
                    d_i = d_val - 1
                    t_i = t - 1
                    if (
                        0 <= c_i < n_total
                        and 0 <= d_i < n_total + n_stacks
                        and 0 <= t_i < n_total
                    ):
                        pheromone[c_i, d_i, mc_rec, t_i] = max(
                            np.float32(tau_min),
                            np.float32(pheromone[c_i, d_i, mc_rec, t_i] * phi),
                        )

                # ── Update best ───────────────────────────────────── #
                if valid and len(S) < best_cost:
                    best_cost        = len(S)
                    S_best           = S.copy()
                    best_path        = path.copy()
                    iter_improved    = True
                    iters_no_improve = 0

            # ── Stagnation reinitialisation ───────────────────────── #
            if not iter_improved:
                iters_no_improve += 1
            if iters_no_improve >= max_const:
                val_b   = 1.0 / max(best_cost - lb_init + 1, 1)
                tau_new = val_b / n_stacks
                tau_min = tau_new / n_stacks
                pheromone[:] = np.float32(tau_new)
                iters_no_improve = 0

            # ── Global pheromone update (Eqs. 34–35) ─────────────── #
            val_b   = 1.0 / max(best_cost - lb_init + 1, 1)
            delta   = np.float32(val_b)
            tau_min = val_b / (n_stacks ** 2)
            for c, d_val, mc_rec, t in S_best:
                c_i = c - 1
                d_i = d_val - 1
                t_i = t - 1
                if (
                    0 <= c_i < n_total
                    and 0 <= d_i < n_total + n_stacks
                    and 0 <= t_i < n_total
                ):
                    old = pheromone[c_i, d_i, mc_rec, t_i]
                    pheromone[c_i, d_i, mc_rec, t_i] = max(
                        np.float32(tau_min),
                        np.float32((1.0 - rho) * old + rho * delta),
                    )

        return [Move(container, source, destination) for container, source, destination in best_path]

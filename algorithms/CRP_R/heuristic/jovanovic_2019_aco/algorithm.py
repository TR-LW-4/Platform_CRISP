"""
JovanovicACO_rBRP
<2019> <heuristic> <restricted> <single-bay> <CRP-R>
Ant Colony System for restricted block relocation
n_iterations --- 5000 --- Colony iterations per layout
n_ants --- 10 --- Ants per iteration
q0 --- 0.9 --- Exploitation rate
rho --- 0.1 --- Global pheromone update rate
phi --- 0.9 --- Local evaporation factor
max_moves --- 10 --- Relocation-count depth in pheromone
max_const_iter --- 100 --- Stagnation reinitialisation threshold

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

import multiprocessing as mp
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig
from core.layout_trace import trace_layout
from core.move_export import export_yard_moves
from .scoring import compute_lb, dif, run_greedy_rbrp


class JovanovicACO_rBRP(BaseAlgorithm):

    name                = "Jovanović et al. (2019) ACO"
    category            = "Heuristic"
    description         = "Jovanović et al. (EJOR 2019) ant colony optimization heuristic."
    compatible_problems = ["CRP-R"]
    geometry            = "single-bay"
    objectives          = ["relocations"]
    fidelity            = "faithful"
    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)

    # ---------------------------------------------------------------- #
    # Training loop                                                      #
    # ---------------------------------------------------------------- #

    def train(
        self,
        problem_factory: Callable,
        result_queue:    mp.Queue,
        stop_event:      mp.Event,
    ) -> None:
        if stop_event.is_set():
            return

        cfg          = self.config
        n_iters      = int(cfg.extra.get("n_iterations",   5000))
        n_ants       = int(cfg.extra.get("n_ants",         10))
        q0           = float(cfg.extra.get("q0",            0.9))
        rho          = float(cfg.extra.get("rho",           0.1))
        phi          = float(cfg.extra.get("phi",           0.9))
        max_moves    = int(cfg.extra.get("max_moves",       10))
        max_const    = int(cfg.extra.get("max_const_iter",  100))
        report_every = max(1, n_iters // 50)

        trace_layout(
            f"JovanovicACO_rBRP.train: n_iters={n_iters}  n_ants={n_ants}  q0={q0}"
        )

        np.random.seed(cfg.seed)

        env = problem_factory()
        env.reset(options={"skip_auto_retrieve": True})

        n_total   = int(env.config.num_containers)
        max_tiers = int(env.config.max_tiers)
        n_stacks  = env.config.num_bays * env.config.num_rows

        all_keys:          List[Any] = []
        key_to_1based_idx: Dict[Any, int] = {}
        stacks_init:       Dict[Any, List[int]] = {}

        for a in range(n_stacks):
            key = env._action_to_stack(a)
            all_keys.append(key)
            key_to_1based_idx[key] = a + 1
            stk = env.yard.stacks.get(key)
            stacks_init[key] = (
                [int(c.priority) for c in stk.containers] if stk else []
            )

        # ── Greedy warm-start + initial LB ───────────────────────── #
        S_best, best_cost = run_greedy_rbrp(
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

        # ── Opt-1: precompute location array and stack-min array ──── #
        # loc_init[c]        = key of stack containing container c (1..N)
        # stack_min_init[key] = min priority in that stack (N+1 if empty)
        # empty_d[key]        = dd*(empty stack) for pheromone indexing
        loc_init: List[Any] = [None] * (n_total + 1)
        stack_min_init: Dict[Any, int] = {}
        empty_d: Dict[Any, int] = {}

        for key, prios in stacks_init.items():
            stack_min_init[key] = min(prios) if prios else (n_total + 1)
            empty_d[key] = n_total + key_to_1based_idx[key]
            for p in prios:
                loc_init[p] = key

        iters_no_improve = 0

        # ── Main ACO loop ─────────────────────────────────────────── #
        for iteration in range(n_iters):
            if stop_event.is_set():
                break

            iter_improved = False

            # ── Colony: each ant builds one solution ──────────────── #
            for _ant in range(n_ants):
                if stop_event.is_set():
                    break

                # Reset per-ant state (O(N+W) copies)
                stacks    = {k: list(v) for k, v in stacks_init.items()}
                M         = [0] * (n_total + 1)
                S: List[Tuple[int, int, int, int, Any]] = []
                valid     = True
                loc       = list(loc_init)                   # Opt-1: O(N)
                smin      = dict(stack_min_init)             # Opt-1: O(W)
                lb_curr   = lb_init                          # Opt-2

                for target in range(1, n_total + 1):
                    if not valid:
                        break

                    # Opt-1: O(1) source-stack lookup
                    src_key = loc[target]
                    if src_key is None:
                        continue

                    # ── Relocate all blockers above target ────────── #
                    while (
                        valid
                        and stacks[src_key]
                        and stacks[src_key][-1] != target
                    ):
                        c = stacks[src_key][-1]  # topmost blocker

                        # ── Build candidate list ─────────────────── #
                        best_g   = -1.0
                        best_dst = None
                        best_d   = 0
                        c_i      = c - 1
                        mc_cl    = min(M[c], max_moves)
                        t_i      = target - 1
                        cands_g: List[float] = []
                        cands_k: List[Any]   = []
                        cands_d: List[int]   = []

                        for dst_key in all_keys:
                            if dst_key == src_key:
                                continue
                            if len(stacks[dst_key]) >= max_tiers:
                                continue

                            # Opt-3: O(1) dd* via smin
                            sm = smin[dst_key]
                            d_val = sm if sm <= n_total else empty_d[dst_key]

                            # heuristic attractiveness
                            dif_val = (d_val - c) if d_val > c else (2 * n_total + 1 - d_val)
                            f_val   = 1.0 / (1.0 + dif_val)

                            tau = float(pheromone[c_i, d_val - 1, mc_cl, t_i])
                            g   = f_val * tau
                            cands_g.append(g)
                            cands_k.append(dst_key)
                            cands_d.append(d_val)

                            if g > best_g:
                                best_g   = g
                                best_dst = dst_key
                                best_d   = d_val

                        if not cands_k:
                            valid = False
                            break

                        # ── Transition rule (Eq. 31–32) ───────────── #
                        if np.random.uniform() < q0 or len(cands_k) == 1:
                            dst_key = best_dst
                            d_val   = best_d
                        else:
                            g_arr = np.asarray(cands_g, dtype=np.float64)
                            g_sum = g_arr.sum()
                            probs = (
                                g_arr / g_sum
                                if g_sum > 0.0
                                else np.ones(len(cands_k)) / len(cands_k)
                            )
                            idx     = np.random.choice(len(cands_k), p=probs)
                            dst_key = cands_k[idx]
                            d_val   = cands_d[idx]

                        # ── Opt-2: incremental LB update ─────────── #
                        # Was c non-well-located in src?
                        # c is on top → non-WL iff something BELOW has smaller due-date
                        # ↔ smin[src] < c  (smin is the stack-wide minimum)
                        was_non_wl   = smin[src_key] < c
                        # Will c be non-well-located in dst?
                        # c goes on top → non-WL iff dst has something with smaller value
                        # ↔ smin[dst] < c  (smin before the push)
                        will_non_wl  = smin[dst_key] < c

                        # Apply relocation
                        mc_rec = mc_cl
                        S.append((c, d_val, mc_rec, target, dst_key))

                        stacks[src_key].pop()
                        if smin[src_key] == c:
                            smin[src_key] = (
                                min(stacks[src_key])
                                if stacks[src_key]
                                else (n_total + 1)
                            )

                        stacks[dst_key].append(c)
                        if c < smin[dst_key]:
                            smin[dst_key] = c

                        loc[c]  = dst_key
                        M[c]   += 1

                        # Update LB
                        if was_non_wl:
                            lb_curr -= 1
                        if will_non_wl:
                            lb_curr += 1

                        # Early termination
                        if len(S) + lb_curr >= best_cost:
                            valid = False

                    # Retrieve target
                    if (
                        valid
                        and stacks[src_key]
                        and stacks[src_key][-1] == target
                    ):
                        stacks[src_key].pop()
                        # Update smin for src after retrieval
                        if smin[src_key] == target:
                            smin[src_key] = (
                                min(stacks[src_key])
                                if stacks[src_key]
                                else (n_total + 1)
                            )
                        loc[target] = None
                        # target was well-located (global minimum) → LB unchanged

                # ── Local pheromone update (Eq. 36) ──────────────── #
                for c, d_val, mc_rec, t, _dst_key in S:
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
                    best_cost     = len(S)
                    S_best        = S.copy()
                    iter_improved = True
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
            for c, d_val, mc_rec, t, _dst_key in S_best:
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

            # ── Periodic GUI push ─────────────────────────────────── #
            if (iteration + 1) % report_every == 0:
                self._push(
                    result_queue,
                    step     = iteration + 1,
                    metric   = float(best_cost),
                    metrics  = {
                        "relocations": float(best_cost),
                        "lower_bound": float(lb_init),
                        "iteration":   float(iteration + 1),
                    },
                    progress = (iteration + 1) / n_iters,
                )

        action_by_key = {key: action for action, key in enumerate(all_keys)}
        solution = [action_by_key[dst_key] for _, _, _, _, dst_key in S_best]
        metrics = env.validate_actions(solution)
        metrics["search_relocations"] = float(best_cost)
        metrics["progress"] = 1.0
        self._best_solution = solution[:]

        self._push(
            result_queue,
            step     = n_iters,
            metric   = float(metrics["relocations"]),
            metrics  = metrics,
            progress = 1.0,
            extra={
                "solution": solution[:],
                "moves": export_yard_moves(env.yard),
                "validation_errors": env.get_last_validation_errors(),
            },
        )

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution

    # ---------------------------------------------------------------- #
    # Configuration schema                                               #
    # ---------------------------------------------------------------- #

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update({
            "n_iterations": {
                "type": "int", "default": 5000, "min": 100, "max": 50000,
                "label": "ACO iterations",
                "help":  "Number of colony iterations per layout (paper default: 5 000).",
            },
            "n_ants": {
                "type": "int", "default": 10, "min": 1, "max": 100,
                "label": "Ants per iteration",
                "help":  "Colony size — number of ants generating solutions per iteration.",
            },
            "q0": {
                "type": "float", "default": 0.9, "min": 0.0, "max": 1.0,
                "label": "Exploitation rate q₀",
                "help": (
                    "Probability of greedy exploitation (pick argmax g(α)) vs. "
                    "probabilistic exploration. Paper default: 0.9."
                ),
            },
            "rho": {
                "type": "float", "default": 0.1, "min": 0.0, "max": 1.0,
                "label": "Global update rate ρ",
                "help":  "Pheromone deposit rate for the best solution. Paper default: 0.1.",
            },
            "phi": {
                "type": "float", "default": 0.9, "min": 0.0, "max": 1.0,
                "label": "Local evaporation φ",
                "help":  "Pheromone decay factor applied after each ant's solution. Paper default: 0.9.",
            },
            "max_moves": {
                "type": "int", "default": 10, "min": 1, "max": 30,
                "label": "Max relocations tracked (MaxMoves)",
                "help":  "Pheromone matrix depth for relocation count dimension. Paper default: 10.",
            },
            "max_const_iter": {
                "type": "int", "default": 100, "min": 10, "max": 1000,
                "label": "Stagnation threshold",
                "help": (
                    "Reinitialise pheromone matrix after this many consecutive "
                    "iterations without improvement. Paper default: 100."
                ),
            },
        })
        return base

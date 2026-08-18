"""
Jovanović, Tuba, Voß (2019) — Ant Colony Optimization for uBRP (CRP-U).

This implements the ACO-C variant (Section 4.2 Gre-C + Section 5 ACO):
  • Gre-C extended candidate list (Eqs. 9–23):
      T   — topmost blocker above the current target → any valid destination
      Or  — any non-well-located stack-top → destinations where it becomes WL
      Ow  — well-located stack-tops (look-ahead) when no Or moves exist
  • Extended heuristic dife (Eq. 21): penalises relocating well-located
    containers by N, offset by imp(c) (improvement in the source stack)
  • Same 4-D pheromone matrix τ[c][d][m_c][t] as the rBRP version
  • Same ACS update rules (local Eq. 36, global Eqs. 34–35) and stagnation
    reinitialisation

Performance optimisations
--------------------------
1. loc[c]   = stack_key        O(1) lookup of any container's stack
2. smin[key] = min priority    O(1) dd*(S) computation and dife evaluation,
               updated incrementally on every push/pop
3. Incremental lower-bound tracking:
     − 1 when a non-WL container leaves its stack
     + 1 when a container arrives at a stack where it is non-WL
   Eliminates the O(W×H) compute_lb() call per relocation.

Candidate-list cost
-------------------
Building Ĉ requires O(W²) work at each step (scan all stacks for Tn, then
scan all destinations for each Tn element).  This is larger than CRP-R (O(W))
but bounded in practice since W ≤ 12 for standard benchmarks.

Pheromone matrix size: N × (N+W) × (MaxMoves+1) × N  (float32)
  For N=99, W=10, MaxMoves=10: ≈ 11.8 M entries ≈ 47 MB

Reference
---------
R. Jovanović, M. Tuba, S. Voß,
"An efficient ant colony optimization algorithm for the blocks relocation
 problem",
European Journal of Operational Research 274 (2019) 78–90.
https://doi.org/10.1016/j.ejor.2018.09.038
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig
from core.layout_trace import trace_layout
from .scoring import compute_lb, dif, run_greedy_ubrp


class JovanovicACO_uBRP(BaseAlgorithm):

    name                = "Jovanović et al. (2019) ACO-C (uBRP)"
    category            = "Heuristic"
    description         = (
        "[single-bay origin]  "
        "Jovanović, Tuba, Voß (EJOR 2019) ACO-C for CRP-U (unrestricted BRP). "
        "4-D pheromone τ[c][d][m_c][t]; Gre-C extended candidate list allows "
        "proactive well-location of non-target-stack tops (Or moves) and "
        "look-ahead relocation of well-located containers (Ow moves). "
        "Outperforms EXP and FB on large instances; ~20–30× faster than FB."
    )
    compatible_problems = ["CRP-U"]
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
        cfg = self.config
        n_seeds = 1  # multi-seed eval removed; single run only
        n_iters      = int(cfg.extra.get("n_iterations",  1000))
        n_ants       = int(cfg.extra.get("n_ants",          10))
        q0           = float(cfg.extra.get("q0",            0.9))
        rho          = float(cfg.extra.get("rho",           0.1))
        phi          = float(cfg.extra.get("phi",           0.9))
        max_moves    = int(cfg.extra.get("max_moves",        10))
        max_const    = int(cfg.extra.get("max_const_iter",  100))
        report_every = max(1, n_iters // 50)

        all_metrics: List[Dict] = []

        for seed in range(n_seeds):
            if stop_event.is_set():
                break

            trace_layout(
                f"JovanovicACO_uBRP.train: seed {seed+1}/{n_seeds}  "
                f"n_iters={n_iters}  n_ants={n_ants}  q0={q0}"
            )

            np.random.seed(cfg.seed + seed * 1000)

            # ── Set up environment ────────────────────────────────────── #
            env = problem_factory()
            env.reset()

            n_total   = int(env.config.num_containers)
            max_tiers = int(env.config.max_tiers)
            n_stacks  = env.config.num_bays * env.config.num_rows

            all_keys:          List[Any]        = []
            key_to_1based_idx: Dict[Any, int]   = {}
            stacks_init:       Dict[Any, List[int]] = {}

            for a in range(n_stacks):
                key = env._action_to_stack(a)
                all_keys.append(key)
                key_to_1based_idx[key] = a + 1
                stk = env.yard.stacks.get(key)
                stacks_init[key] = (
                    [int(c.priority) for c in stk.containers] if stk else []
                )

            # ── Gre-C warm-start + initial LB ────────────────────────── #
            S_best, best_cost = run_greedy_ubrp(
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
                if stop_event.is_set():
                    break

                iter_improved = False

                # ── Colony: each ant builds one solution ──────────────── #
                for _ant in range(n_ants):
                    if stop_event.is_set():
                        break

                    # Reset per-ant state  (O(N + W) copies)
                    stacks: Dict[Any, List[int]] = {
                        k: list(v) for k, v in stacks_init.items()
                    }
                    M         = [0] * (n_total + 1)      # relocation counts
                    S: List[Tuple[int, int, int, int]] = []
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

                # ── Periodic GUI push ─────────────────────────────────── #
                if (iteration + 1) % report_every == 0:
                    frac = (seed * n_iters + iteration + 1) / (n_seeds * n_iters)
                    self._push(
                        result_queue,
                        step     = seed + 1,
                        metric   = float(best_cost),
                        metrics  = {
                            "relocations": float(best_cost),
                            "lower_bound": float(lb_init),
                            "iteration":   float(iteration + 1),
                        },
                        progress = min(frac, (seed + 1) / n_seeds),
                    )

            # ── End of seed ───────────────────────────────────────────── #
            metrics = {
                "relocations": float(best_cost),
                "steps":       float(best_cost),
                "time":        float(best_cost),
                "progress":    1.0,
            }
            all_metrics.append(metrics)

            if float(best_cost) < self._best_metric:
                self._best_metric   = float(best_cost)
                self._best_solution = []

            self._push(
                result_queue,
                step     = seed + 1,
                metric   = float(best_cost),
                metrics  = metrics,
                progress = (seed + 1) / n_seeds,
                snapshot = env.get_state_snapshot(),
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
    # Configuration schema                                               #
    # ---------------------------------------------------------------- #

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update({
            "n_iterations": {
                "type": "int", "default": 1000, "min": 100, "max": 50000,
                "label": "ACO iterations",
                "help": "Number of colony iterations per layout (paper default for uBRP: 1 000).",
            },
            "n_ants": {
                "type": "int", "default": 10, "min": 1, "max": 100,
                "label": "Ants per iteration",
                "help": "Colony size — number of ants generating solutions per iteration.",
            },
            "q0": {
                "type": "float", "default": 0.9, "min": 0.0, "max": 1.0,
                "label": "Exploitation rate q₀",
                "help": (
                    "Probability of greedy exploitation (argmax g(α)) vs. "
                    "probabilistic exploration. Paper default: 0.9."
                ),
            },
            "rho": {
                "type": "float", "default": 0.1, "min": 0.0, "max": 1.0,
                "label": "Global update rate ρ",
                "help": "Pheromone deposit rate for the best solution. Paper default: 0.1.",
            },
            "phi": {
                "type": "float", "default": 0.9, "min": 0.0, "max": 1.0,
                "label": "Local evaporation φ",
                "help": "Pheromone decay factor after each ant's solution. Paper default: 0.9.",
            },
            "max_moves": {
                "type": "int", "default": 10, "min": 1, "max": 30,
                "label": "Max relocations tracked (MaxMoves)",
                "help": "Pheromone matrix depth for relocation-count dimension. Paper default: 10.",
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

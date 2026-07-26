"""
Jovanović, Tuba, Voß (2019) — Ant Colony Optimization for rBRP-Time (CRP-Time).

This is the Section 6 extension of the paper: the same rBRP-ACO algorithm
(ACS, 4-D pheromone matrix, MinMax heuristic) with the objective function
replaced by total crane working time instead of number of relocations.

Differences from the CRP-R version
------------------------------------
1. best_cost  : float (seconds)  instead of int (relocations)
2. lb_init    : n_nwl × spreader_s  instead of n_nwl count
3. val(S)     : 1 / (crane_time(S) − lb_init + 1)
4. Per-ant    : crane_pos + time_so_far tracked alongside stacks/loc/smin
5. Each move  : accumulates move_time_inline() / retrieval_time_inline()
6. Early stop : time_so_far + lb_curr×spreader_s ≥ best_cost
7. Metrics    : crane_time (primary), relocations (secondary)

Everything else (pheromone structure, transition rule, local/global update,
stagnation reinitialisation, performance optimisations) is identical to
the CRP-R version.

Kinematics parameters are read from ProblemConfig.extra (same keys used by
the CRP-Time problem's config_schema):
  gantry_s_per_bay   default 3.5 s   (Lee & Lee 2010)
  trolley_s_per_row  default 1.2 s
  gantry_accel_s     default 40.0 s
  spreader_s         default 30.0 s

Pheromone matrix size: N × (N+W) × (MaxMoves+1) × N  (float32)
  For N=39, W=8,  MaxMoves=10: ≈ 793 K entries ≈ 3 MB
  For N=100,W=10, MaxMoves=10: ≈ 12 M entries ≈ 48 MB

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
from .scoring import compute_lb, compute_lb_time, move_time_inline, retrieval_time_inline, run_greedy_rbrp_time


class JovanovicACO_CRPTime(BaseAlgorithm):

    name                = "Jovanović et al. (2019) ACO (CRP-Time)"
    category            = "Heuristic"
    description         = (
        "[single-bay origin]  "
        "Jovanović, Tuba, Voß (EJOR 2019) rBRP-ACO adapted for CRP-Time "
        "(Section 6). Crane-time objective via platform 2-D KinematicsModel "
        "(bay × row, gantry + trolley). Same 4-D pheromone τ[c][d][m_c][t] "
        "and MinMax heuristic as the CRP-R version; val(S) and early "
        "termination use seconds instead of relocation counts."
    )
    compatible_problems = ["CRP-Time"]
    step_label          = "Seed"

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
        n_seeds      = max(1, cfg.num_eval_seeds)
        n_iters      = int(cfg.extra.get("n_iterations",   5000))
        n_ants       = int(cfg.extra.get("n_ants",         10))
        q0           = float(cfg.extra.get("q0",            0.9))
        rho          = float(cfg.extra.get("rho",           0.1))
        phi          = float(cfg.extra.get("phi",           0.9))
        max_moves    = int(cfg.extra.get("max_moves",       10))
        max_const    = int(cfg.extra.get("max_const_iter",  100))
        report_every = max(1, n_iters // 50)

        all_metrics: List[Dict] = []

        for seed in range(n_seeds):
            if stop_event.is_set():
                break

            trace_layout(
                f"JovanovicACO_CRPTime.train: seed {seed+1}/{n_seeds}  "
                f"n_iters={n_iters}  n_ants={n_ants}  q0={q0}"
            )

            np.random.seed(cfg.seed + seed * 1000)

            # ── Set up environment ────────────────────────────────────── #
            env = problem_factory()
            env.config.seed = seed
            env.reset(options={"skip_auto_retrieve": True})

            n_total   = int(env.config.num_containers)
            max_tiers = int(env.config.max_tiers)
            n_stacks  = env.config.num_bays * env.config.num_rows

            # Read kinematics from problem config (same keys as CRP-Time schema)
            extra      = env.config.extra
            gantry_s   = float(extra.get("gantry_s_per_bay",  3.5))
            trolley_s  = float(extra.get("trolley_s_per_row", 1.2))
            accel_s    = float(extra.get("gantry_accel_s",   40.0))
            spreader_s = float(extra.get("spreader_s",        30.0))

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

            # ── Greedy warm-start (returns crane time) ────────────────── #
            S_best, best_cost = run_greedy_rbrp_time(
                stacks_init, all_keys, n_total, max_tiers, key_to_1based_idx,
                gantry_s, trolley_s, accel_s, spreader_s,
            )
            # best_cost is float (seconds) from this point forward

            # ── Initial lower bound (crane-time) ─────────────────────── #
            lb_init_nwl  = compute_lb(stacks_init)          # integer NWL count
            lb_init_time = compute_lb_time(lb_init_nwl, spreader_s)  # float seconds

            # ── Pheromone initialisation (Eq. 37, crane-time val) ────── #
            val_greedy = 1.0 / max(best_cost - lb_init_time + 1.0, 1.0)
            tau_0      = val_greedy / n_stacks
            tau_min    = tau_0 / n_stacks

            pheromone = np.full(
                (n_total, n_total + n_stacks, max_moves + 1, n_total),
                tau_0, dtype=np.float32,
            )

            # ── Opt-1: precompute location + stack-min arrays ─────────── #
            loc_init:       List[Any]      = [None] * (n_total + 1)
            stack_min_init: Dict[Any, int] = {}
            empty_d:        Dict[Any, int] = {}

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

                    # Reset per-ant state
                    stacks       = {k: list(v) for k, v in stacks_init.items()}
                    M            = [0] * (n_total + 1)
                    S: List[Tuple[int, int, int, int]] = []
                    valid        = True
                    loc          = list(loc_init)
                    smin         = dict(stack_min_init)
                    lb_curr      = lb_init_nwl      # incremental NWL count
                    crane_pos    = (1, 1)            # crane starts at (bay=1, row=1)
                    time_so_far  = 0.0               # accumulated crane time

                    for target in range(1, n_total + 1):
                        if not valid:
                            break

                        src_key = loc[target]
                        if src_key is None:
                            continue

                        # ── Relocate all blockers above target ────────── #
                        while (
                            valid
                            and stacks[src_key]
                            and stacks[src_key][-1] != target
                        ):
                            c = stacks[src_key][-1]

                            # Build candidate list
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

                                # O(1) dd* via smin
                                sm    = smin[dst_key]
                                d_val = sm if sm <= n_total else empty_d[dst_key]

                                # Heuristic attractiveness (unchanged from CRP-R)
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

                            # Transition rule (Eq. 31–32)
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

                            # Incremental NWL tracking (Opt-2, same as CRP-R)
                            was_non_wl  = smin[src_key] < c
                            will_non_wl = smin[dst_key] < c

                            # Record 4-tuple and apply relocation
                            mc_rec = mc_cl
                            S.append((c, d_val, mc_rec, target))

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

                            # Accumulate crane time for this relocation
                            # src_key and dst_key are (bay, row) tuples = positions
                            cost, crane_pos = move_time_inline(
                                crane_pos, src_key, dst_key,
                                gantry_s, trolley_s, accel_s, spreader_s,
                            )
                            time_so_far += cost

                            # Update NWL count
                            if was_non_wl:
                                lb_curr -= 1
                            if will_non_wl:
                                lb_curr += 1

                            # Early termination (crane-time)
                            lb_remaining = lb_curr * spreader_s
                            if time_so_far + lb_remaining >= best_cost:
                                valid = False

                        # Retrieve target (also incurs crane time)
                        if (
                            valid
                            and stacks[src_key]
                            and stacks[src_key][-1] == target
                        ):
                            cost, crane_pos = retrieval_time_inline(
                                crane_pos, src_key,
                                gantry_s, trolley_s, accel_s, spreader_s,
                            )
                            time_so_far += cost

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

                    # ── Update best (crane time) ───────────────────────── #
                    if valid and time_so_far < best_cost:
                        best_cost        = time_so_far
                        S_best           = S.copy()
                        iter_improved    = True
                        iters_no_improve = 0

                # ── Stagnation reinitialisation ───────────────────────── #
                if not iter_improved:
                    iters_no_improve += 1
                if iters_no_improve >= max_const:
                    val_b   = 1.0 / max(best_cost - lb_init_time + 1.0, 1.0)
                    tau_new = val_b / n_stacks
                    tau_min = tau_new / n_stacks
                    pheromone[:] = np.float32(tau_new)
                    iters_no_improve = 0

                # ── Global pheromone update (Eqs. 34–35) ─────────────── #
                val_b   = 1.0 / max(best_cost - lb_init_time + 1.0, 1.0)
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
                            "crane_time":  float(best_cost),
                            "time":        float(best_cost),
                            "relocations": float(len(S_best)),
                            "lower_bound": float(lb_init_time),
                            "iteration":   float(iteration + 1),
                        },
                        progress = min(frac, (seed + 1) / n_seeds),
                    )

            # ── End of seed ───────────────────────────────────────────── #
            metrics = {
                "crane_time":  float(best_cost),
                "time":        float(best_cost),
                "relocations": float(len(S_best)),
                "steps":       float(len(S_best)),
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
            "num_eval_seeds": {
                "type": "int", "default": 1, "min": 1, "max": 50,
                "label": "Evaluation seeds",
                "help": (
                    "Number of independent ACO runs per layout. "
                    "Use 1 for fixed benchmark layouts (Caserta/Zhu). "
                    "Use >1 for random-layout mode."
                ),
            },
            "n_iterations": {
                "type": "int", "default": 5000, "min": 100, "max": 50000,
                "label": "ACO iterations",
                "help": (
                    "Colony iterations per layout. Paper uses 5 000 for rBRP; "
                    "crane-time convergence is slower — consider 10 000+ for "
                    "best quality (see paper Section 7.4)."
                ),
            },
            "n_ants": {
                "type": "int", "default": 10, "min": 1, "max": 100,
                "label": "Ants per iteration",
                "help": "Colony size — number of ants per iteration. Paper default: 10.",
            },
            "q0": {
                "type": "float", "default": 0.9, "min": 0.0, "max": 1.0,
                "label": "Exploitation rate q₀",
                "help": (
                    "Probability of greedy exploitation vs. probabilistic "
                    "exploration. Paper default: 0.9."
                ),
            },
            "rho": {
                "type": "float", "default": 0.1, "min": 0.0, "max": 1.0,
                "label": "Global update rate ρ",
                "help": "Pheromone deposit rate for best solution. Paper default: 0.1.",
            },
            "phi": {
                "type": "float", "default": 0.9, "min": 0.0, "max": 1.0,
                "label": "Local evaporation φ",
                "help": "Pheromone decay after each ant solution. Paper default: 0.9.",
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
                    "Reinitialise pheromone after this many iterations without "
                    "improvement. Paper default: 100."
                ),
            },
        })
        return base

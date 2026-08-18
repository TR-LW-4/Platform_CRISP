"""
Tus, Rendl, Raidl (2014): Metaheuristics for the Two-Dimensional
Container Pre-Marshalling Problem (2D-PMP).

Reference
---------
A. Tus, A. Rendl, G.R. Raidl,
"Metaheuristics for the Two-Dimensional Container Pre-Marshalling Problem".

Paper scope and this module's structure
---------------------------------------
The paper studies one problem family (2D-PMP) and compares three methods:
  1) 2D-LPFH construction heuristic
  2) Pilot method using 2D-LPFH-style compound moves
  3) MMAS (Max-Min Ant System) with move-based pheromone model

To keep a one-paper-one-folder layout, this module exposes three independent
algorithm classes in the same file:
  - TusRendlRaidl2DLPFH
  - TusRendlRaidl2DPilot
  - TusRendlRaidl2DMMAS

All classes are registered under CRP-Prem as requested by the project setup.
The implementation is self-contained and does not import logic from other
algorithm folders.
"""

from __future__ import annotations

import multiprocessing as mp
import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from core.base_algorithm import AlgorithmConfig, BaseAlgorithm

Stacks = Dict[int, List[int]]
Move = Tuple[int, int]


def _clone(stacks: Stacks) -> Stacks:
    return {k: list(v) for k, v in stacks.items()}


def _pos(stacks: Stacks, p: int) -> Optional[Tuple[int, int]]:
    for s, arr in stacks.items():
        for t, x in enumerate(arr):
            if x == p:
                return s, t
    return None


def _v_blocked(stacks: Stacks, p: int) -> bool:
    z = _pos(stacks, p)
    if z is None:
        return False
    s, t = z
    return any(stacks[s][u] > p for u in range(t + 1, len(stacks[s])))


def _h_blocked(stacks: Stacks, p: int, n_stacks: int) -> bool:
    z = _pos(stacks, p)
    if z is None:
        return False
    s, _ = z
    left = any(any(v > p for v in stacks[i]) for i in range(0, s))
    right = any(any(v > p for v in stacks[i]) for i in range(s + 1, n_stacks))
    return left and right


def _eval(stacks: Stacks, n_stacks: int, n_containers: int) -> Tuple[int, int, int, float]:
    bv = 0
    bh = 0
    bt = 0
    for p in range(1, n_containers + 1):
        z = _pos(stacks, p)
        if z is None:
            continue
        if _v_blocked(stacks, p):
            bv += 1
            s, t = z
            bt += max(0, len(stacks[s]) - (t + 1))
        if _h_blocked(stacks, p, n_stacks):
            bh += 1
    return bv, bh, bt, float(bv + 0.5 * bh + bt)


def _move(stacks: Stacks, src: int, dst: int, max_tiers: int) -> bool:
    if src == dst or src not in stacks or dst not in stacks:
        return False
    if not stacks[src] or len(stacks[dst]) >= max_tiers:
        return False
    x = stacks[src].pop()
    stacks[dst].append(x)
    return True


def _encode_action(src_idx: int, dst_idx: int, n_stacks: int) -> int:
    dst_flat = dst_idx if dst_idx < src_idx else dst_idx - 1
    return src_idx * (n_stacks - 1) + dst_flat


def _non_located(stacks: Stacks, n_stacks: int, n_containers: int) -> List[int]:
    return [p for p in range(1, n_containers + 1) if _v_blocked(stacks, p) or _h_blocked(stacks, p, n_stacks)]


def _preferred_stacks(p: int, n_stacks: int, n_containers: int) -> List[int]:
    cps = max(1.0, float(n_containers) / float(max(1, n_stacks)))
    s1 = int(p // (2.0 * cps))
    s1 = max(0, min(n_stacks - 1, s1))
    s2 = (n_stacks - 1) - s1
    out = {s1, s2}
    if s1 > 0:
        out.add(s1 - 1)
    if s2 < n_stacks - 1:
        out.add(s2 + 1)
    return sorted(x for x in out if 0 <= x < n_stacks)


def solve_2d_lpfh(
    stacks_init: Stacks,
    max_tiers: int,
    beta2: int,
    beta3: int,
    seed: int,
    max_iterations: int = 2000,
) -> Tuple[List[Move], Stacks]:
    rng = random.Random(seed)
    stacks = _clone(stacks_init)
    n_stacks = len(stacks)
    n_containers = sum(len(v) for v in stacks.values())
    seq: List[Move] = []

    for _ in range(max_iterations):
        bad = _non_located(stacks, n_stacks, n_containers)
        if not bad:
            break
        c = max(bad)
        z = _pos(stacks, c)
        if z is None:
            break
        src, tier = z
        Rc = [s for s in _preferred_stacks(c, n_stacks, n_containers) if s != src and len(stacks[s]) < max_tiers]
        if not Rc:
            Rc = [s for s in range(n_stacks) if s != src and len(stacks[s]) < max_tiers]
        if not Rc:
            break
        dst = Rc[rng.randrange(max(1, min(beta2, len(Rc))))]

        blockers = len(stacks[src]) - (tier + 1)
        feasible = True
        for _b in range(blockers):
            temps = [s for s in range(n_stacks) if s != src and s != dst and len(stacks[s]) < max_tiers]
            if not temps:
                feasible = False
                break
            tmp = temps[rng.randrange(max(1, min(beta3, len(temps))))]
            if not _move(stacks, src, tmp, max_tiers):
                feasible = False
                break
            seq.append((src, tmp))
        if not feasible:
            break
        if _move(stacks, src, dst, max_tiers):
            seq.append((src, dst))
        else:
            break
    return seq, stacks


def _best_compound_moves(stacks: Stacks, max_tiers: int, top_k: int = 8) -> List[List[Move]]:
    n_stacks = len(stacks)
    n_containers = sum(len(v) for v in stacks.values())
    bad = _non_located(stacks, n_stacks, n_containers)
    if not bad:
        return []
    c = max(bad)
    z = _pos(stacks, c)
    if z is None:
        return []
    src, tier = z
    blockers = len(stacks[src]) - (tier + 1)

    out: List[Tuple[float, List[Move]]] = []
    for dst in range(n_stacks):
        if dst == src or len(stacks[dst]) >= max_tiers:
            continue
        work = _clone(stacks)
        seq: List[Move] = []
        ok = True
        for _ in range(blockers):
            temps = [s for s in range(n_stacks) if s != src and s != dst and len(work[s]) < max_tiers]
            if not temps:
                ok = False
                break
            if not _move(work, src, temps[0], max_tiers):
                ok = False
                break
            seq.append((src, temps[0]))
        if ok and _move(work, src, dst, max_tiers):
            seq.append((src, dst))
            sc = _eval(work, n_stacks, n_containers)[3] + 0.01 * len(seq)
            out.append((sc, seq))
    out.sort(key=lambda x: x[0])
    return [x[1] for x in out[:max(1, top_k)]]


def _apply_seq(stacks: Stacks, seq: List[Move], max_tiers: int) -> Stacks:
    out = _clone(stacks)
    for src, dst in seq:
        _move(out, src, dst, max_tiers)
    return out


def _greedy_rollout(stacks: Stacks, max_tiers: int, k: int) -> List[Move]:
    cur = _clone(stacks)
    out: List[Move] = []
    for _ in range(k):
        cands = _best_compound_moves(cur, max_tiers=max_tiers, top_k=1)
        if not cands:
            break
        out.extend(cands[0])
        cur = _apply_seq(cur, cands[0], max_tiers)
    return out


def solve_pilot(
    stacks_init: Stacks,
    max_tiers: int,
    lookahead_k: int,
    max_master_steps: int,
    seed: int,
) -> Tuple[List[Move], Stacks]:
    random.seed(seed)
    cur = _clone(stacks_init)
    n_stacks = len(cur)
    n_containers = sum(len(v) for v in cur.values())
    out: List[Move] = []
    for _ in range(max_master_steps):
        if _eval(cur, n_stacks, n_containers)[3] <= 0.0:
            break
        roots = _best_compound_moves(cur, max_tiers=max_tiers, top_k=8)
        if not roots:
            break
        best = None
        best_score = float("inf")
        for r in roots:
            st = _apply_seq(cur, r, max_tiers)
            rr = _greedy_rollout(st, max_tiers, max(0, lookahead_k - 1))
            end = _apply_seq(st, rr, max_tiers)
            sc = _eval(end, n_stacks, n_containers)[3] + 0.01 * (len(r) + len(rr))
            if sc < best_score:
                best_score = sc
                best = r
        if not best:
            break
        out.extend(best)
        cur = _apply_seq(cur, best, max_tiers)
    return out, cur


def solve_mmas(
    stacks_init: Stacks,
    max_tiers: int,
    n_ants: int,
    alpha: float,
    beta: float,
    evaporation: float,
    max_iters: int,
    max_steps_per_ant: int,
    seed: int,
) -> Tuple[List[Move], Stacks]:
    rng = random.Random(seed)
    cur0 = _clone(stacks_init)
    n_stacks = len(cur0)
    n_containers = sum(len(v) for v in cur0.values())
    tau: Dict[Tuple[Tuple[Tuple[int, ...], ...], Tuple[Move, ...]], float] = {}
    tau_min, tau_max = 0.01, 5.0

    def _sig(st: Stacks) -> Tuple[Tuple[int, ...], ...]:
        return tuple(tuple(st[s]) for s in sorted(st))

    best_seq: List[Move] = []
    best_sc = float("inf")
    for _ in range(max_iters):
        iter_best: Optional[List[Move]] = None
        iter_best_sc = float("inf")
        for _a in range(max(1, n_ants)):
            st = _clone(cur0)
            seq: List[Move] = []
            for _s in range(max_steps_per_ant):
                if _eval(st, n_stacks, n_containers)[3] <= 0.0:
                    break
                cands = _best_compound_moves(st, max_tiers=max_tiers, top_k=8)
                if not cands:
                    break
                s_sig = _sig(st)
                probs = []
                denom = 0.0
                for c in cands:
                    key = (s_sig, tuple(c))
                    t = tau.get(key, tau_max)
                    nxt = _apply_seq(st, c, max_tiers)
                    eta = 1.0 / (1e-6 + _eval(nxt, n_stacks, n_containers)[3])
                    v = (t ** alpha) * (eta ** beta)
                    probs.append((c, v, key))
                    denom += v
                if denom <= 0.0:
                    chosen, chosen_key = cands[0], (s_sig, tuple(cands[0]))
                else:
                    r = rng.random() * denom
                    acc = 0.0
                    chosen, chosen_key = cands[0], (s_sig, tuple(cands[0]))
                    for c, v, k in probs:
                        acc += v
                        if acc >= r:
                            chosen, chosen_key = c, k
                            break
                st = _apply_seq(st, chosen, max_tiers)
                seq.extend(chosen)
                tau[chosen_key] = min(tau_max, max(tau_min, tau.get(chosen_key, tau_max)))
            sc = _eval(_apply_seq(cur0, seq, max_tiers), n_stacks, n_containers)[3] + 0.01 * len(seq)
            if sc < iter_best_sc:
                iter_best_sc = sc
                iter_best = list(seq)
        if iter_best is None:
            break
        if iter_best_sc < best_sc:
            best_sc = iter_best_sc
            best_seq = iter_best
        for k in list(tau.keys()):
            tau[k] = max(tau_min, tau[k] * (1.0 - evaporation))
    return best_seq, _apply_seq(cur0, best_seq, max_tiers)


class _Base2D(BaseAlgorithm):
    compatible_problems = ["CRP-Prem"]
    def _build_stacks(self, env) -> Tuple[Stacks, int]:
        keys = list(env.yard.stacks.keys())
        key_to_idx = {k: i for i, k in enumerate(keys)}
        stacks0 = {key_to_idx[k]: list(st.priority_snapshot()) for k, st in env.yard.stacks.items()}
        return stacks0, len(keys)

    def get_best_solution(self) -> Optional[List[int]]:
        return self._best_solution


class TusRendlRaidl2DLPFH(_Base2D):
    name = "Tus–Rendl–Raidl 2D-LPFH"
    category = "Heuristic"
    description = "[2D-PMP paper baseline] 2D-LPFH construction heuristic."

    def train(self, problem_factory: Callable, result_queue: mp.Queue, stop_event: mp.Event) -> None:
        cfg = self.config
        n_seeds = 1  # multi-seed eval removed; single run only
        beta2 = int(cfg.extra.get("beta2", 2))
        beta3 = int(cfg.extra.get("beta3", 2))
        max_iterations = int(cfg.extra.get("max_iterations", 2000))
        all_metrics: List[Dict[str, float]] = []
        for seed in range(n_seeds):
            if stop_event.is_set():
                break
            env = problem_factory()
            env.reset()
            stacks0, n_stacks = self._build_stacks(env)
            seq, final = solve_2d_lpfh(stacks0, int(env.config.max_tiers), beta2, beta3, int(cfg.seed) + seed, max_iterations)
            bv, bh, bt, f = _eval(final, n_stacks, int(env.config.num_containers))
            metric = float(f + 0.01 * len(seq))
            actions = [_encode_action(s, d, n_stacks) for (s, d) in seq]
            if metric < self._best_metric:
                self._best_metric = metric
                self._best_solution = actions
            m = {"objective": f, "moves": float(len(seq)), "vertical_blocked": float(bv), "horizontal_blocked": float(bh), "blocking_above": float(bt), "bad_overlaps": float(bv + bh), "time": float(len(seq)), "solved": 1.0 if f <= 0.0 else 0.0}
            all_metrics.append(m)
            self._push(result_queue, step=seed + 1, metric=metric, metrics=m, progress=(seed + 1) / n_seeds, snapshot=env.get_state_snapshot())
        if all_metrics:
            agg = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}
            self._push(result_queue, step=n_seeds, metric=self._best_metric, metrics=agg, progress=1.0)

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update({
            "beta2": {"type": "int", "default": 2, "min": 1, "max": 20, "label": "RCL size for destination stack"},
            "beta3": {"type": "int", "default": 2, "min": 1, "max": 20, "label": "RCL size for temporary stack"},
            "max_iterations": {"type": "int", "default": 2000, "min": 10, "max": 50000, "label": "Max LPFH iterations"},
        })
        return base


class TusRendlRaidl2DPilot(_Base2D):
    name = "Tus–Rendl–Raidl 2D-Pilot"
    category = "Heuristic"
    description = "[2D-PMP paper metaheuristic] Pilot method over compound moves."

    def train(self, problem_factory: Callable, result_queue: mp.Queue, stop_event: mp.Event) -> None:
        cfg = self.config
        n_seeds = 1  # multi-seed eval removed; single run only
        lookahead_k = int(cfg.extra.get("lookahead_k", 7))
        max_master_steps = int(cfg.extra.get("max_master_steps", 500))
        all_metrics: List[Dict[str, float]] = []
        for seed in range(n_seeds):
            if stop_event.is_set():
                break
            env = problem_factory()
            env.reset()
            stacks0, n_stacks = self._build_stacks(env)
            seq, final = solve_pilot(stacks0, int(env.config.max_tiers), lookahead_k, max_master_steps, int(cfg.seed) + seed)
            f = _eval(final, n_stacks, int(env.config.num_containers))[3]
            metric = float(f + 0.01 * len(seq))
            actions = [_encode_action(s, d, n_stacks) for (s, d) in seq]
            if metric < self._best_metric:
                self._best_metric = metric
                self._best_solution = actions
            m = {"objective": float(f), "moves": float(len(seq)), "bad_overlaps": float(f), "time": float(len(seq)), "solved": 1.0 if f <= 0.0 else 0.0}
            all_metrics.append(m)
            self._push(result_queue, step=seed + 1, metric=metric, metrics=m, progress=(seed + 1) / n_seeds, snapshot=env.get_state_snapshot())
        if all_metrics:
            agg = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}
            self._push(result_queue, step=n_seeds, metric=self._best_metric, metrics=agg, progress=1.0)

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update({
            "lookahead_k": {"type": "int", "default": 7, "min": 1, "max": 50, "label": "Pilot lookahead depth"},
            "max_master_steps": {"type": "int", "default": 500, "min": 10, "max": 20000, "label": "Master construction steps"},
        })
        return base


class TusRendlRaidl2DMMAS(_Base2D):
    name = "Tus–Rendl–Raidl 2D-MMAS"
    category = "Heuristic"
    description = "[2D-PMP paper main metaheuristic] MMAS with move-based pheromone."

    def train(self, problem_factory: Callable, result_queue: mp.Queue, stop_event: mp.Event) -> None:
        cfg = self.config
        n_seeds = 1  # multi-seed eval removed; single run only
        n_ants = int(cfg.extra.get("n_ants", 8))
        alpha = float(cfg.extra.get("alpha", 1.0))
        beta = float(cfg.extra.get("beta", 2.0))
        evaporation = float(cfg.extra.get("evaporation", 0.02))
        max_iters = int(cfg.extra.get("max_iters", 200))
        max_steps_per_ant = int(cfg.extra.get("max_steps_per_ant", 250))
        all_metrics: List[Dict[str, float]] = []
        for seed in range(n_seeds):
            if stop_event.is_set():
                break
            env = problem_factory()
            env.reset()
            stacks0, n_stacks = self._build_stacks(env)
            seq, final = solve_mmas(
                stacks0,
                int(env.config.max_tiers),
                n_ants,
                alpha,
                beta,
                evaporation,
                max_iters,
                max_steps_per_ant,
                int(cfg.seed) + seed,
            )
            f = _eval(final, n_stacks, int(env.config.num_containers))[3]
            metric = float(f + 0.01 * len(seq))
            actions = [_encode_action(s, d, n_stacks) for (s, d) in seq]
            if metric < self._best_metric:
                self._best_metric = metric
                self._best_solution = actions
            m = {"objective": float(f), "moves": float(len(seq)), "bad_overlaps": float(f), "time": float(len(seq)), "solved": 1.0 if f <= 0.0 else 0.0}
            all_metrics.append(m)
            self._push(result_queue, step=seed + 1, metric=metric, metrics=m, progress=(seed + 1) / n_seeds, snapshot=env.get_state_snapshot())
        if all_metrics:
            agg = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}
            self._push(result_queue, step=n_seeds, metric=self._best_metric, metrics=agg, progress=1.0)

    @classmethod
    def config_schema(cls) -> Dict:
        base = super().config_schema()
        base.update({
            "n_ants": {"type": "int", "default": 8, "min": 1, "max": 128, "label": "Ant count"},
            "alpha": {"type": "float", "default": 1.0, "min": 0.0, "max": 10.0, "label": "Pheromone weight"},
            "beta": {"type": "float", "default": 2.0, "min": 0.0, "max": 10.0, "label": "Heuristic weight"},
            "evaporation": {"type": "float", "default": 0.02, "min": 0.0, "max": 1.0, "label": "Evaporation rate"},
            "max_iters": {"type": "int", "default": 200, "min": 1, "max": 5000, "label": "MMAS iterations"},
            "max_steps_per_ant": {"type": "int", "default": 250, "min": 10, "max": 10000, "label": "Max moves per ant"},
        })
        return base


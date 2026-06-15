"""
CRP-Stow — Blocks Relocation Problem with Stowage Plan (BRLP)

Problem (Jovanović et al. 2019; Ji et al. 2015)
------------------------------------------------
- N containers occupy a 1-D yard bay: YS stacks each at most YT tiers high.
- A vessel bay has VS stacks; vessel stack s has a maximum tier H_v[s].
- Every container c has a designated vessel position:
      vs(c)  – vessel stack index  (stored as c.group,    0-based, A=0 B=1 …)
      vt(c)  – vessel tier index   (stored as c.priority, 0-based, bottom = 0)
- Container c is RETRIEVABLE (cdd(c) == 0) iff its vessel tier equals the
  number of containers already loaded into vessel stack vs(c).  In other words,
  the containers within each vessel stack must be loaded bottom-up by tier.
- Only the TOP container of each yard stack can be accessed.
- When the target is buried, every blocker above it must be relocated to
  another yard stack first — each relocation costs +1.
- Once retrieved to the vessel a container cannot be moved again.
- Objective: minimise total yard relocations.

Two-phase Gym interface
-----------------------
Action space : Discrete(YS) — flat yard stack index  0 … YS-1.

Phase "high"  (H_L — retrieval selection, Jovanović 2019 §5.2)
    Agent selects a YARD STACK that contains at least one retrievable container.
    The env automatically targets the HIGHEST (topmost) retrievable container
    in that stack (fewest blockers = easiest to expose).
    · If the target is already on top → auto-retrieve, stay "high".
    · Otherwise → switch to "low" (record target + source stack).

Phase "low"  (H_R — relocation, Jovanović 2019 §5.1)
    Agent selects a DESTINATION YARD STACK for the TOP blocker of the source stack.
    · Blocker is relocated; relocations += 1.
    · If the new top of source == target → auto-retrieve, return to "high".
    · Otherwise stay "low" (same source + target).

Reward   : −1 per relocation, 0 otherwise; terminal when all containers loaded.
Metrics  : relocations (primary, matches Jovanović paper), steps, vessel_utilisation.

Benchmark support
-----------------
Set extra["layout_file_path"] = "/path/to/Bay-A-VS-YS-YT_seed.pro"
to load a Jovanović / Ji benchmark .pro instance.  The problem config is
updated in-place from the file (YS, YT, VS, H_v).

For RL training without a .pro file set num_bays=YS, max_tiers=YT,
num_groups=VS (num_rows is always forced to 1).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym

from core.base_problem import BaseProblem, ProblemConfig
from core.container import Container, ContainerSize, ContainerType
from core.yard import Yard
from core.benchmark_keys import layout_path_from_extra


# ================================================================ #
#  .pro file parser                                                  #
# ================================================================ #

def parse_pro_file(path: "str | Path") -> Dict:
    """
    Parse a Jovanović BRLP benchmark .pro file.

    Returns a dict:
        N      – int             : total number of containers
        YS     – int             : number of yard stacks
        YT     – int             : maximum allowed tier in yard stacks
        VS     – int             : number of vessel stacks
        H_v    – List[int]       : max tier of each vessel stack (length VS)
        stacks – List[List[Tuple[int,int]]] : per yard stack, containers
                 in bottom→top order as (vs:int, vt:int) pairs

    Container token "A_2" → vs = ord('A')-ord('A') = 0, vt = 2.
    Each (vs, vt) pair uniquely identifies a container (valid BRLP instance).
    """
    lines = [l.strip() for l in Path(path).read_text().splitlines()]

    nums: List[int] = []
    in_yard        = False
    stacks_raw: List[str] = []
    prev_was_header = False   # True after reading a "Stack N:" line

    for line in lines:
        if not line:
            continue
        if "Yard Bay" in line:
            in_yard = True
            continue
        if in_yard:
            if re.match(r'^Stack\s*\d+\s*:', line, re.IGNORECASE):
                if prev_was_header:
                    stacks_raw.append("")   # previous stack was empty
                prev_was_header = True
            elif prev_was_header:
                stacks_raw.append(line)
                prev_was_header = False
        else:
            if re.match(r'^\d+$', line):
                nums.append(int(line))

    if prev_was_header:
        stacks_raw.append("")   # last stack empty

    if len(nums) < 4:
        raise ValueError(
            f"parse_pro_file({path}): expected ≥4 numeric values, got {nums}"
        )

    N, YS, YT, VS = nums[0], nums[1], nums[2], nums[3]
    H_v = nums[4 : 4 + VS]
    if len(H_v) < VS:
        raise ValueError(
            f"parse_pro_file({path}): expected {VS} H_v values, got {H_v}"
        )

    token_pat = re.compile(r'([A-Za-z])_(\d+)')
    stacks: List[List[Tuple[int, int]]] = []
    for raw in stacks_raw:
        containers = [
            (ord(m.group(1).upper()) - ord('A'), int(m.group(2)))
            for m in token_pat.finditer(raw)
        ]
        stacks.append(containers)
    while len(stacks) < YS:
        stacks.append([])

    return {"N": N, "YS": YS, "YT": YT, "VS": VS, "H_v": H_v, "stacks": stacks}


# ================================================================ #
#  CRP-Stow (BRLP) problem                                         #
# ================================================================ #

class CRP_Stow(BaseProblem):
    """
    Blocks Relocation Problem with Stowage Plan.

    Container attributes (re-uses existing Container fields):
        c.group    = vs(c) — vessel stack (0-based)
        c.priority = vt(c) — vessel tier  (0-based, 0 = bottom)

    Each vessel stack s must be loaded in order of increasing tier:
        cdd(c) == 0  ⟺  c.priority == _vessel_loaded[c.group]
    """

    name         = "CRP-Stow"
    description  = (
        "Blocks Relocation Problem with Stowage Plan (BRLP, Jovanović 2019). "
        "Load N containers from a 1-D yard bay into their designated vessel "
        "positions, minimising yard relocations.  Each container has a fixed "
        "vessel stack (vs) and vessel tier (vt); within each vessel stack "
        "containers must be loaded bottom-up.  Supports Jovanović benchmark "
        ".pro files via extra[\"layout_file_path\"]."
    )
    tags         = ["relocation", "stowage", "brlp", "vessel",
                    "two-phase", "jovanovic-2019"]
    metric_names = ["relocations", "steps", "vessel_utilisation"]

    # ---------------------------------------------------------------- #
    # Init                                                               #
    # ---------------------------------------------------------------- #

    def __init__(
        self,
        config: Optional[ProblemConfig] = None,
        render_mode: Optional[str]      = None,
    ):
        if config is None:
            config = ProblemConfig()
        config.num_rows = 1   # BRLP yard is always a 1-D array of stacks
        super().__init__(config, render_mode)

        # Episode state — reset per episode in reset()
        self._total_relocations: int              = 0
        self._total_steps:       int              = 0
        self._total_retrieved:   int              = 0
        self._done:              bool             = False

        # Two-phase state
        self._mode:              str              = "high"
        self._source_stack_idx:  Optional[int]    = None      # flat 0-based idx
        self._target_container:  Optional[Container] = None

        # Vessel configuration — set by _build_episode, persistent per episode
        self._vessel_loaded:    List[int] = []   # [vs] = tiers loaded so far
        self._vessel_max_tier:  List[int] = []   # [vs] = H_v[vs]

    # ---------------------------------------------------------------- #
    # Observation / action spaces                                        #
    # ---------------------------------------------------------------- #

    def _setup_spaces(self) -> None:
        cfg      = self.config
        n_stacks = cfg.num_bays        # num_rows == 1 always
        VS       = max(1, cfg.num_groups)
        obs_size = (n_stacks + VS) * 5
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(obs_size,), dtype=np.int32
        )
        self.action_space = gym.spaces.Discrete(max(n_stacks, 1))

    # ---------------------------------------------------------------- #
    # Reset                                                              #
    # ---------------------------------------------------------------- #

    def reset(
        self,
        seed:    Optional[int]  = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.config.seed = seed

        self.yard.clear()
        self._total_relocations = 0
        self._total_steps       = 0
        self._total_retrieved   = 0
        self._done              = False
        self._mode              = "high"
        self._source_stack_idx  = None
        self._target_container  = None

        self._build_episode()
        self._advance_auto_retrievals()

        return self._get_obs(), self._get_info()

    # ---------------------------------------------------------------- #
    # Episode builders                                                   #
    # ---------------------------------------------------------------- #

    def _build_episode(self) -> None:
        path_str = layout_path_from_extra(self.config.extra)
        if path_str:
            self._load_pro_file(Path(path_str))
        else:
            self._build_random_episode()

    def _load_pro_file(self, path: Path) -> None:
        """Populate yard and vessel config from a Jovanović .pro file."""
        data = parse_pro_file(path)
        YS, YT, VS    = data["YS"], data["YT"], data["VS"]
        H_v           = data["H_v"]
        stacks_raw    = data["stacks"]   # List[List[(vs, vt)]]

        # Update config to match file dimensions, then rebuild yard.
        self.config.num_bays   = YS
        self.config.num_rows   = 1
        self.config.max_tiers  = YT
        self.config.num_groups = VS
        self.yard = Yard(YS, 1, YT)

        self._vessel_loaded   = [0] * VS
        self._vessel_max_tier = list(H_v)

        # Build Container objects (group=vs, priority=vt).
        self.containers = []
        cid = 0
        for raw_stack in stacks_raw:
            for vs, vt in raw_stack:
                self.containers.append(Container(
                    id=cid, group=vs, priority=vt,
                ))
                cid += 1

        # Fast lookup: (vs, vt) → Container (pairs are unique in valid instances).
        c_map: Dict[Tuple[int, int], Container] = {
            (c.group, c.priority): c for c in self.containers
        }

        # Place containers into yard stacks bottom-to-top (file order).
        for s_idx, raw_stack in enumerate(stacks_raw):
            bay = s_idx + 1
            for vs, vt in raw_stack:
                self.yard.place(bay, 1, c_map[(vs, vt)])

        # Rebuild Gym spaces to reflect new dimensions.
        self._setup_spaces()

    def _build_random_episode(self) -> None:
        """Generate a random BRLP instance (used for RL training)."""
        cfg = self.config
        rng = np.random.RandomState(cfg.seed)
        VS  = max(1, cfg.num_groups)
        YS  = max(1, cfg.num_bays)

        # Assign containers to vessel stacks round-robin, then shuffle.
        vt_count = [0] * VS
        vs_list  = [i % VS for i in range(cfg.num_containers)]
        rng.shuffle(vs_list)

        self.containers = []
        for i, vs in enumerate(vs_list):
            vt = vt_count[vs]
            vt_count[vs] += 1
            self.containers.append(Container(id=i, group=vs, priority=vt))

        self._vessel_loaded   = [0] * VS
        self._vessel_max_tier = [max(vt_count[s], 1) for s in range(VS)]

        # Place containers randomly into yard stacks.
        order      = rng.permutation(len(self.containers)).tolist()
        stack_keys = sorted(self.yard.stacks.keys())
        for rank, idx in enumerate(order):
            key = stack_keys[rank % len(stack_keys)]
            stk = self.yard.stacks[key]
            if not stk.is_full:
                self.yard.place(*key, self.containers[idx])

    # ---------------------------------------------------------------- #
    # Retrieval helpers                                                  #
    # ---------------------------------------------------------------- #

    def _is_retrievable(self, c: Container) -> bool:
        """cdd(c) == 0: c.priority equals the number already loaded for vs(c)."""
        if c.group >= len(self._vessel_loaded):
            return False
        return c.priority == self._vessel_loaded[c.group]

    def _do_retrieve(self, c: Container, stack_key: Tuple[int, int]) -> None:
        """Retrieve container c (must be on top of stack_key) into vessel."""
        stk = self.yard.stacks[stack_key]
        assert stk.top is c, "_do_retrieve: c is not on top"
        stk.pop()
        self._vessel_loaded[c.group] += 1
        self._total_retrieved       += 1
        self.yard.total_retrievals  += 1

    def _advance_auto_retrievals(self) -> None:
        """
        Repeatedly scan all yard stacks and retrieve any top container with
        cdd == 0 until no more automatic retrievals are possible.
        """
        changed = True
        while changed:
            changed = False
            for stk in self.yard.stacks.values():
                if stk.is_empty:
                    continue
                top = stk.top
                if self._is_retrievable(top):
                    self._do_retrieve(top, (stk.bay, stk.row))
                    changed = True
                    break   # restart scan after each retrieval

        # Check termination after all cascaded retrievals.
        if all(stk.is_empty for stk in self.yard.stacks.values()):
            self._done = True

    # ---------------------------------------------------------------- #
    # Step                                                               #
    # ---------------------------------------------------------------- #

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self._done:
            return self._get_obs(), 0.0, True, False, self._get_info()

        self._total_steps += 1
        action    = int(action) % self._n_stacks
        stack_key = self._idx_to_stack(action)

        if self._mode == "high":
            reward = self._step_high(action, stack_key)
        else:
            reward = self._step_low(action, stack_key)

        return self._get_obs(), reward, self._done, False, self._get_info()

    def _step_high(self, action: int, stack_key: Tuple[int, int]) -> float:
        """
        H_L phase: agent picks a stack containing a retrievable container.
        Target = the HIGHEST (topmost) retrievable container in that stack.
        """
        stk = self.yard.stacks.get(stack_key)
        if stk is None or stk.is_empty:
            return 0.0

        # Find topmost retrievable container (scan top → bottom).
        target: Optional[Container] = None
        for c in reversed(stk.containers):
            if self._is_retrievable(c):
                target = c
                break

        if target is None:
            return 0.0   # stack has no retrievable container; invalid action

        if stk.top is target:
            # Target already on top → retrieve for free.
            self._do_retrieve(target, stack_key)
            self._advance_auto_retrievals()
        else:
            # Blocked → enter low phase to remove blockers.
            self._target_container = target
            self._source_stack_idx = action
            self._mode             = "low"

        return 0.0

    def _step_low(self, action: int, dst_key: Tuple[int, int]) -> float:
        """
        H_R phase: agent picks a destination for the top blocker of source stack.
        One relocation is performed.
        """
        src_key = self._idx_to_stack(self._source_stack_idx)
        src_stk = self.yard.stacks.get(src_key)
        dst_stk = self.yard.stacks.get(dst_key)

        if (
            src_stk is None or src_stk.is_empty
            or dst_stk is None or dst_stk.is_full
            or dst_key == src_key
        ):
            return 0.0   # invalid; silently ignore

        self.yard.relocate(src_key, dst_key)
        self._total_relocations += 1

        # After relocation: check if target is now on top of source.
        src_stk = self.yard.stacks[src_key]   # re-fetch (mutable)
        if not src_stk.is_empty and src_stk.top is self._target_container:
            self._do_retrieve(self._target_container, src_key)
            self._target_container = None
            self._source_stack_idx = None
            self._mode             = "high"
            self._advance_auto_retrievals()

        return -1.0

    # ---------------------------------------------------------------- #
    # Evaluate (for EA / batch planners)                                #
    # ---------------------------------------------------------------- #

    def evaluate(self, solution: List[int]) -> Dict[str, float]:
        self.reset()
        for action in solution:
            if self._done:
                break
            self.step(action)
        return self.get_metrics()

    # ---------------------------------------------------------------- #
    # Metrics                                                            #
    # ---------------------------------------------------------------- #

    def get_metrics(self) -> Dict[str, float]:
        total_capacity = max(sum(self._vessel_max_tier), 1) if self._vessel_max_tier else 1
        return {
            "relocations":        float(self._total_relocations),
            "steps":              float(self._total_steps),
            "time":               float(self._total_relocations),
            "vessel_utilisation": self._total_retrieved / total_capacity,
            "retrieved":          float(self._total_retrieved),
        }

    # ---------------------------------------------------------------- #
    # Observation / info / mask                                         #
    # ---------------------------------------------------------------- #

    @property
    def _n_stacks(self) -> int:
        return self.config.num_bays   # num_rows is always 1

    def _get_obs(self) -> np.ndarray:
        """
        Flat (YS + VS) × 5 observation.

        Yard node i  : [i+1, height, n_retrievable_in_stack, top_vs+1, top_vt+1]
        Vessel node j: [j+1, loaded, max_tier, 0, 0]

        All values 0-padded for empty/absent entries.
        """
        n_stacks = self._n_stacks
        VS       = self.config.num_groups
        obs      = np.zeros((n_stacks + VS, 5), dtype=np.int32)

        for i in range(n_stacks):
            key = self._idx_to_stack(i)
            stk = self.yard.stacks.get(key)
            if stk and not stk.is_empty:
                n_ret  = sum(1 for c in stk.containers if self._is_retrievable(c))
                top    = stk.top
                obs[i] = [i + 1, stk.height, n_ret, top.group + 1, top.priority + 1]
            else:
                obs[i] = [i + 1, 0, 0, 0, 0]

        for j in range(VS):
            loaded   = self._vessel_loaded[j]   if j < len(self._vessel_loaded)   else 0
            max_tier = self._vessel_max_tier[j] if j < len(self._vessel_max_tier) else 0
            obs[n_stacks + j] = [j + 1, loaded, max_tier, 0, 0]

        return obs.flatten()

    def _get_info(self) -> Dict:
        info = self.get_metrics()
        info["action_mask"] = self._build_action_mask()
        info["mode"]        = self._mode
        return info

    def _build_action_mask(self) -> np.ndarray:
        """
        "high": valid = stacks containing at least one retrievable container.
        "low":  valid = non-full stacks ≠ source stack.
        """
        n    = self._n_stacks
        mask = np.zeros(n, dtype=np.bool_)
        if self._done:
            return mask

        if self._mode == "high":
            for i in range(n):
                key = self._idx_to_stack(i)
                stk = self.yard.stacks.get(key)
                if stk and any(self._is_retrievable(c) for c in stk.containers):
                    mask[i] = True
        else:
            src_key = self._idx_to_stack(self._source_stack_idx)
            for i in range(n):
                key = self._idx_to_stack(i)
                if key == src_key:
                    continue
                stk = self.yard.stacks.get(key)
                if stk and not stk.is_full:
                    mask[i] = True

        return mask

    # ---------------------------------------------------------------- #
    # State snapshot for GUI                                             #
    # ---------------------------------------------------------------- #

    def get_state_snapshot(self) -> Dict:
        snap = super().get_state_snapshot()
        snap["mode"]          = self._mode
        snap["vessel_loaded"] = list(self._vessel_loaded)
        snap["vessel_max"]    = list(self._vessel_max_tier)
        return snap

    # ---------------------------------------------------------------- #
    # Helpers                                                            #
    # ---------------------------------------------------------------- #

    def _idx_to_stack(self, idx: int) -> Tuple[int, int]:
        """Flat 0-based index → (bay, 1) 1-indexed key."""
        return (int(idx) % self._n_stacks + 1, 1)

    # ---------------------------------------------------------------- #
    # Config schema for GUI                                              #
    # ---------------------------------------------------------------- #

    @classmethod
    def config_schema(cls) -> Dict:
        schema = super().config_schema()
        schema.update({
            "num_groups": {
                "type": "int", "default": 5, "min": 1, "max": 26,
                "label": "Vessel stacks (VS)",
                "help":  "Number of vessel stacks; containers are assigned A=0, B=1, … Z=25.",
            },
        })
        # Remove vessel_bays / vessel_rows / vessel_tiers — not used in BRLP
        # (vessel geometry is encoded per-container in vs/vt, not as a global slot array).
        for key in ("vessel_bays", "vessel_rows", "vessel_tiers"):
            schema.pop(key, None)
        return schema

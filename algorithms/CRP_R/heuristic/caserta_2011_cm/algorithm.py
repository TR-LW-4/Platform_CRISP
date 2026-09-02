"""
CasertaCM
<2011> <heuristic> <restricted> <single-bay> <CRP-R>
Corridor Method: DP-inspired search in a local relocation corridor
delta --- 2 --- Horizontal corridor half-width
time_limit --- 5.0 --- Wall-clock limit per instance in seconds

------------------------------- Reference --------------------------------
M. Caserta, S. Voß, M. Sniedovich,
"Applying the corridor method to a blocks relocation problem",
OR Spectrum 33 (2011) 915–929.
------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
paper listed in the Reference section.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import multiprocessing as mp
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from core.base_algorithm import BaseAlgorithm, AlgorithmConfig
from core.layout_trace import trace_layout
from core.plan import Movement, RelocationPlan
from .cm_export import yard_to_cm_instance


class CasertaCM(BaseAlgorithm):

    name                = "Caserta et al. (2011) CM"
    category            = "Heuristic"
    description         = "Caserta et al. (OR Spectrum 2011) corridor method heuristic."
    compatible_problems = ["CRP-R"]
    _MOVES_RE = re.compile(r"CM\s*:\s*Solution found with\s*(\d+)\s*moves")
    _MOVE_LINE_RE = re.compile(r"^CM_MOVE\s+(\d+)\s+(\d+)\s+(-?\d+)$")

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config)

    # ---------------------------------------------------------------- #
    # Binary resolution                                                  #
    # ---------------------------------------------------------------- #

    def _resolve_binary(self) -> Path:
        p = Path(__file__).resolve().parent / "vendor" / "brp_cm"
        if not p.is_file():
            raise FileNotFoundError(
                f"CM binary not found at {p}. "
                "Re-compile with:\n"
                "  cd vendor && CPLUS_INCLUDE_PATH='' g++ -O3 -DREPL "
                "timer.cpp options.cpp heuristic.cpp containers.cpp -o brp_cm"
            )
        return p

    @classmethod
    def _parse_cm_output(
        cls, stdout: str
    ) -> Tuple[int, List[Tuple[int, int, int]]]:
        """Parse the machine-readable complete path emitted by ``brp_cm``."""
        count_match = cls._MOVES_RE.search(stdout)
        if count_match is None:
            raise RuntimeError("CM output contains no solution count")

        records: List[Tuple[int, int, int]] = []
        in_moves = False
        saw_begin = False
        saw_end = False
        for line in stdout.splitlines():
            stripped = line.strip()
            if stripped == "CM_MOVES_BEGIN":
                in_moves = True
                saw_begin = True
                continue
            if stripped == "CM_MOVES_END":
                in_moves = False
                saw_end = True
                break
            if not in_moves:
                continue
            move_match = cls._MOVE_LINE_RE.match(stripped)
            if move_match is None:
                raise RuntimeError(f"Malformed CM move line: {stripped!r}")
            records.append(tuple(int(v) for v in move_match.groups()))

        if not saw_begin or not saw_end or not records:
            raise RuntimeError("CM binary returned a count but no replayable path")
        return int(count_match.group(1)), records

    # ---------------------------------------------------------------- #
    # Core subprocess call                                               #
    # ---------------------------------------------------------------- #

    def _run_cm(
        self,
        instance_text: str,
        delta:         int,
        max_height:    int,
        time_limit:    float,
    ) -> Tuple[int, List[Tuple[int, int, int]], str]:
        """
        Write instance to a temp file, run the CM binary, and parse output.

        Parameters
        ----------
        delta      : horizontal corridor half-width (-d parameter)
        max_height : maximum stack height (-n with -c 1); use env.config.max_tiers
        time_limit : wall-clock time limit in seconds (-t parameter)

        Returns
        -------
        (relocations, move_records, raw_stdout), where each move record is
        ``(container_id, zero_based_src_stack, zero_based_dst_stack)`` and
        destination ``-1`` denotes retrieval.
        """
        binary = self._resolve_binary()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as tmp:
            tmp.write(instance_text)
            tmp_path = tmp.name

        try:
            result = subprocess.run(
                [
                    str(binary),
                    "-f", tmp_path,
                    "-d", str(delta),
                    "-n", str(max_height),
                    "-t", str(int(time_limit)),
                    "-c", "1",   # constant vertical corridor: -n is max stack height
                ],
                capture_output=True,
                text=True,
                timeout=time_limit + 30,
                env={
                    **os.environ,
                    "CPLUS_INCLUDE_PATH": "",
                    "CRISP_SEED": str(int(self.config.seed)),
                },
            )
            stdout = result.stdout.strip()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        try:
            relocations, records = self._parse_cm_output(stdout)
        except RuntimeError as exc:
            raise RuntimeError(
                f"CM binary produced unexpected output: {stdout!r}\n"
                f"stderr: {result.stderr[:300]!r}"
            ) from exc
        return relocations, records, stdout

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

        cfg        = self.config
        delta      = int(cfg.extra.get("delta", 2))
        time_limit = float(cfg.extra.get("time_limit", 5.0))

        trace_layout(
            f"CasertaCM.train: delta={delta}  time_limit={time_limit}s"
        )

        env = problem_factory()
        env.reset(options={"skip_auto_retrieve": True})

        max_height = int(env.config.max_tiers)
        instance_text = yard_to_cm_instance(env.config, env.yard)
        search_relocations, records, _ = self._run_cm(
            instance_text, delta, max_height, time_limit
        )

        num_rows = int(env.config.num_rows)

        def stack_pos(index: int) -> Tuple[int, int]:
            return index // num_rows + 1, index % num_rows + 1

        plan = RelocationPlan([
            Movement(
                container_id=container_id,
                from_pos=stack_pos(src),
                to_pos=None if dst < 0 else stack_pos(dst),
            )
            for container_id, src, dst in records
        ])
        metrics = env.validate_plan(plan)
        metrics["search_relocations"] = float(search_relocations)
        metrics["count_match"] = float(
            metrics["feasible"] == 1.0
            and metrics["relocations"] == float(search_relocations)
        )
        metrics["progress"] = 1.0
        solution = [
            (move.to_pos[0] - 1) * num_rows + (move.to_pos[1] - 1)
            for move in plan.movements
            if move.to_pos is not None
        ]
        self._best_solution = solution

        self._push(
            result_queue,
            step     = 1,
            metric   = float(metrics["relocations"]),
            metrics  = metrics,
            progress = 1.0,
            extra={
                "solution": solution[:],
                "moves": [
                    {
                        "container_id": move.container_id,
                        "from": list(move.from_pos),
                        "to": list(move.to_pos) if move.to_pos is not None else None,
                        "kind": "retrieve" if move.to_pos is None else "relocate",
                    }
                    for move in plan.movements
                ],
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
            "delta": {
                "type":    "int",
                "default": 2,
                "min":     1,
                "max":     20,
                "label":   "Horizontal corridor half-width (δ)",
                "help": (
                    "Blocks above the target may only be relocated to stacks "
                    "within [i−δ, i+δ].  Larger δ improves quality but increases "
                    "runtime.  Paper uses δ = 1 for most small instances, δ = 2 "
                    "for larger ones."
                ),
            },
            "time_limit": {
                "type":    "float",
                "default": 5.0,
                "min":     1.0,
                "max":     300.0,
                "label":   "Time limit per instance (s)",
                "help": (
                    "Wall-clock time limit passed to the CM binary. "
                    "The algorithm runs multiple restarts until this limit is reached "
                    "and returns the best solution found. "
                    "Default 5 s is sufficient for small/medium instances. "
                    "Paper uses 60 s for small/medium and 300 s for very large instances."
                ),
            },
        })
        return base

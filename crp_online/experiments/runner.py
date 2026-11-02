import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Union
from ..core.instance import ProblemInstance
from ..core.environment import Environment
from ..evaluator.prefix_evaluator import PrefixExpression
from ..heuristics.restricted import solve_restricted
from ..heuristics.unrestricted import solve_unrestricted
from ..heuristics.manual import (
    select_tlp_destination,
    select_ri_destination,
    select_minmax_destination,
    select_pr4_destination,
    select_rdh_destination,
    select_eri_destination,
    select_em_destination,
    solve_egh,
    solve_pu1,
    solve_pu2,
)


@dataclass
class BenchmarkResult:
    """Benchmark evaluation summary across problem instances."""
    condition_name: str
    num_instances: int
    totals: Dict[str, int] = field(default_factory=dict)
    means: Dict[str, float] = field(default_factory=dict)
    elapsed_seconds: Dict[str, float] = field(default_factory=dict)
    instance_relocations: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def print_summary(self) -> None:
        print(f"\n{'='*70}")
        print(f"Benchmark Summary: {self.condition_name} ({self.num_instances} instances)")
        print(f"{'='*70}")
        print(f"{'Heuristic':<15} | {'Total Relocations':<18} | {'Mean / Instance':<16} | {'Runtime'}")
        print(f"{'-'*70}")
        for name in sorted(self.totals.keys(), key=lambda k: self.totals[k]):
            tot = self.totals[name]
            mean = self.means[name]
            sec = self.elapsed_seconds.get(name, 0.0)
            print(f"{name:<15} | {tot:<18} | {mean:<16.4f} | {sec:.2f}s")
        print(f"{'='*70}\n")


class BenchmarkRunner:
    """Orchestrates standard experimental benchmarks across Caserta benchmark instances."""

    STANDARD_HEURISTICS: Dict[str, str] = {
        "TLP": "restricted_manual",
        "RI": "restricted_manual",
        "MM": "restricted_manual",
        "PR4": "restricted_manual",
        "RDH": "restricted_manual",
        "ERI": "restricted_manual",
        "EM": "restricted_manual",
        "EGH": "custom_solver",
        "PU1": "custom_solver",
        "PU2": "custom_solver",
    }

    @staticmethod
    def run_heuristic(
        name: str,
        inst: ProblemInstance,
        look_ahead_level: float = 1.0,
        horizon_strategy: str = "ignore",
        use_container_groups: bool = False,
        num_groups: int = 2,
        group_size: Optional[int] = None,
        known_within_group: bool = False,
        mode: str = "cpp_compat",
    ) -> int:
        """Solves a single instance with the named heuristic and returns relocation count."""
        env = Environment(inst, store_moves=False)

        kwargs = dict(
            look_ahead_level=look_ahead_level,
            horizon_strategy=horizon_strategy,
            use_container_groups=use_container_groups,
            num_groups=num_groups,
            group_size=group_size,
            known_within_group=known_within_group,
            mode=mode,
        )

        if name == "TLP":
            solve_restricted(env, select_tlp_destination, **kwargs)
        elif name == "RI":
            solve_restricted(env, select_ri_destination, **kwargs)
        elif name == "MM":
            solve_restricted(env, select_minmax_destination, **kwargs)
        elif name == "PR4":
            solve_restricted(env, select_pr4_destination, **kwargs)
        elif name == "RDH":
            solve_restricted(env, select_rdh_destination, **kwargs)
        elif name == "ERI":
            solve_restricted(env, select_eri_destination, **kwargs)
        elif name == "EM":
            solve_restricted(env, select_em_destination, **kwargs)
        elif name == "EGH":
            solve_egh(env, **kwargs)
        elif name == "PU1":
            solve_pu1(env, **kwargs)
        elif name == "PU2":
            solve_pu2(env, **kwargs)
        else:
            # Assume name is a prefix expression
            expr = PrefixExpression(name)
            solve_restricted(env, expr, **kwargs)

        return env.total_relocations

    def run_benchmark(
        self,
        condition_name: str,
        heuristics: Sequence[str],
        instances: Sequence[ProblemInstance],
        look_ahead_level: float = 1.0,
        horizon_strategy: str = "ignore",
        use_container_groups: bool = False,
        num_groups: int = 2,
        group_size: Optional[int] = None,
        known_within_group: bool = False,
        mode: str = "cpp_compat",
    ) -> BenchmarkResult:
        """Runs a suite of heuristics across all supplied problem instances."""
        res = BenchmarkResult(condition_name=condition_name, num_instances=len(instances))

        for h_name in heuristics:
            t0 = time.time()
            tot = 0
            inst_map: Dict[str, int] = {}

            for inst in instances:
                relocs = self.run_heuristic(
                    h_name,
                    inst,
                    look_ahead_level=look_ahead_level,
                    horizon_strategy=horizon_strategy,
                    use_container_groups=use_container_groups,
                    num_groups=num_groups,
                    group_size=group_size,
                    known_within_group=known_within_group,
                    mode=mode,
                )
                tot += relocs
                inst_map[inst.name] = relocs

            elapsed = time.time() - t0
            res.totals[h_name] = tot
            res.means[h_name] = tot / float(len(instances))
            res.elapsed_seconds[h_name] = elapsed
            res.instance_relocations[h_name] = inst_map

        return res

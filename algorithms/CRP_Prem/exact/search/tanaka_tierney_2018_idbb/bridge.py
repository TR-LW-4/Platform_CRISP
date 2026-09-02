"""
Python–C bridge for TanakaTierney2018IDBB.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import math
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .core import Move, Stacks

_RELOCATION_RE = re.compile(r"Relocation\s+\d+:\s*[\[<]\s*-?\d+\s*[\]>]\s*(\d+)->(\d+)")
_OPT_RE = re.compile(r"^opt=(\d+)\s*$", re.MULTILINE)
_BEST_RE = re.compile(r"^best=(\d+)\s*$", re.MULTILINE)
_NO_SOLUTION_RE = re.compile(r"No feasible solution found\.")

_build_lock = threading.Lock()


@dataclass
class VendorRunResult:
    moves: List[Move] = field(default_factory=list)
    solved: bool = False
    proved_optimal: bool = False
    n_relocation: Optional[int] = None
    timed_out: bool = False
    error: Optional[str] = None
    raw_stdout: str = ""
    raw_stderr: str = ""


def _vendor_root() -> Path:
    return Path(__file__).resolve().parent / "vendor" / "pmp-1.02"


def _binary_path() -> Path:
    return _vendor_root() / "pmp"


def _run_cmd(cmd: List[str], cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({' '.join(cmd)}):\n{proc.stdout}\n{proc.stderr}".strip()
        )


def ensure_binary_built() -> Path:
    """Compile the vendored solver on first use; reuse the cached binary
    afterwards. Thread-safe (guards against concurrent training runs racing
    to build at the same time)."""
    root = _vendor_root()
    if not root.exists():
        raise RuntimeError(f"Vendored IDBB source missing: {root}")

    exe = _binary_path()
    with _build_lock:
        if exe.exists():
            return exe
        _run_cmd(["make", "clean"], cwd=root)
        _run_cmd(["make"], cwd=root)
        if not exe.exists():
            raise RuntimeError(f"Build finished but binary not found at {exe}")
        return exe


def _encode_instance(stacks_init: Stacks, max_tiers: int, stack_order: List[int]) -> str:
    n_stacks = len(stack_order)
    n_containers = sum(len(stacks_init.get(s, [])) for s in stack_order)
    lines = [
        f"Tiers: {int(max_tiers)}",
        f"Stacks: {n_stacks}",
        f"Containers: {n_containers}",
    ]
    for i, s in enumerate(stack_order):
        values = stacks_init.get(s, [])
        payload = " ".join(str(int(v)) for v in values)
        lines.append(f"Stack {i + 1}: {payload}")
    return "\n".join(lines) + "\n"


def _parse_output(stdout: str, stderr: str, hit_internal_time_budget: bool) -> VendorRunResult:
    result = VendorRunResult(raw_stdout=stdout, raw_stderr=stderr)

    if _NO_SOLUTION_RE.search(stderr):
        # The vendored program prints this whenever no feasible solution was
        # *recorded* by the time solve() returned, for either of two very
        # different reasons that its own output does not directly tell
        # apart: (a) the search ran to completion and proved no solution
        # exists within MAX_N_RELOCATION=200 relocations (genuine
        # infeasibility -- practically only reachable for pathologically
        # over-packed yards with zero buffer space), or (b) the internal
        # ``-t`` time budget expired before any solution was found (b&b
        # incomplete -- solvability is simply unknown). We disambiguate
        # using wall-clock time: real infeasibility proofs on the tiny
        # instances this represents are near-instant, so a run that
        # actually consumed close to the full time budget is (b), not (a).
        result.timed_out = hit_internal_time_budget
        return result

    m_opt = _OPT_RE.search(stderr)
    m_best = _BEST_RE.search(stderr)
    if m_opt is None and m_best is None:
        result.error = "unable to parse solver output (no opt=/best=/No-feasible marker found)"
        return result

    if m_opt is not None:
        result.n_relocation = int(m_opt.group(1))
        result.proved_optimal = True
    else:
        result.n_relocation = int(m_best.group(1))
        result.proved_optimal = False

    moves: List[Move] = []
    for src_1idx, dst_1idx in _RELOCATION_RE.findall(stdout):
        moves.append((int(src_1idx) - 1, int(dst_1idx) - 1))

    result.moves = moves
    result.solved = len(moves) == result.n_relocation
    return result


def run_idbb(
    stacks_init: Stacks,
    max_tiers: int,
    time_limit_s: float = 30.0,
) -> VendorRunResult:
    """Run the vendored IDBB solver on one instance.

    ``stacks_init`` keys may be arbitrary ints; stacks are ordered by
    ``sorted(stacks_init.keys())`` for the C program's 1-indexed stack
    numbering, and returned moves are translated back to the original keys.
    """
    stack_order = sorted(stacks_init.keys())
    if not stack_order:
        return VendorRunResult(moves=[], solved=True, proved_optimal=True, n_relocation=0)

    try:
        exe = ensure_binary_built()
    except Exception as exc:
        return VendorRunResult(error=f"build_failed: {exc}")

    instance_text = _encode_instance(stacks_init, max_tiers, stack_order)

    tlimit_int = max(1, math.ceil(time_limit_s))
    hard_timeout = max(5.0, time_limit_s * 1.5 + 10.0)

    start = time.perf_counter()
    try:
        proc = subprocess.run(
            [str(exe), "-s", "-t", str(tlimit_int)],
            input=instance_text,
            capture_output=True,
            text=True,
            check=False,
            timeout=hard_timeout,
        )
        stdout, stderr = proc.stdout, proc.stderr
        hard_timed_out = False
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout or ""
        stderr = e.stderr or ""
        hard_timed_out = True
    elapsed = time.perf_counter() - start

    # The internal ``-t`` check only fires every 200000 explored nodes, so
    # small/fast instances can safely finish in a fraction of a second even
    # with a generous time budget; only treat this as "hit the internal
    # budget" when the run actually consumed most of the requested window.
    hit_internal_time_budget = hard_timed_out or elapsed >= 0.9 * tlimit_int

    parsed = _parse_output(stdout, stderr, hit_internal_time_budget)
    if hard_timed_out:
        parsed.timed_out = True
    elif parsed.n_relocation is not None and not parsed.proved_optimal:
        parsed.timed_out = True

    parsed.moves = [(stack_order[s], stack_order[d]) for (s, d) in parsed.moves]
    return parsed

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .core import Move, Stacks


@dataclass
class VendorRunResult:
    moves: List[Move]
    duration_seconds: float
    stdout: str


def _write_bay_file(path: Path, stacks_init: Stacks, max_tiers: int) -> None:
    width = len(stacks_init)
    container_count = sum(len(v) for v in stacks_init.values())
    # BayReader adds +2 for this header format.
    declared_height = max(1, int(max_tiers) - 2)

    lines = [
        "CPMP instance",
        f"Width: {width}",
        f"Height: {declared_height}",
        f"Containers: {container_count}",
    ]
    for stack_idx in range(width):
        arr = stacks_init.get(stack_idx, [])
        payload = " ".join(str(x) for x in arr)
        lines.append(f"{stack_idx + 1}: {payload}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_moves(stdout: str) -> List[Move]:
    move_pattern = re.compile(r"^### Move \d+: \((\d+),(\d+)\)$", re.MULTILINE)
    out: List[Move] = []
    for src, dst in move_pattern.findall(stdout):
        out.append((int(src), int(dst)))
    return out


def _vendor_root() -> Path:
    return (
        Path(__file__).resolve().parent
        / "vendor"
        / "eusorpb-brkga-cpmp"
        / "CpmpSolver"
    )


def _find_executable(project_root: Path) -> Optional[Path]:
    candidates = [
        project_root / "bin" / "Release" / "CpmpSolver.exe",
        project_root / "bin" / "Debug" / "CpmpSolver.exe",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _run_cmd(cmd: Sequence[str], cwd: Path) -> None:
    proc = subprocess.run(
        list(cmd),
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({' '.join(cmd)}):\n{proc.stdout}\n{proc.stderr}".strip()
        )


def _ensure_vendor_exe(project_root: Path) -> Path:
    exe = _find_executable(project_root)
    if exe is not None:
        return exe

    sln_root = project_root.parent
    if shutil.which("nuget"):
        _run_cmd(["nuget", "restore", "CpmpSolver.sln"], cwd=sln_root)

    build_cmds = []
    if shutil.which("msbuild"):
        build_cmds.append(["msbuild", "CpmpSolver.csproj", "/p:Configuration=Release"])
    if shutil.which("xbuild"):
        build_cmds.append(["xbuild", "CpmpSolver.csproj", "/p:Configuration=Release"])

    last_err: Optional[Exception] = None
    for cmd in build_cmds:
        try:
            _run_cmd(cmd, cwd=project_root)
            exe = _find_executable(project_root)
            if exe is not None:
                return exe
        except Exception as exc:  # pragma: no cover - best effort fallback path
            last_err = exc

    if last_err is not None:
        raise RuntimeError(f"Failed to build vendored BRKGA solver: {last_err}") from last_err
    raise RuntimeError(
        "No build tool found for vendored BRKGA solver (expected msbuild or xbuild)."
    )


def run_vendor_solver(
    stacks_init: Stacks,
    max_tiers: int,
    seed: int,
    extra: Dict[str, object],
) -> VendorRunResult:
    project_root = _vendor_root()
    if not project_root.exists():
        raise RuntimeError(f"Vendored BRKGA source missing: {project_root}")

    exe_path = _ensure_vendor_exe(project_root)

    with tempfile.TemporaryDirectory(prefix="crisp_brkga_vendor_") as td:
        instance_path = Path(td) / "instance.bay"
        _write_bay_file(instance_path, stacks_init=stacks_init, max_tiers=max_tiers)

        cmd = [
            "mono",
            str(exe_path),
            "-i",
            str(instance_path),
            "-s",
            str(int(seed)),
            "-p",
            str(int(extra.get("population_size", 190))),
            "-q",
            str(int(extra.get("elite_population_size", 13))),
            "-r",
            str(int(extra.get("mutation_offspring_size", 19))),
            "-t",
            str(int(extra.get("offspring_size", 130))),
            "-u",
            str(int(extra.get("max_generations", 1164))),
            "-w",
            str(int(extra.get("max_generations_without_improvement", 74))),
            "-x",
            str(int(extra.get("vendor_biased_percentage", 65))),
            "-y",
            str(int(extra.get("vendor_max_cpu_time", 1200))),
            "-z",
            str(float(extra.get("vendor_min_share_valid", 0.210087))),
        ]

        if not shutil.which("mono"):
            raise RuntimeError("mono runtime not found, cannot execute vendored CpmpSolver.exe.")

        start = time.perf_counter()
        proc = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=False,
        )
        duration = time.perf_counter() - start
        output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        if proc.returncode != 0:
            raise RuntimeError(f"Vendored solver failed (exit={proc.returncode}):\n{output}")

    return VendorRunResult(moves=_parse_moves(output), duration_seconds=duration, stdout=output)

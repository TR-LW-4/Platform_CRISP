"""Loader and validator for the Ku & Arthanari (2016) / Galle et al. (2018) CRPTW benchmark."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class KuInstance:
    """Represents a Container Relocation Problem with Time Windows (CRPTW) instance.
    
    Attributes:
        name: Instance identifier (e.g., 'T271014_0503_001.txt').
        bay_id: Bay identifier (typically 1).
        num_stacks: Number of stacks (S).
        num_tiers: Maximum stack capacity / tiers (T = H).
        num_containers: Total number of containers (C).
        num_batches: Number of departure time windows declared in header (W).
        dense_window_count: Number of unique departure windows actually present in the bay.
        nominal_fill: Nominal target fill rate (0.50 or 0.67).
        occupancy: Actual storage occupancy fraction (C / (S * T)).
        stack_windows: Tuple of tuples listing departure time windows per stack from floor (tier 1) to top.
    """
    name: str
    bay_id: int
    num_stacks: int
    num_tiers: int
    num_containers: int
    num_batches: int
    dense_window_count: int
    nominal_fill: float
    occupancy: float
    stack_windows: Tuple[Tuple[int, ...], ...]

    @property
    def max_stack_height(self) -> int:
        """Alias for num_tiers, consistent with CRP Environment capacity H."""
        return self.num_tiers

    def get_initial_blocking_bound(self) -> float:
        """Computes Galle et al. (2018) blocking lower bound b(n_0) on the initial bay layout."""
        total_b = 0.0
        for stk in self.stack_windows:
            h = len(stk)
            if h == 0:
                continue
            # stk is floor-to-top: index 0 is tier 1, index h-1 is top
            # For each tier t (1..h), check if container at tier t equals min(stk[0..t-1])
            for t_idx in range(h):
                val = stk[t_idx]
                prefix = stk[:t_idx + 1]
                min_prefix = min(prefix)
                if val == min_prefix:
                    count_equal_min = prefix.count(min_prefix)
                    total_b += 1.0 - (1.0 / count_equal_min)
                else:
                    total_b += 1.0
        return total_b

    def __repr__(self) -> str:
        return (
            f"KuInstance('{self.name}', S={self.num_stacks}, T={self.num_tiers}, "
            f"C={self.num_containers}, W={self.num_batches}, fill={self.nominal_fill:.2f})"
        )


def parse_ku_file(filepath: Path | str) -> KuInstance:
    """Parses a single Ku & Arthanari (2016) CRPTW text instance file.
    
    Enforces all preflight validation gates:
    - Verifies header fields: name, bay, S, T, C, W.
    - Asserts es_i == ls_i for every container (exact time windows).
    - Asserts container conservation: sum(stack heights) == header C.
    - Asserts physical feasibility: C <= S * T - (T - 1).
    - Preserves bottom-to-top container layout.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Ku instance file not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise ValueError(f"Empty instance file: {path}")

    # Header parsing
    header_parts = lines[0].split()
    if len(header_parts) < 6:
        raise ValueError(f"Malformed header in {path.name}: '{lines[0]}'")

    instance_name = header_parts[0]
    bay_id = int(header_parts[1])
    num_stacks = int(header_parts[2])
    num_tiers = int(header_parts[3])
    num_containers = int(header_parts[4])
    num_batches = int(header_parts[5])

    # Infer nominal fill from filename
    if "T27" in instance_name:
        nominal_fill = 0.50
    elif "T28" in instance_name:
        nominal_fill = 0.67
    else:
        # Fallback to nearest nominal fill
        nominal_fill = 0.50 if (num_containers / (num_stacks * num_tiers)) < 0.585 else 0.67

    occupancy = num_containers / (num_stacks * num_tiers)

    # Stack parsing
    if len(lines) < num_stacks + 1:
        raise ValueError(f"Expected {num_stacks} stack rows, found {len(lines) - 1} in {path.name}")

    stacks_list: List[Tuple[int, ...]] = []
    total_parsed_containers = 0
    unique_windows = set()

    for s_idx in range(num_stacks):
        row_parts = [int(p) for p in lines[1 + s_idx].split()]
        if len(row_parts) < 3:
            raise ValueError(f"Malformed stack line {s_idx + 1} in {path.name}: {lines[1 + s_idx]}")

        stack_id = row_parts[1]
        height = row_parts[2]

        expected_pairs_len = 3 + 2 * height
        if len(row_parts) != expected_pairs_len:
            raise ValueError(
                f"Stack {stack_id} in {path.name} specifies height {height} but has {len(row_parts) - 3} pair values"
            )

        stack_windows_list: List[int] = []
        for t in range(height):
            es = row_parts[3 + 2 * t]
            ls = row_parts[4 + 2 * t]
            if es != ls:
                raise ValueError(
                    f"Gate 1 Invariant Broken: Container at stack {stack_id}, tier {t + 1} has es={es} != ls={ls}"
                )
            stack_windows_list.append(es)
            unique_windows.add(es)

        stacks_list.append(tuple(stack_windows_list))
        total_parsed_containers += height

    # Preflight Assertions
    if total_parsed_containers != num_containers:
        raise ValueError(
            f"Container conservation failed in {path.name}: Header states C={num_containers}, parsed {total_parsed_containers}"
        )

    # Feasibility assertion: C <= S * T - (T - 1)
    max_feasible_c = num_stacks * num_tiers - (num_tiers - 1)
    if num_containers > max_feasible_c:
        raise ValueError(
            f"Feasibility gate broken in {path.name}: C={num_containers} exceeds max feasible capacity {max_feasible_c}"
        )

    # Ensure stack count matches
    if len(stacks_list) != num_stacks:
        raise ValueError(f"Parsed {len(stacks_list)} stacks, expected {num_stacks}")

    return KuInstance(
        name=path.name,
        bay_id=bay_id,
        num_stacks=num_stacks,
        num_tiers=num_tiers,
        num_containers=num_containers,
        num_batches=num_batches,
        dense_window_count=len(unique_windows),
        nominal_fill=nominal_fill,
        occupancy=occupancy,
        stack_windows=tuple(stacks_list),
    )


def load_all_ku_instances(data_dir: Path | str) -> Dict[str, KuInstance]:
    """Discovers and parses all 1,440 Ku & Arthanari instances across the 24 subfolders."""
    base_path = Path(data_dir)
    if not base_path.is_dir():
        # Auto-extract from ku_arthanari.zip if available
        zip_candidates = [
            base_path.with_suffix(".zip"),
            base_path.parent / "ku_arthanari.zip",
            base_path.parent.parent / "ku_arthanari.zip",
            Path("data/ku_arthanari.zip"),
            Path("ku_arthanari.zip"),
        ]
        extracted = False
        for z in zip_candidates:
            if z.is_file():
                import zipfile
                base_path.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(z, "r") as zf:
                    zf.extractall(base_path)
                extracted = True
                break
        if not extracted and not base_path.is_dir():
            raise FileNotFoundError(f"Ku dataset directory not found: {base_path.resolve()}")

    instances: Dict[str, KuInstance] = {}
    for txt_file in sorted(base_path.glob("*/*.txt")):
        inst = parse_ku_file(txt_file)
        instances[inst.name] = inst

    if len(instances) != 1440:
        # Check if searching flat directory
        flat_files = list(base_path.glob("*.txt"))
        if flat_files:
            for txt_file in sorted(flat_files):
                inst = parse_ku_file(txt_file)
                instances[inst.name] = inst

    return instances


def generate_ku_manifest_checksums(data_dir: Path | str, output_file: Path | str) -> int:
    """Computes SHA-256 hashes for all Ku instance files and saves a verification manifest."""
    base_path = Path(data_dir)
    out_path = Path(output_file)

    files = sorted(base_path.glob("*/*.txt"))
    if not files:
        files = sorted(base_path.glob("*.txt"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        for f in files:
            sha256 = hashlib.sha256(f.read_bytes()).hexdigest()
            rel_path = f.relative_to(base_path).as_posix()
            out.write(f"{sha256}  {rel_path}\n")

    return len(files)


def parse_ku_as_problem_instance(filepath: Path | str, seed: int = 0):
    """Parses a Ku & Arthanari text file and maps it to a standard ProblemInstance."""
    import numpy as np
    from ..core.instance import ProblemInstance
    from ..core.crptw_environment import precompute_departure_sequence

    ku_inst = parse_ku_file(filepath)
    rng = np.random.Generator(np.random.PCG64(seed))
    seq = precompute_departure_sequence(ku_inst, rng)

    rank_map = {cid: rank + 1 for rank, cid in enumerate(seq)}

    cid = 0
    layout = []
    for stk in ku_inst.stack_windows:
        stk_ranks = []
        for _ in stk:
            stk_ranks.append(rank_map[cid])
            cid += 1
        layout.append(tuple(stk_ranks))

    initial_height = max((len(s) for s in layout), default=1)
    max_height = max(ku_inst.num_tiers, initial_height + 1)

    return ProblemInstance(
        name=ku_inst.name,
        num_stacks=ku_inst.num_stacks,
        num_containers=ku_inst.num_containers,
        max_stack_height=max_height,
        initial_height=initial_height,
        initial_layout=tuple(layout),
    )

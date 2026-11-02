"""Parser and loader for Caserta et al. (2011) CRP benchmark dataset files."""

import re
from pathlib import Path
from typing import Dict, List, Optional
from ..core.instance import ProblemInstance


_FILENAME_PATTERN = re.compile(r"^data(\d+)-(\d+)-(\d+)\.dat$")


def parse_caserta_file(file_path: Path | str) -> ProblemInstance:
    """Parses a single Caserta .dat file into an immutable ProblemInstance.
    
    File format:
      Line 1: S N (number of stacks, total number of containers)
      Lines 2..S+1: count c_0 c_1 ... c_{k-1}
        where count is the number of containers on that stack,
        c_0 is bottom tier (tier 0 / floor), and c_{k-1} is top tier.
    
    Capacity rule:
      H = h + 2 where h = initial stack height extracted from filename 'data{h}-{S}-{idx}.dat'.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Caserta benchmark file not found: {path}")

    filename = path.name
    match = _FILENAME_PATTERN.match(filename)
    if not match:
        raise ValueError(f"Filename '{filename}' does not match Caserta format 'data{{h}}-{{S}}-{{idx}}.dat'")

    h_from_file = int(match.group(1))
    s_from_file = int(match.group(2))
    idx = int(match.group(3))
    instance_name = f"data{h_from_file}-{s_from_file}-{idx}"

    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise ValueError(f"Empty benchmark file: {path}")

    first_line_parts = lines[0].split()
    num_stacks = int(first_line_parts[0])
    num_containers = int(first_line_parts[1])

    if num_stacks != s_from_file:
        raise ValueError(
            f"Stack count mismatch in {filename}: header has {num_stacks}, filename specifies {s_from_file}"
        )

    initial_height = h_from_file
    max_stack_height = initial_height + 2  # H = h + 2 Caserta standard invariant

    stacks_layout: List[tuple[int, ...]] = []
    total_found = 0

    for line in lines[1 : 1 + num_stacks]:
        parts = line.split()
        count = int(parts[0])
        containers = [int(x) for x in parts[1 : 1 + count]]
        if len(containers) != count:
            raise ValueError(f"Container count mismatch in {filename}: line says {count}, got {len(containers)}")
        stacks_layout.append(tuple(containers))
        total_found += count

    if total_found != num_containers:
        raise ValueError(
            f"Container total mismatch in {filename}: header specifies {num_containers}, found {total_found}"
        )

    return ProblemInstance(
        name=instance_name,
        num_stacks=num_stacks,
        num_containers=num_containers,
        max_stack_height=max_stack_height,
        initial_height=initial_height,
        initial_layout=tuple(stacks_layout),
    )


def load_all_caserta_instances(data_dir: Path | str) -> Dict[str, ProblemInstance]:
    """Loads all 840 Caserta benchmark instances from the given directory into a dictionary keyed by name."""
    dir_path = Path(data_dir)
    if not dir_path.is_dir():
        # Auto-extract from caserta.zip if available
        zip_candidates = [
            dir_path.with_suffix(".zip"),
            dir_path.parent / "caserta.zip",
            dir_path.parent.parent / "caserta.zip",
            Path("data/caserta.zip"),
            Path("caserta.zip"),
        ]
        extracted = False
        for z in zip_candidates:
            if z.is_file():
                import zipfile
                dir_path.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(z, "r") as zf:
                    zf.extractall(dir_path)
                extracted = True
                break
        if not extracted and not dir_path.is_dir():
            raise NotADirectoryError(f"Caserta benchmark directory not found: {dir_path}")

    instances: Dict[str, ProblemInstance] = {}
    for p in sorted(dir_path.glob("data*-*-*.dat")):
        inst = parse_caserta_file(p)
        instances[inst.name] = inst

    return instances


def load_scenario_instances(data_dir: Path | str, height: int, stacks: int) -> List[ProblemInstance]:
    """Loads the 40 instances for a specific scenario (e.g. h=3, S=3)."""
    dir_path = Path(data_dir)
    instances: List[ProblemInstance] = []
    for idx in range(1, 41):
        file_path = dir_path / f"data{height}-{stacks}-{idx}.dat"
        if not file_path.is_file():
            raise FileNotFoundError(f"Instance file missing: {file_path}")
        instances.append(parse_caserta_file(file_path))
    return instances

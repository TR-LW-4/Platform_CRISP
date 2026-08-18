"""
Caserta-style benchmark (.dat) indexing and layout loading for CRP-R.

File format (see benchmark/Caserta_dataset):
    Line 1:  S  N   — stacks (width), number of containers
    Next S lines:  H  p1 p2 … pH   — stack from left to right;
                    priorities bottom → top (1…N permutation).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.container import Container, ContainerSize, ContainerType
from core.benchmark_keys import LAYOUT_FILE_EXTRA_KEY
from core.layout_trace import trace_layout

# Project root (parent of core/)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASERTA_DIR = _PROJECT_ROOT / "benchmark" / "Caserta_dataset"

# Literature Set 1 style height labels commonly used in Caserta-style benchmarks
CASERTA_HEIGHT_WHITELIST: Tuple[int, ...] = (3, 4, 5, 6, 10)


def parse_caserta_filename(path: Path) -> Optional[Tuple[int, int, int]]:
    """
    Parse ``data{height}-{w}-{id}.dat`` (Caserta naming).

    The **first** integer is treated as **stack tier capacity / height label** in the
    dataset naming scheme; the **second** is **stack count** (width); the **third**
    is the **instance index** within that (height, w) group.

    Returns ``None`` for files that do not match (e.g. ``data_random.dat``,
    ``data3-37.dat`` with only two segments).
    """
    name = path.name
    if not name.startswith("data") or not name.endswith(".dat"):
        return None
    stem = name[len("data") : -len(".dat")]
    parts = stem.split("-")
    if len(parts) != 3:
        return None
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None


def list_caserta_heights_from_filenames(root: Optional[Path] = None) -> List[int]:
    """Distinct **first** segments (height labels) from triple ``dataH-W-ID.dat`` names."""
    root = root or DEFAULT_CASERTA_DIR
    if not root.is_dir():
        return []
    heights = set()
    for path in root.glob("data*.dat"):
        t = parse_caserta_filename(path)
        if t:
            heights.add(t[0])
    return sorted(heights)


def index_ws_by_height(root: Optional[Path] = None) -> Dict[int, List[int]]:
    """Map height → sorted distinct *w* values present in filenames."""
    root = root or DEFAULT_CASERTA_DIR
    from collections import defaultdict

    d: Dict[int, set[int]] = defaultdict(set)
    if not root.is_dir():
        return {}
    for path in root.glob("data*.dat"):
        t = parse_caserta_filename(path)
        if t:
            d[t[0]].add(t[1])
    return {h: sorted(v) for h, v in sorted(d.items())}


def collect_paths_from_hw_queue(
    root: Optional[Path],
    queue: List[Dict[str, Any]],
) -> List[Path]:
    """
    Each queue item is ``{"h": int, "ws": List[int]}``.
    All files ``data{h}-{w}-{id}.dat`` matching every pair *(h, w)* are included
    (**every** instance id / third segment).
    """
    root = root or DEFAULT_CASERTA_DIR
    if not root.is_dir() or not queue:
        return []
    seen: set = set()
    out: List[Path] = []
    for block in queue:
        h = int(block["h"])
        for w in block["ws"]:
            pattern = f"data{h}-{w}-*.dat"
            for path in sorted(root.glob(pattern)):
                t = parse_caserta_filename(path)
                if t and t[0] == h and t[1] == w:
                    key = str(path.resolve())
                    if key not in seen:
                        seen.add(key)
                        out.append(path)

    def sort_key(p: Path) -> Tuple[int, int, int]:
        t = parse_caserta_filename(p)
        return t if t else (0, 0, 0)

    out.sort(key=sort_key)
    return out


def merge_hw_queue_item(queue: List[Dict[str, Any]], h: int, ws: List[int]) -> None:
    """Append or merge *ws* into an existing block with the same *h*."""
    if not ws:
        return
    h = int(h)
    ws_set = set(int(x) for x in ws)
    for block in queue:
        if int(block["h"]) == h:
            block["ws"] = sorted(set(block["ws"]) | ws_set)
            return
    queue.append({"h": h, "ws": sorted(ws_set)})


def parse_caserta_dat(path: Path) -> Tuple[int, int, List[List[int]]]:
    """Return (S, N, stacks) where stacks[k] = priorities bottom→top."""
    text = path.read_text().strip()
    if not text:
        raise ValueError(f"empty file: {path}")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    parts0 = lines[0].split()
    if len(parts0) < 2:
        raise ValueError(f"bad header in {path}: {lines[0]!r}")
    S, N = int(parts0[0]), int(parts0[1])
    if S <= 0 or N <= 0 or len(lines) < S + 1:
        raise ValueError(f"invalid dimensions in {path}")
    stacks: List[List[int]] = []
    for i in range(1, S + 1):
        toks = lines[i].split()
        if not toks:
            raise ValueError(f"empty stack line {i} in {path}")
        H = int(toks[0])
        prios = [int(x) for x in toks[1 : 1 + H]]
        if len(prios) != H:
            raise ValueError(f"stack {i} H mismatch in {path}")
        stacks.append(prios)
    total = sum(len(s) for s in stacks)
    if total != N:
        raise ValueError(f"{path}: sum stack heights {total} != N {N}")
    return S, N, stacks


def implied_tier_capacity(w: int, n: int, stacks: List[List[int]]) -> int:
    """Tier capacity h: Set 1 formula when valid, else at least physical stack height."""
    phys = max(len(s) for s in stacks) if stacks else 1
    if w > 0 and n % w == 0:
        h_lit = n // w + 2
        if w * (h_lit - 2) == n:
            return max(h_lit, phys)
    return max(phys, 1)


def containers_for_caserta(n: int, stacks: List[List[int]]) -> Dict[int, Container]:
    """Build priority → Container map (id = priority, unique 1…n)."""
    by_pri: Dict[int, Container] = {}
    for row in stacks:
        for pri in row:
            if pri < 1 or pri > n or pri in by_pri:
                raise ValueError(f"invalid priority {pri} for n={n}")
            by_pri[pri] = Container(
                id=pri,
                group=0,
                priority=pri,
                weight=10.0,
                size=ContainerSize.TEU,
                ctype=ContainerType.STANDARD,
            )
    if set(by_pri.keys()) != set(range(1, n + 1)):
        raise ValueError("priorities are not a permutation of 1…n")
    return by_pri


def apply_caserta_file_to_yard(yard, cfg, path: Path) -> List[Container]:
    """
    Clear stacks (caller should yard.clear()), place containers from .dat file.

    Expects cfg.num_bays == S, cfg.num_rows layout matches stack keys (bay 1..S, row 1),
    cfg.max_tiers >= max stack height, cfg.num_containers == N.
    """
    trace_layout(f"apply_caserta_file_to_yard(path={path.resolve()})")
    S, N, stacks = parse_caserta_dat(path)
    if S != cfg.num_bays or cfg.num_rows < 1:
        raise ValueError(f"yard bays {cfg.num_bays} != file S={S}")
    if N != cfg.num_containers:
        raise ValueError(f"cfg.num_containers {cfg.num_containers} != file N={N}")
    by_pri = containers_for_caserta(N, stacks)
    for stack_idx, prios in enumerate(stacks):
        bay = stack_idx + 1
        row = 1
        for pri in prios:
            c = by_pri[pri]
            yard.place(bay, row, c)
    return sorted(by_pri.values(), key=lambda c: c.priority)


def problem_config_for_caserta_dat(dat_path: Path, prob_params_no_geom: Dict[str, Any]) -> "ProblemConfig":
    """
    Build ``ProblemConfig`` for one .dat file. Geometry is taken from the file;
    ``prob_params_no_geom`` should be GUI params **excluding** num_bays, num_rows,
    max_tiers, num_containers, num_groups, seed (those are overwritten here).
    """
    from core.base_problem import ProblemConfig

    S, N, stacks = parse_caserta_dat(dat_path)
    h_cap = implied_tier_capacity(S, N, stacks)

    cfg = ProblemConfig()
    for k, v in prob_params_no_geom.items():
        if k in ("num_bays", "num_rows", "max_tiers", "num_containers", "num_groups", "seed"):
            continue
        if hasattr(cfg, k) and k != "extra":
            setattr(cfg, k, v)
        else:
            cfg.extra[k] = v

    cfg.num_bays = S
    cfg.num_rows = 1
    cfg.max_tiers = h_cap
    cfg.num_containers = N
    cfg.num_groups = 1
    cfg.seed = 0
    cfg.extra[LAYOUT_FILE_EXTRA_KEY] = str(dat_path.resolve())
    trace_layout(
        "problem_config_for_caserta_dat → "
        f"S={cfg.num_bays} N={cfg.num_containers} max_tiers={cfg.max_tiers} "
        f"path={cfg.extra[LAYOUT_FILE_EXTRA_KEY]}"
    )
    return cfg

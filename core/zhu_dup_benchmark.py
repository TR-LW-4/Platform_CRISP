"""
ZhuDup benchmark loader — ``benchmark/dup_dataset/{alpha}/{H-S-N}/{file}.txt``

Directory layout::

    dup_dataset/
        alpha=0.2/
            3-6-15/
                00001.txt
                ...
        alpha=0.4/
            ...

File format (comment lines optional)::

    # converted from 00001.txt
    # ratio=0.2 seed=1
     6 15              ← S stacks, N total containers
     1  1              ← height H  then H group-IDs bottom→top
     2  3  3
     ...

Unlike Caserta/Zhu, **group IDs are not a 1…N permutation** — they can repeat
(duplicate priorities).  Each container gets a unique sequential ``id`` (1…N)
while ``priority = group = <file value>``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from core.container import Container, ContainerSize, ContainerType
from core.benchmark_keys import LAYOUT_FILE_EXTRA_KEY
from core.layout_trace import trace_layout

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ZHU_DUP_DIR = _PROJECT_ROOT / "benchmark" / "dup_dataset"

ZHU_DUP_ALPHAS: Tuple[str, ...] = ("alpha=0.2", "alpha=0.4", "alpha=0.6", "alpha=0.8")
ZHU_DUP_HEIGHT_WHITELIST: Tuple[int, ...] = (3, 4, 5, 6, 7, 10)


# ================================================================ #
#  Parsing                                                           #
# ================================================================ #

def parse_zhu_dup_folder_name(dirname: str) -> Optional[Tuple[int, int, int]]:
    """Parse ``H-S-N`` directory name. Returns ``None`` if not three integers."""
    parts = dirname.split("-")
    if len(parts) != 3:
        return None
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None


def parse_zhu_dup_file(path: Path) -> Tuple[int, int, List[List[int]]]:
    """
    Parse a ZhuDup ``.txt`` file, skipping ``#`` comment lines.

    Returns ``(S, N, stacks)`` where ``stacks[k]`` = list of group IDs
    bottom→top for stack *k*.  Group IDs may repeat.
    """
    text = path.read_text().strip()
    if not text:
        raise ValueError(f"empty file: {path}")
    lines = [
        ln.strip()
        for ln in text.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    if not lines:
        raise ValueError(f"no non-comment lines in {path}")
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
        groups = [int(x) for x in toks[1: 1 + H]]
        if len(groups) != H:
            raise ValueError(f"stack {i} H mismatch in {path}")
        stacks.append(groups)
    total = sum(len(s) for s in stacks)
    if total != N:
        raise ValueError(f"{path}: sum of stack heights {total} != N {N}")
    return S, N, stacks


# ================================================================ #
#  Container factory                                                 #
# ================================================================ #

def containers_for_dup(stacks: List[List[int]]) -> List[Container]:
    """
    Build container list for ZhuDup: unique sequential ``id`` (1…N),
    ``priority = group = <group ID from file>``.

    Containers are returned in stack order (stack 0 bottom→top, then
    stack 1, …) matching ``apply_dup_file_to_yard``.
    """
    containers: List[Container] = []
    cid = 1
    for stack_groups in stacks:
        for grp in stack_groups:
            containers.append(Container(
                id=cid,
                group=grp,
                priority=grp,
                weight=10.0,
                size=ContainerSize.TEU,
                ctype=ContainerType.STANDARD,
            ))
            cid += 1
    return containers


# ================================================================ #
#  Yard loading                                                      #
# ================================================================ #

def apply_dup_file_to_yard(yard, cfg, path: Path) -> List[Container]:
    """
    Clear stacks are assumed empty; place ZhuDup containers from *path*
    into *yard*.

    Stack index *k* → bay ``k + 1``, row ``1`` (single-row layout).
    """
    trace_layout(f"apply_dup_file_to_yard(path={path.resolve()})")
    S, N, stacks = parse_zhu_dup_file(path)
    if S != cfg.num_bays:
        raise ValueError(f"yard bays {cfg.num_bays} != file S={S}")
    if N != cfg.num_containers:
        raise ValueError(
            f"cfg.num_containers {cfg.num_containers} != file N={N}"
        )
    containers = containers_for_dup(stacks)
    c_iter = iter(containers)
    for stack_idx, group_list in enumerate(stacks):
        bay = stack_idx + 1
        row = 1
        for _ in group_list:
            c = next(c_iter)
            yard.place(bay, row, c)
    return containers


# ================================================================ #
#  ProblemConfig builder                                             #
# ================================================================ #

def problem_config_for_zhu_dup_txt(
    path: Path,
    prob_params_no_geom: Dict[str, Any],
):
    """
    Build ``ProblemConfig`` for one ZhuDup file.

    Geometry is read from the file itself.  Tier capacity = max(H_label + 2,
    physical height) when the parent folder name is a valid ``H-S-N`` string.
    ``num_groups`` is derived from the distinct group IDs in the file.
    """
    from core.base_problem import ProblemConfig

    S, N, stacks = parse_zhu_dup_file(path)
    phys = max(len(s) for s in stacks) if stacks else 1

    parsed = parse_zhu_dup_folder_name(path.parent.name)
    if parsed:
        h_label = parsed[0]
        h_cap = max(h_label + 2, phys)
    else:
        h_cap = max(phys, 1)

    all_groups: Set[int] = set()
    for s in stacks:
        all_groups.update(s)

    cfg = ProblemConfig()
    for k, v in prob_params_no_geom.items():
        if k in (
            "num_bays", "num_rows", "max_tiers",
            "num_containers", "num_groups", "seed",
        ):
            continue
        if hasattr(cfg, k) and k != "extra":
            setattr(cfg, k, v)
        else:
            cfg.extra[k] = v

    cfg.num_bays = S
    cfg.num_rows = 1
    cfg.max_tiers = h_cap
    cfg.num_containers = N
    cfg.num_groups = len(all_groups)
    cfg.seed = 0
    cfg.extra[LAYOUT_FILE_EXTRA_KEY] = str(path.resolve())
    trace_layout(
        f"problem_config_for_zhu_dup_txt → S={S} N={N} "
        f"groups={cfg.num_groups} max_tiers={h_cap} path={path}"
    )
    return cfg


# ================================================================ #
#  Index / queue helpers (mirrors zhu_benchmark style)              #
# ================================================================ #

def index_sn_pairs_by_height_dup(
    root: Optional[Path] = None,
    alpha: str = "alpha=0.2",
) -> Dict[int, List[Tuple[int, int]]]:
    """Map H → sorted *(S, N)* pairs present in ``{root}/{alpha}/``."""
    root = root or DEFAULT_ZHU_DUP_DIR
    alpha_root = root / alpha
    acc: Dict[int, Set[Tuple[int, int]]] = {}
    if not alpha_root.is_dir():
        return {}
    for p in alpha_root.iterdir():
        if not p.is_dir():
            continue
        t = parse_zhu_dup_folder_name(p.name)
        if t is None:
            continue
        h, s, n = t
        acc.setdefault(h, set()).add((s, n))
    return {h: sorted(v) for h, v in sorted(acc.items())}


def collect_paths_from_zhu_dup_queue(
    root: Optional[Path],
    queue: List[Dict[str, Any]],
    alpha: str = "alpha=0.2",
) -> List[Path]:
    """
    Collect ``.txt`` paths from ``{root}/{alpha}/{H}-{S}-{N}/`` matching *queue*.

    Each queue item: ``{"h": int, "sn_pairs": [[s, n], ...]}``.
    """
    root = root or DEFAULT_ZHU_DUP_DIR
    alpha_root = root / alpha
    if not alpha_root.is_dir() or not queue:
        return []
    seen: set = set()
    out: List[Path] = []
    for block in queue:
        h = int(block["h"])
        for pair in block["sn_pairs"]:
            s, n = int(pair[0]), int(pair[1])
            sub = alpha_root / f"{h}-{s}-{n}"
            if not sub.is_dir():
                continue
            for p in sorted(sub.glob("*.txt")):
                key = str(p.resolve())
                if key not in seen:
                    seen.add(key)
                    out.append(p)

    def _sort_key(p: Path) -> Tuple[int, int, int, str]:
        parent = parse_zhu_dup_folder_name(p.parent.name)
        try:
            inst = int(p.stem)
        except ValueError:
            inst = 0
        if parent:
            return parent[0], parent[1], parent[2], f"{inst:06d}"
        return 0, 0, 0, p.name

    out.sort(key=_sort_key)
    return out


def merge_zhu_dup_queue_item(
    queue: List[Dict[str, Any]],
    h: int,
    sn_pairs: List[Tuple[int, int]],
) -> None:
    """Append or merge *(S, N)* pairs into an existing block with the same *h*."""
    if not sn_pairs:
        return
    h = int(h)
    pair_set = {(int(a), int(b)) for a, b in sn_pairs}
    for block in queue:
        if int(block["h"]) == h:
            existing = {(int(p[0]), int(p[1])) for p in block["sn_pairs"]}
            block["sn_pairs"] = sorted(existing | pair_set)
            return
    queue.append({"h": h, "sn_pairs": sorted(pair_set)})

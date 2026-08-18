"""
Zhu-style benchmark: one subdirectory per instance size ``H-S-N``, each with many ``.txt``
layouts (same body format as Caserta ``.dat``).

Directory layout (see ``benchmark/Zhu_dataset``)::

    Zhu_dataset/
        5-8-39/
            06101.txt
            ...

The first segment **H** is the literature tier label (whitelist includes **7** in
addition to the usual Caserta-style set); **S** is stack count (bays); **N** is
number of containers. Each folder typically
holds 100 instance files; all ``*.txt`` in a selected folder are included.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from core.benchmarks.caserta import problem_config_for_caserta_dat

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ZHU_DIR = _PROJECT_ROOT / "benchmark" / "Zhu_dataset"

# Allowed first-segment *H* labels in ``Zhu_dataset`` (includes 7; Caserta list has no 7)
ZHU_HEIGHT_WHITELIST: Tuple[int, ...] = (3, 4, 5, 6, 7, 10)


def parse_zhu_folder_name(dirname: str) -> Optional[Tuple[int, int, int]]:
    """Parse ``H-S-N`` (three integers). Returns ``None`` if invalid."""
    parts = dirname.split("-")
    if len(parts) != 3:
        return None
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None


def index_sn_pairs_by_height(root: Optional[Path] = None) -> Dict[int, List[Tuple[int, int]]]:
    """Map height label *H* → sorted distinct *(S, N)* pairs present as subdir names."""
    root = root or DEFAULT_ZHU_DIR
    acc: Dict[int, Set[Tuple[int, int]]] = {}
    if not root.is_dir():
        return {}
    for path in root.iterdir():
        if not path.is_dir():
            continue
        t = parse_zhu_folder_name(path.name)
        if t is None:
            continue
        h, s, n = t
        acc.setdefault(h, set()).add((s, n))
    return {h: sorted(v) for h, v in sorted(acc.items())}


def merge_zhu_queue_item(
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
            merged = existing | pair_set
            block["sn_pairs"] = sorted(merged)
            return
    queue.append({"h": h, "sn_pairs": sorted(pair_set)})


def collect_paths_from_zhu_queue(
    root: Optional[Path],
    queue: List[Dict[str, Any]],
) -> List[Path]:
    """
    Each queue item is ``{"h": int, "sn_pairs": List[[s, n], ...]}``.
    For every *(h, s, n)*, all ``*.txt`` files in ``{root}/{h}-{s}-{n}/`` are included.
    """
    root = root or DEFAULT_ZHU_DIR
    if not root.is_dir() or not queue:
        return []
    seen: set = set()
    out: List[Path] = []
    for block in queue:
        h = int(block["h"])
        for pair in block["sn_pairs"]:
            s, n = int(pair[0]), int(pair[1])
            sub = root / f"{h}-{s}-{n}"
            if not sub.is_dir():
                continue
            for path in sorted(sub.glob("*.txt")):
                key = str(path.resolve())
                if key not in seen:
                    seen.add(key)
                    out.append(path)

    def sort_key(p: Path) -> Tuple[int, int, int, str]:
        parent = parse_zhu_folder_name(p.parent.name)
        base = p.stem
        try:
            inst = int(base)
        except ValueError:
            inst = 0
        if parent:
            return parent[0], parent[1], parent[2], f"{inst:06d}"
        return 0, 0, 0, p.name

    out.sort(key=sort_key)
    return out


def problem_config_for_zhu_txt(dat_path: Path, prob_params_no_geom: Dict[str, Any]):
    """
    Build ``ProblemConfig`` from a Zhu ``.txt`` file (same layout encoding as Caserta).

    When the file lives under a subdirectory named ``H-S-N``, tier capacity is at least
    ``H + 2`` (literature height label **H**), matching Caserta Set-1 style slack. This
    avoids an overly tight ``max_tiers`` when ``implied_tier_capacity`` falls back to the
    physical stack height only (e.g. ``N % S != 0``).
    """
    from core.layout_trace import trace_layout

    cfg = problem_config_for_caserta_dat(dat_path, prob_params_no_geom)
    parsed = parse_zhu_folder_name(dat_path.parent.name)
    if parsed is None:
        return cfg
    h_label, _, _ = parsed
    floor = h_label + 2
    if cfg.max_tiers < floor:
        trace_layout(
            "problem_config_for_zhu_txt → bump max_tiers "
            f"{cfg.max_tiers} → {floor} (folder label H={h_label}, use H+2)"
        )
        cfg.max_tiers = floor
    return cfg

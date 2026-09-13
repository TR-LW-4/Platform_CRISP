"""Index and load the official Jovanović BRLP ``.pro`` benchmark set."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from core.benchmark_keys import LAYOUT_FILE_EXTRA_KEY


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STOW_DIR = _PROJECT_ROOT / "benchmark" / "crp_stow" / "Data" / "Gen"

_STOW_RE = re.compile(
    r"^Bay-(?P<a>\d+)-(?P<vs>\d+)-(?P<ys>\d+)-(?P<yt>\d+)"
    r"_(?P<seed>\d+)\.pro$",
    re.IGNORECASE,
)

StowClass = Tuple[int, int, int]  # vessel stacks, yard stacks, yard tiers


def parse_stow_filename(path: str | Path) -> Tuple[int, int, int, int, int]:
    """Return ``(A, VS, YS, YT, seed)`` from an official filename."""
    match = _STOW_RE.match(Path(path).name)
    if match is None:
        raise ValueError(f"not an official CRP-Stow filename: {Path(path).name}")
    return tuple(
        int(match.group(key))
        for key in ("a", "vs", "ys", "yt", "seed")
    )


def index_classes_by_min_vessel_height(
    root: Path = DEFAULT_STOW_DIR,
) -> Dict[int, List[StowClass]]:
    """Discover available ``A -> [(VS, YS, YT), ...]`` benchmark classes."""
    indexed: Dict[int, set[StowClass]] = {}
    if not root.is_dir():
        return {}
    for path in root.glob("Bay-*-*-*-*_*.pro"):
        try:
            a, vs, ys, yt, _seed = parse_stow_filename(path)
        except ValueError:
            continue
        indexed.setdefault(a, set()).add((vs, ys, yt))
    return {
        a: sorted(classes, key=lambda item: (item[0], item[1], item[2]))
        for a, classes in sorted(indexed.items())
    }


def _normalize_classes(raw: Sequence[Any]) -> List[StowClass]:
    classes: set[StowClass] = set()
    for item in raw:
        if isinstance(item, Mapping):
            classes.add((
                int(item["vs"]),
                int(item["ys"]),
                int(item["yt"]),
            ))
        else:
            classes.add((int(item[0]), int(item[1]), int(item[2])))
    return sorted(classes)


def collect_paths_from_stow_queue(
    root: Path,
    queue: Sequence[Mapping[str, Any]],
) -> List[Path]:
    """Resolve selected benchmark classes to deterministically sorted files."""
    paths: List[Path] = []
    for block in queue:
        a = int(block["h"])
        for vs, ys, yt in _normalize_classes(block.get("stow_classes", [])):
            matching = list(root.glob(f"Bay-{a}-{vs}-{ys}-{yt}_*.pro"))
            matching.sort(key=lambda path: parse_stow_filename(path)[4])
            paths.extend(matching)
    return paths


def problem_config_for_stow_pro(
    pro_path: Path,
    prob_params_no_geom: Mapping[str, Any],
):
    """Build a ``ProblemConfig`` whose geometry comes from one ``.pro`` file."""
    from core.base_problem import ProblemConfig
    from problems.CRP_Stow import parse_pro_file

    data = parse_pro_file(pro_path)
    cfg = ProblemConfig()
    geometry = {
        "num_bays",
        "num_rows",
        "max_tiers",
        "num_containers",
        "num_groups",
        "seed",
    }
    for key, value in prob_params_no_geom.items():
        if key in geometry:
            continue
        if hasattr(cfg, key) and key != "extra":
            setattr(cfg, key, value)
        else:
            cfg.extra[key] = value

    cfg.num_bays = int(data["YS"])
    cfg.num_rows = 1
    cfg.max_tiers = int(data["YT"])
    cfg.num_containers = int(data["N"])
    cfg.num_groups = int(data["VS"])
    try:
        cfg.seed = parse_stow_filename(pro_path)[4]
    except ValueError:
        cfg.seed = 0
    cfg.extra[LAYOUT_FILE_EXTRA_KEY] = str(pro_path.resolve())
    return cfg

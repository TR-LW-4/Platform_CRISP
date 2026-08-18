"""
Aggregate saved benchmark runs by size class (mean / std).

Does not touch algorithm logic: consumes result JSON written by ``save_run``.
Caserta files use ``data{H}-{w}-{id}.dat``; Zhu uses parent folder ``H-S-N``.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Skip non-scalar / non-numeric noise, and low-value display metrics.
# retrievals ≈ N (often constant); total_moves = relocations + retrievals.
SKIP_METRIC_KEYS = frozenset({
    "optimal_proven",
    "progress",
    "retrievals",
    "total_moves",
    "steps",  # usually ≈ relocations for CRP-R step heuristics
})
# Back-compat alias
_SKIP_METRIC_KEYS = SKIP_METRIC_KEYS
_LAYOUT_KEYS = ("layout_file_path", "caserta_dat_path")


def _parse_caserta_filename(path: Path) -> Optional[Tuple[int, int, int]]:
    """Parse ``data{H}-{w}-{id}.dat`` → (H, w, id)."""
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


def _parse_zhu_folder_name(dirname: str) -> Optional[Tuple[int, int, int]]:
    """Parse ``H-S-N`` folder name."""
    parts = dirname.split("-")
    if len(parts) != 3:
        return None
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None


def _class_from_layout_path(path: Path) -> Optional[Tuple[str, str, Dict[str, int]]]:
    """
    Return ``(source, label, dims)`` or ``None``.

    * Caserta: source=caserta, label=\"3×3\", dims={h, w}
    * Zhu:     source=zhu,     label=\"5-8-39\", dims={h, s, n}
    """
    parsed = _parse_caserta_filename(path)
    if parsed is not None:
        h, w, _inst = parsed
        return "caserta", f"{h}×{w}", {"h": h, "w": w}

    folder = _parse_zhu_folder_name(path.parent.name)
    if folder is not None:
        h, s, n = folder
        return "zhu", f"{h}-{s}-{n}", {"h": h, "s": s, "n": n}
    return None


def _layout_path_from_run(data: Mapping[str, Any]) -> Optional[Path]:
    pc = data.get("prob_config") or {}
    if not isinstance(pc, Mapping):
        return None
    raw = None
    for key in _LAYOUT_KEYS:
        if pc.get(key):
            raw = pc.get(key)
            break
    if raw is None:
        extra = pc.get("extra") or {}
        if isinstance(extra, Mapping):
            for key in _LAYOUT_KEYS:
                if extra.get(key):
                    raw = extra.get(key)
                    break
    if not raw:
        return None
    return Path(str(raw))


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)  # sample std
    return mean, math.sqrt(var)


def summarize_run_dicts(runs: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """
    Group runs by benchmark class and compute per-metric mean/std.

    Returns
    -------
    {
      "classes": [
        {
          "source": "caserta",
          "label": "3×3",
          "h": 3, "w": 3,
          "n": 40,
          "metrics": {
            "relocations": {"mean": ..., "std": ...},
            ...
          }
        },
        ...
      ],
      "ungrouped": 0,
      "total_runs": 40,
    }
    """
    buckets: Dict[Tuple[str, str], Dict[str, Any]] = {}
    metric_values: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    ungrouped = 0
    total = 0

    for data in runs:
        total += 1
        layout = _layout_path_from_run(data)
        if layout is None:
            ungrouped += 1
            continue
        classified = _class_from_layout_path(layout)
        if classified is None:
            ungrouped += 1
            continue
        source, label, dims = classified
        key = (source, label)
        if key not in buckets:
            buckets[key] = {
                "source": source,
                "label": label,
                **dims,
            }
        metrics = data.get("metrics") or {}
        if not isinstance(metrics, Mapping):
            continue
        for name, raw in metrics.items():
            if name in _SKIP_METRIC_KEYS:
                continue
            if isinstance(raw, bool):
                continue
            if isinstance(raw, (int, float)):
                metric_values[key][str(name)].append(float(raw))

    classes: List[Dict[str, Any]] = []
    for key in sorted(buckets.keys(), key=lambda k: (k[0], k[1])):
        meta = dict(buckets[key])
        per_metric = metric_values.get(key, {})
        n = max((len(v) for v in per_metric.values()), default=0)
        out_metrics: Dict[str, Dict[str, float]] = {}
        for mname in sorted(per_metric.keys()):
            mean, std = _mean_std(per_metric[mname])
            out_metrics[mname] = {
                "mean": round(mean, 6),
                "std": round(std, 6),
            }
        meta["n"] = n
        meta["metrics"] = out_metrics
        classes.append(meta)

    return {
        "classes": classes,
        "ungrouped": ungrouped,
        "total_runs": total,
    }


def summarize_result_files(paths: Sequence[str]) -> Dict[str, Any]:
    """Load result JSON files and return a class-wise summary."""
    runs: List[Dict[str, Any]] = []
    for p in paths:
        path = Path(p)
        if not path.is_file():
            continue
        try:
            with open(path, encoding="utf-8") as f:
                runs.append(json.load(f))
        except Exception:
            continue
    return summarize_run_dicts(runs)

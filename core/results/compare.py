"""
Multi-algorithm comparison over saved result folders.

Builds PlatEMO-style class × algorithm tables (mean ± std) from
``results/<problem>/<algo_folder>/*.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

from core.results.aggregate import (
    SKIP_METRIC_KEYS,
    _layout_path_from_run,
    summarize_run_dicts,
)

_DEFAULT_DIR = Path(__file__).resolve().parents[2] / "results"

# Metrics where lower mean is better (highlight min).
_LOWER_IS_BETTER = frozenset({
    "relocations",
    "crane_time",
    "time",
    "steps",
    "moves",
    "shifters",
    "total_moves",
    "expected_relocations",
})


def _safe_name(name: str) -> str:
    return name.replace(" ", "_").replace("/", "-")


def _iter_run_files(folder: Path) -> List[Path]:
    files = [
        p
        for p in folder.glob("*.json")
        if p.is_file() and not p.name.startswith("batch_summary")
    ]
    # Newest first by filename timestamp prefix when present.
    return sorted(files, key=lambda p: p.name, reverse=True)


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def list_result_folders(
    problem: str,
    base_dir: Path = _DEFAULT_DIR,
) -> List[Dict[str, Any]]:
    """List algorithm folders under ``results/<problem>/`` with run counts."""
    root = base_dir / _safe_name(problem)
    if not root.is_dir():
        return []
    out: List[Dict[str, Any]] = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        files = _iter_run_files(folder)
        if not files:
            continue
        algo_name = folder.name.replace("_", " ")
        latest_ts = ""
        category = ""
        sample = _load_json(files[0])
        if sample:
            algo_name = str(sample.get("algorithm") or algo_name)
            category = str(sample.get("category") or "")
            latest_ts = str(sample.get("timestamp") or "")
        out.append({
            "problem": problem,
            "algorithm": algo_name,
            "folder": str(folder),
            "folder_name": folder.name,
            "category": category,
            "n_runs": len(files),
            "latest": latest_ts,
        })
    # Prefer most recent activity first.
    out.sort(key=lambda r: r.get("latest") or "", reverse=True)
    return out


def _folder_for_algorithm(
    problem: str,
    algorithm: str,
    base_dir: Path,
) -> Optional[Path]:
    root = base_dir / _safe_name(problem)
    if not root.is_dir():
        return None
    preferred = root / _safe_name(algorithm)
    if preferred.is_dir() and _iter_run_files(preferred):
        return preferred
    # Fallback: match algorithm field inside JSON.
    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        for path in _iter_run_files(folder)[:5]:
            data = _load_json(path)
            if data and str(data.get("algorithm")) == algorithm:
                return folder
    return None


def _dedup_latest_per_layout(runs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for run in runs:
        layout = _layout_path_from_run(run)
        key = str(layout.resolve()) if layout is not None else f"__nolayout__{id(run)}"
        ts = str(run.get("timestamp") or "")
        prev = best.get(key)
        if prev is None or ts >= str(prev.get("timestamp") or ""):
            best[key] = dict(run)
    return list(best.values())


def load_algorithm_runs(
    problem: str,
    algorithm: str,
    *,
    limit: Optional[int] = None,
    dedup: bool = True,
    base_dir: Path = _DEFAULT_DIR,
) -> List[Dict[str, Any]]:
    folder = _folder_for_algorithm(problem, algorithm, base_dir)
    if folder is None:
        return []
    paths = _iter_run_files(folder)
    if limit is not None and limit > 0:
        paths = paths[: int(limit)]
    runs: List[Dict[str, Any]] = []
    for path in paths:
        data = _load_json(path)
        if data is None:
            continue
        data["_file"] = str(path)
        runs.append(data)
    if dedup:
        runs = _dedup_latest_per_layout(runs)
    return runs


def compare_algorithms(
    problem: str,
    algorithms: Sequence[str],
    *,
    metric: str = "relocations",
    limit: Optional[int] = None,
    dedup: bool = True,
    base_dir: Path = _DEFAULT_DIR,
) -> Dict[str, Any]:
    """
    Compare multiple algorithms on class-wise mean±std of *metric*.

    Loads all run JSON files under each algorithm folder by default (no
    newest-N cutoff). With PlatEMO-style overwrite filenames, each instance
    typically has one file; ``dedup`` still collapses legacy timestamped
    duplicates.

    Returns a pivot table:
      rows: benchmark classes
      columns: algorithms → {n, mean, std, best}
    """
    algos = [a for a in algorithms if a]
    per_algo_summary: Dict[str, Dict[str, Any]] = {}
    per_algo_n_files: Dict[str, int] = {}
    metric_keys: Set[str] = set()

    for algo in algos:
        runs = load_algorithm_runs(
            problem, algo, limit=limit, dedup=dedup, base_dir=base_dir,
        )
        per_algo_n_files[algo] = len(runs)
        summary = summarize_run_dicts(runs)
        per_algo_summary[algo] = summary
        for row in summary.get("classes") or []:
            metric_keys.update((row.get("metrics") or {}).keys())

    preferred = [
        "relocations", "crane_time", "time", "moves", "shifters",
    ]
    metric_keys = {m for m in metric_keys if m not in SKIP_METRIC_KEYS}
    available_metrics = [m for m in preferred if m in metric_keys]
    for m in sorted(metric_keys):
        if m not in available_metrics:
            available_metrics.append(m)
    if metric not in available_metrics and available_metrics:
        metric = available_metrics[0]

    # Union of class labels preserving caserta/zhu sort via summarize order.
    class_order: List[str] = []
    class_meta: Dict[str, Dict[str, Any]] = {}
    for algo in algos:
        for row in (per_algo_summary[algo].get("classes") or []):
            label = str(row.get("label") or "?")
            if label not in class_meta:
                class_order.append(label)
                class_meta[label] = {
                    "label": label,
                    "source": row.get("source"),
                    "h": row.get("h"),
                    "w": row.get("w"),
                    "s": row.get("s"),
                    "n_dim": row.get("n"),
                }

    lower_better = metric in _LOWER_IS_BETTER
    rows: List[Dict[str, Any]] = []
    for label in class_order:
        cells: Dict[str, Any] = {}
        means: Dict[str, float] = {}
        for algo in algos:
            cell = {"n": 0, "mean": None, "std": None, "best": False}
            for crow in (per_algo_summary[algo].get("classes") or []):
                if str(crow.get("label")) != label:
                    continue
                agg = (crow.get("metrics") or {}).get(metric) or {}
                cell["n"] = int(crow.get("n") or 0)
                if "mean" in agg:
                    cell["mean"] = float(agg["mean"])
                    means[algo] = float(agg["mean"])
                if "std" in agg:
                    cell["std"] = float(agg["std"])
                break
            cells[algo] = cell

        if means:
            best_val = min(means.values()) if lower_better else max(means.values())
            for algo, val in means.items():
                if abs(val - best_val) <= 1e-9:
                    cells[algo]["best"] = True

        rows.append({
            **class_meta[label],
            "cells": cells,
        })

    return {
        "problem": problem,
        "metric": metric,
        "lower_is_better": lower_better,
        "algorithms": list(algos),
        "available_metrics": available_metrics,
        "dedup": dedup,
        "limit": limit,
        "n_files": per_algo_n_files,
        "rows": rows,
        "per_algorithm": {
            algo: {
                "total_runs": per_algo_summary[algo].get("total_runs", 0),
                "ungrouped": per_algo_summary[algo].get("ungrouped", 0),
                "n_classes": len(per_algo_summary[algo].get("classes") or []),
            }
            for algo in algos
        },
    }


def delete_result_file(path: str, base_dir: Path = _DEFAULT_DIR) -> bool:
    """Delete one result JSON under results/. Returns True if removed."""
    candidate = Path(path).resolve()
    root = base_dir.resolve()
    if root not in candidate.parents:
        raise ValueError("Path is outside the results directory")
    if candidate.suffix != ".json" or not candidate.is_file():
        raise FileNotFoundError(str(candidate))
    candidate.unlink()
    return True


def delete_algorithm_folder(
    problem: str,
    algorithm: str,
    base_dir: Path = _DEFAULT_DIR,
) -> int:
    """Delete all run JSON files in an algorithm folder. Returns count removed."""
    folder = _folder_for_algorithm(problem, algorithm, base_dir)
    if folder is None:
        return 0
    removed = 0
    for path in _iter_run_files(folder):
        try:
            path.unlink()
            removed += 1
        except FileNotFoundError:
            pass
    # Remove empty batch_summary sidecars too.
    for path in folder.glob("batch_summary_*.json"):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    try:
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()
    except OSError:
        pass
    return removed

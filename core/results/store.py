"""
Result persistence layer.

Every completed run is saved as a JSON file under::

  results/<problem>/<algorithm>/<stable_stem>.json

Naming follows a PlatEMO-style key (problem/instance identity), so re-running
the same algorithm on the same layout **overwrites** the previous file instead
of accumulating timestamped copies.

Examples
--------
* Caserta ``data3-3-7.dat`` → ``caserta_3x3_07.json``
* Zhu folder ``5-8-39`` / ``06101.txt`` → ``zhu_5-8-39_06101.json``
* No layout (random config) → ``random_seed<N>.json``

File format
-----------
{
  "problem":    "CRP-R",
  "algorithm":  "Caserta (2012) HEUR",
  "category":   "Heuristic",
  "seed":       0,
  "timestamp":  "2026-04-21T14:30:00",
  "prob_config": {...},
  "algo_config": {...},
  "metrics":    {"relocations": 5.0, ...},
  "history":    [...],
  "moves":      [...]
}
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from core.benchmark_keys import layout_path_from_extra
from core.move_export import moves_from_final_record

# Default results directory (project root, i.e. the parent of core/)
_DEFAULT_DIR = Path(__file__).resolve().parents[2] / "results"

_CASERTA_RE = re.compile(
    r"^data(?P<h>\d+)-(?P<w>\d+)-(?P<iid>\d+)\.dat$",
    re.IGNORECASE,
)
_ZHU_FOLDER_RE = re.compile(r"^(?P<h>\d+)-(?P<s>\d+)-(?P<n>\d+)$")


def _safe_token(name: str) -> str:
    return name.replace(" ", "_").replace("/", "-")


def stable_result_stem(
    prob_config: Optional[Mapping[str, Any]],
    seed: int = 0,
) -> str:
    """
    Build a stable, overwrite-friendly filename stem from problem config.

    Prefer layout identity (Caserta / Zhu); fall back to ``random_seed<N>``.
    """
    extra: Mapping[str, Any] = {}
    if isinstance(prob_config, Mapping):
        raw_extra = prob_config.get("extra")
        if isinstance(raw_extra, Mapping):
            extra = raw_extra
        else:
            # Call sites often pass a flat dict already containing layout keys.
            extra = prob_config

    layout_raw = layout_path_from_extra(dict(extra)) if extra else None
    if not layout_raw and isinstance(prob_config, Mapping):
        layout_raw = layout_path_from_extra(dict(prob_config))

    if layout_raw:
        path = Path(str(layout_raw))
        m = _CASERTA_RE.match(path.name)
        if m:
            h = int(m.group("h"))
            w = int(m.group("w"))
            iid = int(m.group("iid"))
            return f"caserta_{h}x{w}_{iid:02d}"

        folder_m = _ZHU_FOLDER_RE.match(path.parent.name)
        if folder_m:
            h = int(folder_m.group("h"))
            s = int(folder_m.group("s"))
            n = int(folder_m.group("n"))
            stem = path.stem
            safe_stem = re.sub(r"[^\w.\-]+", "_", stem)
            return f"zhu_{h}-{s}-{n}_{safe_stem}"

        # Unknown layout filename: still stable per path basename.
        safe = re.sub(r"[^\w.\-]+", "_", path.stem)
        return f"layout_{safe}"

    return f"random_seed{int(seed)}"


def _result_path(
    problem: str,
    algorithm: str,
    seed: int,
    prob_config: Optional[Mapping[str, Any]] = None,
    base_dir: Path = _DEFAULT_DIR,
) -> Path:
    safe_algo = _safe_token(algorithm)
    safe_prob = _safe_token(problem)
    folder = base_dir / safe_prob / safe_algo
    folder.mkdir(parents=True, exist_ok=True)
    stem = stable_result_stem(prob_config, seed=seed)
    return folder / f"{stem}.json"


def save_run(
    problem: str,
    algorithm: str,
    category: str,
    seed: int,
    prob_config: Dict[str, Any],
    algo_config: Dict[str, Any],
    metrics: Dict[str, float],
    history: List[Dict],
    base_dir: Path = _DEFAULT_DIR,
    moves: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    """Save one completed run to disk (overwrites same instance key). Returns path."""
    record: Dict[str, Any] = {
        "problem": problem,
        "algorithm": algorithm,
        "category": category,
        "seed": seed,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "prob_config": prob_config,
        "algo_config": algo_config,
        "metrics": metrics,
        "history": history,
    }
    if moves:
        record["moves"] = moves
    path = _result_path(problem, algorithm, seed, prob_config, base_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    return path


def moves_from_records(records: List[Any]) -> Optional[List[Dict[str, Any]]]:
    """Extract ``moves`` from the last progress record, if any."""
    if not records:
        return None
    return moves_from_final_record(records[-1])


def load_run(path: Path) -> Dict:
    """Load a single result file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_saved_runs(base_dir: Path = _DEFAULT_DIR) -> List[Dict]:
    """
    Scan the results directory and return metadata for all saved runs.
    Returns list of dicts sorted by timestamp (newest first).
    """
    runs = []
    if not base_dir.exists():
        return runs
    for json_file in sorted(base_dir.rglob("*.json"), reverse=True):
        if json_file.name.startswith("batch_summary"):
            continue
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
            runs.append({
                "file": str(json_file),
                "problem": data.get("problem", "?"),
                "algorithm": data.get("algorithm", "?"),
                "category": data.get("category", "?"),
                "seed": data.get("seed", 0),
                "timestamp": data.get("timestamp", ""),
                "metrics": data.get("metrics", {}),
                "n_steps": len(data.get("history", [])),
            })
        except Exception:
            continue
    runs.sort(key=lambda r: str(r.get("timestamp") or ""), reverse=True)
    return runs


def load_runs_for_compare(
    file_paths: List[str],
) -> List[Dict]:
    """Load multiple result files for the Compare tab."""
    results = []
    for fp in file_paths:
        try:
            results.append(load_run(Path(fp)))
        except Exception:
            continue
    return results


def delete_run(path: str) -> None:
    """Delete a saved result file."""
    try:
        os.remove(path)
    except FileNotFoundError:
        pass

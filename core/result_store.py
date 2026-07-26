"""
Result persistence layer.

Every completed run (Test or Experiment) is saved as a JSON file under
  results/<problem>/<algorithm>/<timestamp>_seed<N>.json

The Compare tab can load any combination of saved files and overlay them.

File format
-----------
{
  "problem":    "CRP-Stow",
  "algorithm":  "Caserta (2012) HEUR",
  "category":   "Heuristic",
  "seed":       0,
  "timestamp":  "2026-04-21T14:30:00",
  "prob_config": {...},
  "algo_config": {...},
  "metrics":    {"shifters": 12.0, "time": 12.0, ...},
  "history":    [          ← one entry per report_interval step
    {"step": 10, "metric": 15.0, "metrics": {...}},
    {"step": 20, "metric": 13.0, "metrics": {...}},
    ...
  ]
}
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Default results directory (next to this file's package root)
_DEFAULT_DIR = Path(__file__).parent.parent / "results"


def _result_path(
    problem:   str,
    algorithm: str,
    seed:      int,
    base_dir:  Path = _DEFAULT_DIR,
) -> Path:
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_algo = algorithm.replace(" ", "_").replace("/", "-")
    safe_prob = problem.replace(" ", "_").replace("/", "-")
    folder    = base_dir / safe_prob / safe_algo
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{ts}_seed{seed}.json"


def save_run(
    problem:     str,
    algorithm:   str,
    category:    str,
    seed:        int,
    prob_config: Dict[str, Any],
    algo_config: Dict[str, Any],
    metrics:     Dict[str, float],
    history:     List[Dict],
    base_dir:    Path = _DEFAULT_DIR,
) -> Path:
    """Save one completed run to disk. Returns the saved file path."""
    record = {
        "problem":     problem,
        "algorithm":   algorithm,
        "category":    category,
        "seed":        seed,
        "timestamp":   datetime.now().isoformat(timespec="seconds"),
        "prob_config": prob_config,
        "algo_config": algo_config,
        "metrics":     metrics,
        "history":     history,
    }
    path = _result_path(problem, algorithm, seed, base_dir)
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    return path


def load_run(path: Path) -> Dict:
    """Load a single result file."""
    with open(path) as f:
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
        try:
            with open(json_file) as f:
                data = json.load(f)
            runs.append({
                "file":      str(json_file),
                "problem":   data.get("problem",   "?"),
                "algorithm": data.get("algorithm", "?"),
                "category":  data.get("category",  "?"),
                "seed":      data.get("seed",       0),
                "timestamp": data.get("timestamp",  ""),
                "metrics":   data.get("metrics",    {}),
                "n_steps":   len(data.get("history", [])),
            })
        except Exception:
            continue
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

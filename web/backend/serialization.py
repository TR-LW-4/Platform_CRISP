"""JSON-safe representations for multiprocessing progress messages."""

from __future__ import annotations

import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        try:
            return json_safe(value.tolist())
        except (TypeError, ValueError):
            pass
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return str(value)


def snapshot_to_json(snapshot: Optional[Dict]) -> Optional[Dict[str, Any]]:
    if not snapshot:
        return None
    result = {
        str(key): json_safe(value)
        for key, value in snapshot.items()
        if key != "yard"
    }
    yard = snapshot.get("yard")
    if isinstance(yard, dict):
        stacks = []
        for position, containers in yard.items():
            if isinstance(position, tuple) and len(position) == 2:
                bay, row = position
            else:
                text = str(position).strip("()[]")
                pieces = [piece.strip() for piece in text.split(",")]
                bay, row = pieces[:2] if len(pieces) >= 2 else (0, 0)
            stacks.append({
                "bay": int(bay),
                "row": int(row),
                "containers": json_safe(containers),
            })
        result["yard"] = sorted(stacks, key=lambda item: (item["bay"], item["row"]))
    else:
        result["yard"] = []
    return result


def progress_to_json(record: Any, sequence: int) -> Dict[str, Any]:
    if isinstance(record, dict):
        getter = record.get
    else:
        getter = lambda name, default=None: getattr(record, name, default)
    metric = getter("metric", 0.0)
    best_metric = getter("best_metric", metric)
    return {
        "sequence": sequence,
        "step": int(getter("step", 0)),
        "metric": json_safe(metric),
        "metrics": json_safe(getter("metrics", {})),
        "best_metric": json_safe(best_metric),
        "progress": max(0.0, min(1.0, float(getter("progress", 0.0)))),
        "yard_snapshot": snapshot_to_json(getter("yard_snapshot")),
        "extra": json_safe(getter("extra", {})),
    }

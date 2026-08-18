"""
Export yard move history to a JSON-friendly plan and optional crane time.

Used by env-step algorithms (e.g. Caserta HEUR) so runs can persist a full
relocate/retrieve sequence and compute Lee & Lee–style crane working time.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from core.objectives import KinematicsModel, compute_crane_time
from core.plan import Movement, RelocationPlan


def plan_from_moves(moves: Iterable[Any]) -> RelocationPlan:
    """
    Build a RelocationPlan from yard.Move-like objects
    (``kind``, ``container_id``, ``src``, ``dst``).
    """
    plan = RelocationPlan()
    for move in moves:
        kind = getattr(move, "kind", None)
        cid = int(getattr(move, "container_id"))
        src = tuple(getattr(move, "src"))
        dst = getattr(move, "dst", None)
        if kind == "retrieve" or dst is None:
            plan.add(Movement(cid, (int(src[0]), int(src[1])), None))
        elif kind in ("relocate", "load"):
            dst_t = tuple(dst)
            plan.add(
                Movement(
                    cid,
                    (int(src[0]), int(src[1])),
                    (int(dst_t[0]), int(dst_t[1])),
                )
            )
    return plan


def plan_from_yard(yard: Any) -> RelocationPlan:
    """RelocationPlan from ``yard.move_history``."""
    return plan_from_moves(getattr(yard, "move_history", []) or [])


def moves_to_json(moves: Iterable[Any]) -> List[Dict[str, Any]]:
    """Serialize Move-like objects for result JSON / ProgressRecord.extra."""
    out: List[Dict[str, Any]] = []
    for move in moves:
        kind = str(getattr(move, "kind", "relocate"))
        src = tuple(getattr(move, "src"))
        dst = getattr(move, "dst", None)
        entry: Dict[str, Any] = {
            "kind": kind,
            "container_id": int(getattr(move, "container_id")),
            "from": [int(src[0]), int(src[1])],
        }
        if kind == "retrieve" or dst is None:
            entry["to"] = None
        else:
            dst_t = tuple(dst)
            entry["to"] = [int(dst_t[0]), int(dst_t[1])]
        out.append(entry)
    return out


def export_yard_moves(yard: Any) -> List[Dict[str, Any]]:
    """JSON list of moves from ``yard.move_history``."""
    return moves_to_json(getattr(yard, "move_history", []) or [])


def plan_from_json(moves: Sequence[Dict[str, Any]]) -> RelocationPlan:
    """Rebuild RelocationPlan from ``moves_to_json`` output."""
    plan = RelocationPlan()
    for entry in moves:
        frm = entry["from"]
        to = entry.get("to")
        src: Tuple[int, int] = (int(frm[0]), int(frm[1]))
        if entry.get("kind") == "retrieve" or to is None:
            plan.add(Movement(int(entry["container_id"]), src, None))
        else:
            plan.add(
                Movement(
                    int(entry["container_id"]),
                    src,
                    (int(to[0]), int(to[1])),
                )
            )
    return plan


def crane_time_for_yard(
    yard: Any,
    config_extra: Optional[Dict[str, Any]] = None,
    initial_pos: Tuple[int, int] = (1, 1),
) -> float:
    """Lee & Lee–style total crane seconds for the yard's move history."""
    plan = plan_from_yard(yard)
    if not plan.movements:
        return 0.0
    kin = KinematicsModel.from_config_extra(config_extra or {})
    return float(compute_crane_time(plan, kin, initial_pos=initial_pos))


def attach_crane_time(
    metrics: Dict[str, float],
    yard: Any,
    config_extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Copy *metrics* and add ``crane_time`` (seconds) from yard history."""
    out = dict(metrics)
    out["crane_time"] = crane_time_for_yard(yard, config_extra)
    return out


def moves_from_progress_extra(extra: Any) -> Optional[List[Dict[str, Any]]]:
    """Pull ``moves`` list from a ProgressRecord.extra dict, if present."""
    if not isinstance(extra, dict):
        return None
    moves = extra.get("moves")
    if isinstance(moves, list) and moves:
        return moves
    return None


def moves_from_final_record(final: Any) -> Optional[List[Dict[str, Any]]]:
    """Pull ``moves`` from the last progress record (object or dict)."""
    if final is None:
        return None
    if isinstance(final, dict):
        return moves_from_progress_extra(final.get("extra"))
    return moves_from_progress_extra(getattr(final, "extra", None))

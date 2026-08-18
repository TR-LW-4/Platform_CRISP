"""Benchmark dataset indexes for the Web workbench (no algorithm changes)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from core.caserta_benchmark import (
    CASERTA_HEIGHT_WHITELIST,
    DEFAULT_CASERTA_DIR,
    collect_paths_from_hw_queue,
    index_ws_by_height,
    problem_config_for_caserta_dat,
)
from core.zhu_benchmark import (
    DEFAULT_ZHU_DIR,
    ZHU_HEIGHT_WHITELIST,
    collect_paths_from_zhu_queue,
    index_sn_pairs_by_height,
    problem_config_for_zhu_txt,
)
from core.zhu_dup_benchmark import (
    DEFAULT_ZHU_DUP_DIR,
    ZHU_DUP_ALPHAS,
    ZHU_DUP_HEIGHT_WHITELIST,
    collect_paths_from_zhu_dup_queue,
    index_sn_pairs_by_height_dup,
    problem_config_for_zhu_dup_txt,
)

BENCH_PROBLEMS = frozenset({"CRP-R", "CRP-U"})
BENCH_DUP_PROBLEMS = frozenset({"CRP-D"})

SOURCE_RANDOM = "random"
SOURCE_CASERTA = "caserta"
SOURCE_ZHU = "zhu"
SOURCE_ZHU_DUP = "zhu_dup"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def caserta_root() -> Path:
    return DEFAULT_CASERTA_DIR if DEFAULT_CASERTA_DIR.is_dir() else _project_root() / "benchmark" / "Caserta_dataset"


def zhu_root() -> Path:
    return DEFAULT_ZHU_DIR if DEFAULT_ZHU_DIR.is_dir() else _project_root() / "benchmark" / "Zhu_dataset"


def dup_root() -> Path:
    return DEFAULT_ZHU_DUP_DIR if DEFAULT_ZHU_DUP_DIR.is_dir() else _project_root() / "benchmark" / "dup_dataset"


def available_for_problem(problem_name: str) -> Dict[str, Any]:
    """Return selectable instance sources and indexes for one problem."""
    sources: List[Dict[str, Any]] = [{
        "id": SOURCE_RANDOM,
        "label": "Random Layout",
        "available": True,
    }]

    if problem_name in BENCH_PROBLEMS:
        ws_by_height = index_ws_by_height(caserta_root())
        heights = [h for h in CASERTA_HEIGHT_WHITELIST if h in ws_by_height]
        sources.append({
            "id": SOURCE_CASERTA,
            "label": "Caserta Benchmark",
            "available": bool(heights),
            "heights": heights,
            "ws_by_height": {str(h): ws_by_height[h] for h in heights},
            "caption": (
                "Filename data[H]-[w]-[id].dat. Select height H and one or more "
                "stack counts w, then add to the instance list."
            ),
        })

        sn_by_height = index_sn_pairs_by_height(zhu_root())
        zhu_heights = [h for h in ZHU_HEIGHT_WHITELIST if h in sn_by_height]
        sources.append({
            "id": SOURCE_ZHU,
            "label": "Zhu Benchmark",
            "available": bool(zhu_heights),
            "heights": zhu_heights,
            "sn_by_height": {
                str(h): [{"s": s, "n": n} for s, n in sn_by_height[h]]
                for h in zhu_heights
            },
            "caption": (
                "Folders named H-S-N. Select height H and one or more (S, N) "
                "scales; every .txt in each selected folder is included."
            ),
        })

    if problem_name in BENCH_DUP_PROBLEMS:
        alphas = [a for a in ZHU_DUP_ALPHAS if (dup_root() / a).is_dir()]
        alpha_indexes: Dict[str, Any] = {}
        for alpha in alphas:
            sn = index_sn_pairs_by_height_dup(dup_root(), alpha)
            heights = [h for h in ZHU_DUP_HEIGHT_WHITELIST if h in sn]
            alpha_indexes[alpha] = {
                "heights": heights,
                "sn_by_height": {
                    str(h): [{"s": s, "n": n} for s, n in sn[h]]
                    for h in heights
                },
            }
        sources.append({
            "id": SOURCE_ZHU_DUP,
            "label": "Zhu Dup Benchmark",
            "available": any(item["heights"] for item in alpha_indexes.values()),
            "alphas": alphas,
            "alpha_indexes": alpha_indexes,
            "caption": (
                "Directory dup_dataset/{alpha}/{H-S-N}. Select alpha, height H, "
                "and one or more (S, N) scales."
            ),
        })

    return {
        "problem_name": problem_name,
        "sources": sources,
        "notes": (
            "Caserta/Zhu support CRP-R and CRP-U; ZhuDup supports CRP-D."
            if problem_name not in BENCH_PROBLEMS | BENCH_DUP_PROBLEMS
            else None
        ),
    }


def _normalize_hw_queue(queue: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for block in queue:
        h = int(block["h"])
        ws = sorted({int(w) for w in block.get("ws", [])})
        if ws:
            out.append({"h": h, "ws": ws})
    return out


def _normalize_sn_queue(queue: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for block in queue:
        h = int(block["h"])
        pairs: List[Tuple[int, int]] = []
        for pair in block.get("sn_pairs", []):
            if isinstance(pair, Mapping):
                pairs.append((int(pair["s"]), int(pair["n"])))
            else:
                pairs.append((int(pair[0]), int(pair[1])))
        if pairs:
            out.append({"h": h, "sn_pairs": sorted(set(pairs))})
    return out


def resolve_paths(
    source: str,
    queue: Sequence[Mapping[str, Any]],
    alpha: Optional[str] = None,
    first_only: bool = False,
) -> List[Path]:
    """Materialize layout paths from a GUI instance queue."""
    if source == SOURCE_CASERTA:
        paths = collect_paths_from_hw_queue(caserta_root(), _normalize_hw_queue(queue))
    elif source == SOURCE_ZHU:
        paths = collect_paths_from_zhu_queue(zhu_root(), _normalize_sn_queue(queue))
    elif source == SOURCE_ZHU_DUP:
        chosen = alpha or (ZHU_DUP_ALPHAS[0] if ZHU_DUP_ALPHAS else "alpha=0.2")
        paths = collect_paths_from_zhu_dup_queue(
            dup_root(),
            _normalize_sn_queue(queue),
            chosen,
        )
    else:
        return []
    if first_only and paths:
        return paths[:1]
    return paths


def problem_config_for_layout(
    source: str,
    layout_path: Path,
    problem_values: Mapping[str, Any],
):
    """Build ProblemConfig for one benchmark layout file."""
    params = dict(problem_values)
    if source == SOURCE_ZHU_DUP:
        return problem_config_for_zhu_dup_txt(layout_path, params)
    if source == SOURCE_ZHU:
        return problem_config_for_zhu_txt(layout_path, params)
    return problem_config_for_caserta_dat(layout_path, params)


def source_tag(source: str) -> str:
    return {
        SOURCE_CASERTA: "caserta_batch",
        SOURCE_ZHU: "zhu_batch",
        SOURCE_ZHU_DUP: "zhu_dup_batch",
    }.get(source, source)


def validate_source_for_problem(problem_name: str, source: str) -> None:
    if source == SOURCE_RANDOM:
        return
    if source in {SOURCE_CASERTA, SOURCE_ZHU} and problem_name not in BENCH_PROBLEMS:
        raise ValueError(f"{source} benchmark requires CRP-R or CRP-U")
    if source == SOURCE_ZHU_DUP and problem_name not in BENCH_DUP_PROBLEMS:
        raise ValueError("zhu_dup benchmark requires CRP-D")
    if source not in {
        SOURCE_RANDOM, SOURCE_CASERTA, SOURCE_ZHU, SOURCE_ZHU_DUP,
    }:
        raise ValueError(f"Unknown instance source: {source}")

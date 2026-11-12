"""Find problem and algorithm classes, and list them from the command line."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Optional


_PROBLEMS: Dict[str, type] = {}
_ALGORITHMS: Dict[str, type] = {}
_loaded = False


def _walk(package_name: str):
    try:
        package = importlib.import_module(package_name)
    except ImportError:
        return
    for root in getattr(package, "__path__", []):
        base = Path(root)
        for path in sorted(base.rglob("*.py")):
            if path.name == "__init__.py":
                continue
            module_name = package_name + "." + ".".join(
                path.relative_to(base).with_suffix("").parts
            )
            try:
                yield importlib.import_module(module_name)
            except Exception:
                continue


def _ensure_loaded() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    from core.problem import BaseProblem

    for module in _walk("problems"):
        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                obj.__module__ == module.__name__
                and issubclass(obj, BaseProblem)
            ):
                _PROBLEMS[getattr(obj, "name", obj.__name__)] = obj

    for module in _walk("algorithms"):
        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ != module.__name__ or obj.__name__ == "Algorithm":
                continue
            if "main" in obj.__dict:
                _ALGORITHMS[getattr(obj, "name", obj.__name__)] = obj


def get_problem_class(name: str) -> Optional[type]:
    _ensure_loaded()
    return _PROBLEMS.get(name)




def list_problems() -> List[str]:
    _ensure_loaded()
    return sorted(_PROBLEMS)


def get_problem_info() -> List[Dict]:
    _ensure_loaded()
    return [
        {
            "name": name,
            "description": getattr(cls, "description", ""),
            "tags": getattr(cls, "tags", []),
            "metric_names": getattr(cls, "metric_names", ["relocations"]),
        }
        for name, cls in sorted(_PROBLEMS.items())
    ]


def get_algorithm_info() -> List[Dict]:
    _ensure_loaded()
    return [
        {
            "name": name,
            "category": getattr(cls, "category", "Unknown"),
            "description": getattr(cls, "description", ""),
            "compatible_problems": getattr(cls, "compatible_problems", []),
        }
        for name, cls in sorted(_ALGORITHMS.items())
    ]


def compatible_algorithms(problem_name: str) -> List[str]:
    _ensure_loaded()
    names = []
    for name, cls in _ALGORITHMS.items():
        compat = getattr(cls, "compatible_problems", [])
        if not compat or problem_name in compat:
            names.append(name)
    return sorted(names)


def main() -> None:
    print("\n=== Problems ===")
    problems = get_problem_info()
    if not problems:
        print("  (none)")
    for problem in problems:
        tags = ", ".join(problem["tags"])
        label = f"[{tags}] " if tags else ""
        print(f"  {label}{problem['name']}: {problem['description'][:70]}")

    print("\n=== Algorithms ===")
    algorithms = get_algorithm_info()
    if not algorithms:
        print("  (none)")
    for algorithm in algorithms:
        print(
            f"  [{algorithm['category']}] {algorithm['name']}: "
            f"{algorithm['description'][:70]}"
        )


if __name__ == "__main__":
    main()

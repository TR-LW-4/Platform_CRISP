"""
Auto-discovery registry for problems and algorithms.

On import, the registry scans the `problems/` and `algorithms/`
packages and registers all concrete subclasses of BaseProblem /
BaseAlgorithm.  The GUI reads from the registry to populate its
dropdown menus – no manual registration needed.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Dict, List, Optional, Type


def _discover_subclasses(
    package_name: str,
    base_class: type,
) -> Dict[str, type]:
    """
    Recursively import all modules in *package_name* and collect
    concrete subclasses of *base_class*.
    Returns {class.name: class}.
    """
    result: Dict[str, type] = {}

    try:
        package = importlib.import_module(package_name)
    except ImportError:
        return result

    prefix = package.__name__ + "."
    for _importer, modname, _ispkg in pkgutil.walk_packages(
        path=package.__path__,
        prefix=prefix,
    ):
        try:
            module = importlib.import_module(modname)
        except Exception:
            continue

        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, base_class)
                and obj is not base_class
                and not inspect.isabstract(obj)
                and obj.__module__ == module.__name__
            ):
                key = getattr(obj, "name", obj.__name__)
                result[key] = obj

    return result


# ================================================================ #
#  Problem registry                                                  #
# ================================================================ #

_PROBLEM_REGISTRY: Dict[str, type] = {}


def _problem_shown_in_picker(cls: type) -> bool:
    """If False, problem is registered but excluded from ``list_problems()`` / ``get_problem_info()``."""
    return not getattr(cls, "hide_from_problem_list", False)


def register_problem(cls: type) -> type:
    """Decorator: manually register a problem class."""
    from core.base_problem import BaseProblem
    assert issubclass(cls, BaseProblem), f"{cls} must subclass BaseProblem"
    _PROBLEM_REGISTRY[cls.name] = cls
    return cls


def get_problem_class(name: str) -> Optional[type]:
    _ensure_loaded()
    return _PROBLEM_REGISTRY.get(name)


def list_problems() -> List[str]:
    _ensure_loaded()
    return sorted(
        name for name, cls in _PROBLEM_REGISTRY.items() if _problem_shown_in_picker(cls)
    )


def get_problem_info() -> List[Dict]:
    """Return list of dicts for GUI display."""
    _ensure_loaded()
    out = []
    for name, cls in _PROBLEM_REGISTRY.items():
        if not _problem_shown_in_picker(cls):
            continue
        out.append({
            "name":        name,
            "description": getattr(cls, "description", ""),
            "tags":        getattr(cls, "tags", []),
            "metric_names": getattr(cls, "metric_names", ["relocations"]),
        })
    return sorted(out, key=lambda d: d["name"])


# ================================================================ #
#  Algorithm registry                                                #
# ================================================================ #

_ALGORITHM_REGISTRY: Dict[str, type] = {}


def register_algorithm(cls: type) -> type:
    """Decorator: manually register an algorithm class."""
    from core.base_algorithm import BaseAlgorithm
    assert issubclass(cls, BaseAlgorithm), f"{cls} must subclass BaseAlgorithm"
    _ALGORITHM_REGISTRY[cls.name] = cls
    return cls


def get_algorithm_class(name: str) -> Optional[type]:
    _ensure_loaded()
    return _ALGORITHM_REGISTRY.get(name)


def list_algorithms() -> List[str]:
    _ensure_loaded()
    return sorted(_ALGORITHM_REGISTRY.keys())


def get_algorithm_info() -> List[Dict]:
    """Return list of dicts for GUI display."""
    _ensure_loaded()
    out = []
    for name, cls in _ALGORITHM_REGISTRY.items():
        out.append({
            "name":                 name,
            "category":             getattr(cls, "category", "Unknown"),
            "description":          getattr(cls, "description", ""),
            "compatible_problems":  getattr(cls, "compatible_problems", []),
        })
    return sorted(out, key=lambda d: (d["category"], d["name"]))


# ================================================================ #
#  Compatible algorithms for a given problem                         #
# ================================================================ #

def compatible_algorithms(problem_name: str) -> List[str]:
    """Return algorithm names compatible with *problem_name*."""
    _ensure_loaded()
    result = []
    for name, cls in _ALGORITHM_REGISTRY.items():
        compat = getattr(cls, "compatible_problems", [])
        if not compat or problem_name in compat:
            result.append(name)
    return sorted(result)


# ================================================================ #
#  Lazy auto-load                                                    #
# ================================================================ #

_loaded = False


def _ensure_loaded() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True

    from core.base_problem   import BaseProblem
    from core.base_algorithm import BaseAlgorithm

    _PROBLEM_REGISTRY.update(
        _discover_subclasses("problems", BaseProblem)
    )
    _ALGORITHM_REGISTRY.update(
        _discover_subclasses("algorithms", BaseAlgorithm)
    )


def reload_all() -> None:
    """Force re-scan (useful during development)."""
    global _loaded
    _loaded = False
    _PROBLEM_REGISTRY.clear()
    _ALGORITHM_REGISTRY.clear()
    _ensure_loaded()

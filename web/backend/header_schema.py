"""Parameter form for the workbench.

Same role as PlatEMO's ``uilist.m``: common fields live here, and
``name --- default --- label`` lines in a file header become text boxes.
"""

from __future__ import annotations

import inspect
from typing import Dict


def parse_header(text: str) -> Dict[str, Dict]:
    """Return one field per ``---`` line in a module docstring."""
    schema: Dict[str, Dict] = {}
    for line in (text or "").splitlines():
        parts = [part.strip() for part in line.split("---")]
        if len(parts) < 2 or not parts[0].isidentifier():
            continue
        schema[parts[0]] = {
            "type": "str",
            "default": parts[1],
            "label": parts[2] if len(parts) > 2 else parts[0],
        }
    return schema


def base_algorithm_schema() -> Dict[str, Dict]:
    """Fields every algorithm form shows unless the file header replaces them."""
    return {
        "max_iterations": {
            "type": "int", "default": 500, "min": 10, "max": 100_000,
        },
        "report_interval": {
            "type": "int", "default": 10, "min": 1, "max": 100,
        },
    }


def algorithm_config_schema(cls) -> Dict[str, Dict]:
    """Schema the workbench and CLI use for one algorithm class."""
    if cls is None:
        return {}
    if _defines_config_schema(cls):
        return {**base_algorithm_schema(), **cls.config_schema()}
    header = parse_header(inspect.getmodule(cls).__doc__ or "")
    return header or base_algorithm_schema()


def _defines_config_schema(cls) -> bool:
    for klass in cls.__mro__:
        if klass.__name__ == "BaseAlgorithm":
            return False
        if "config_schema" in klass.__dict__:
            return True
    return False

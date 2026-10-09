"""
Parameter values of an algorithm, with config_schema() as the only place that
holds the defaults (the role of PlatEMO's Algorithm.ParameterSet).

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Dict

from core.base_algorithm import AlgorithmConfig, BaseAlgorithm


def parameters(config: AlgorithmConfig, algorithm: type) -> Dict[str, Any]:
    """The algorithm's own parameters (not those of BaseAlgorithm, such as seed):
    the value in config.extra if given, otherwise the default in its schema.
    Values are returned as given; the algorithm converts them where it uses them.

    Example:
        p = parameters(self.config, type(self))
        rho = int(p["rho"])
    """
    base = BaseAlgorithm.config_schema()
    return {key: config.extra.get(key, spec["default"])
            for key, spec in algorithm.config_schema().items() if key not in base}

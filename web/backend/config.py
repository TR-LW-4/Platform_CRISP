"""Framework-independent configuration adapters for the Web API."""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Mapping, TypeVar

from core.base_algorithm import AlgorithmConfig
from core.base_problem import ProblemConfig
from core.objectives import ObjectiveSpec


ConfigT = TypeVar("ConfigT", AlgorithmConfig, ProblemConfig)


def _from_values(config_type: type[ConfigT], values: Mapping[str, Any]) -> ConfigT:
    field_names = {field.name for field in fields(config_type)}
    direct = {
        key: value
        for key, value in values.items()
        if key in field_names and key != "extra"
    }
    extra = {
        key: value
        for key, value in values.items()
        if key not in field_names
    }
    config = config_type(**direct)
    config.extra.update(extra)
    return config


def problem_config(values: Mapping[str, Any]) -> ProblemConfig:
    config = _from_values(ProblemConfig, values)
    if "objective_mode" in values:
        ObjectiveSpec.from_config(config)
    return config


def algorithm_config(values: Mapping[str, Any]) -> AlgorithmConfig:
    return _from_values(AlgorithmConfig, values)

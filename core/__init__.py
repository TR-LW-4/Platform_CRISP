"""
CRP Platform – core package.

Public re-exports for convenient imports elsewhere in the project.
"""

from .container import Container, ContainerSize, ContainerType, make_containers
from .yard import Yard, Stack, Move
from .base_problem import BaseProblem, ProblemConfig
from .base_algorithm import BaseAlgorithm, AlgorithmConfig, TrainingSession
from .plan import Movement, RelocationPlan, SimResult, simulate_plan, ConflictType
from .objectives import KinematicsModel, compute_crane_time, lower_bound_relocations

__all__ = [
    # container
    "Container", "ContainerSize", "ContainerType", "make_containers",
    # yard
    "Yard", "Stack", "Move",
    # base classes
    "BaseProblem", "ProblemConfig",
    "BaseAlgorithm", "AlgorithmConfig", "TrainingSession",
    # plan primitives
    "Movement", "RelocationPlan", "SimResult", "simulate_plan", "ConflictType",
    # objectives / kinematics
    "KinematicsModel", "compute_crane_time", "lower_bound_relocations",
]

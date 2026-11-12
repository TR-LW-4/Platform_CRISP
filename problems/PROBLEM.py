from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .yard import Yard
from .container import Container


class BaseProblem:

    def __init__(self, num_bays=4, num_rows=1, max_tiers=3, num_containers=0):
        self.num_bays = num_bays
        self.num_rows = num_rows
        self.max_tiers = max_tiers
        self.num_containers = num_containers
        self.yard = Yard(num_bays, num_rows, max_tiers)
        self.containers = []
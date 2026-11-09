from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .yard import Yard
from .container import Container


@dataclass
class ProblemConfig:

    # ── Yard geometry ────────────────────────────────────────────── #
    num_bays: int = 4
    num_rows: int = 1
    max_tiers: int = 3

    # ── Containers ───────────────────────────────────────────────── #
    num_containers: int = 0 
    
   
    extra: Dict[str, Any] = field(default_factory=dict)



class BaseProblem(ABC):

    def __init__(
        self,
        config: Optional[ProblemConfig] = None,
    ):
        super().__init__()
        self.config      = config if config is not None else ProblemConfig()

        self.yard = Yard(
            self.config.num_bays,
            self.config.num_rows,
            self.config.max_tiers,
        )

        self.containers: List[Container] = []


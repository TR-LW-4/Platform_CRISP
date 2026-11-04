from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .yard import Yard
from .container import Container


@dataclass
class ProblemConfig:

    # ── Yard geometry ────────────────────────────────────────────── #
    num_bays:   int   = 4
    num_rows:   int   = 2
    max_tiers:  int   = 3


    # ── Containers ───────────────────────────────────────────────── #
    num_containers: int  = 12
    
   
    extra: Dict[str, Any] = field(default_factory=dict)



class BaseProblem(ABC):

    # ── Metadata (override in subclasses) ─────────────────────────── #
    name:         str  = "BaseProblem"
    description:  str  = ""
    tags:         List[str] = []
    metric_names: List[str] = ["relocations"]


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


# Preferred key on ``ProblemConfig.extra`` (absolute path to layout file).
LAYOUT_FILE_EXTRA_KEY = "layout_file_path"

# Legacy key (still read for backward compatibility).
_LEGACY_LAYOUT_EXTRA_KEY = "caserta_dat_path"


def layout_path_from_extra(extra: Dict[str, Any]) -> Optional[str]:
    """Return layout file path string, or ``None``."""
    if not extra:
        return None
    p = extra.get(LAYOUT_FILE_EXTRA_KEY) or extra.get(_LEGACY_LAYOUT_EXTRA_KEY)
    return str(p) if p else None

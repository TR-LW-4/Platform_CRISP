"""
Stdout layout parser for BacciBC2020.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

_PERIOD_RE = re.compile(r"\*\*\*\* Time period = (\d+)")
_CELL_RE = re.compile(r"\[([^\]]*)\]")


def parse_bacci_period_layouts(
    stdout: str,
    n_stacks: int,
    max_tiers: int,
) -> Dict[int, Dict[int, Tuple[int, int]]]:
    """Parse verbS=1 yard dumps into {period: {priority: (col, tier)}}."""
    layouts: Dict[int, Dict[int, Tuple[int, int]]] = {}
    current_t: Optional[int] = None
    rows: List[List[int]] = []

    def flush() -> None:
        if current_t is None or not rows:
            return
        pos: Dict[int, Tuple[int, int]] = {}
        n_rows = len(rows)
        for r_i, row in enumerate(rows):
            tier = n_rows - r_i
            for j, pri in enumerate(row):
                if pri > 0:
                    pos[pri] = (j + 1, tier)
        layouts[current_t] = pos

    for line in stdout.splitlines():
        period = _PERIOD_RE.search(line)
        if period:
            flush()
            current_t = int(period.group(1))
            rows = []
            continue
        if current_t is None:
            continue
        cells = _CELL_RE.findall(line)
        if len(cells) != n_stacks:
            continue
        vals = []
        for cell in cells:
            tok = cell.strip()
            vals.append(int(tok) if tok else 0)
        rows.append(vals)
    flush()
    return layouts

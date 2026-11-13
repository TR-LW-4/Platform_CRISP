"""
Lee & Lee (2010) container retrieval instances, parse only.

Y. Lee, Y.-J. Lee, "A heuristic for retrieving containers from a yard",
Computers & Operations Research 37 (2010) 1139-1147; instances published at
http://sites.google.com/site/smallcontainerworld/ (p. 1145), sets R (Table 1,
p. 1145) and U (Table 3, p. 1146), and Example 1 (R020306_0020_001, Fig. 6,
p. 1144). Local copy: benchmark/lee_lee/ (gitignored).

File format (not documented in the paper; read from the files, see
docs/validation/forster_bortfeldt_2011_multibay.md, P1):
    Line 1:  name B S H C X   -- bays, stacks per bay, maximum stack height,
                                 containers, X (sixth field, kept as n_groups)
    Next B*S lines, bay by bay, stack by stack:
             b s k  id_1 g_1  id_2 g_2 ... id_k g_k
                              -- k containers, bottom to top, as (id, group)
File names: <set><BB><SS><HH>_<CCCC>_<NNN>.txt, e.g. R011606_0070_001.

Not wired into problems/, the web backend, the UI or the CLI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEE_LEE_DIR = _PROJECT_ROOT / "benchmark" / "lee_lee"

_NAME = re.compile(r"^([A-Z])(\d{2})(\d{2})(\d{2})_(\d{4})_(\d{3})$")

Layout = Tuple[Tuple[Tuple[int, ...], ...], ...]


class LeeLeeName(NamedTuple):
    set: str
    n_bays: int
    n_stacks: int
    max_height: int
    n_containers: int
    index: int


@dataclass(frozen=True)
class LeeLeeInstance:
    """groups[b][s] and ids[b][s]: bay b, stack s (both from 0), bottom to top."""
    name: str
    n_bays: int
    n_stacks: int
    max_height: int
    n_containers: int
    n_groups: int
    groups: Layout
    ids: Layout


def parse_lee_lee_name(name: str) -> Optional[LeeLeeName]:
    """'R011606_0070_001' (with or without .txt) -> ('R', 1, 16, 6, 70, 1); None if it does not match."""
    m = _NAME.match(name[:-4] if name.endswith(".txt") else name)
    if not m:
        return None
    return LeeLeeName(m.group(1), *(int(x) for x in m.groups()[1:]))


def parse_lee_lee(path: Path) -> LeeLeeInstance:
    lines = [ln.split() for ln in Path(path).read_text().splitlines() if ln.strip()]
    if not lines or len(lines[0]) != 6:
        raise ValueError(f"{path}: bad header")
    name = lines[0][0]
    B, S, H, C, X = (int(x) for x in lines[0][1:])
    if len(lines) != 1 + B * S:
        raise ValueError(f"{path}: {len(lines) - 1} stack lines, header gives {B} x {S}")
    groups: List[List[Tuple[int, ...]]] = [[] for _ in range(B)]
    ids: List[List[Tuple[int, ...]]] = [[] for _ in range(B)]
    for i, toks in enumerate(lines[1:]):
        b, s, k, *rest = (int(x) for x in toks)
        if (b, s) != (i // S + 1, i % S + 1):
            raise ValueError(f"{path}: line {i + 2} is bay {b} stack {s}, expected {i // S + 1} {i % S + 1}")
        if len(rest) != 2 * k or k > H:
            raise ValueError(f"{path}: bay {b} stack {s}: height {k}, {len(rest)} numbers, H = {H}")
        ids[b - 1].append(tuple(rest[0::2]))
        groups[b - 1].append(tuple(rest[1::2]))
    if sum(len(s) for bay in groups for s in bay) != C:
        raise ValueError(f"{path}: container count differs from the header ({C})")
    return LeeLeeInstance(name, B, S, H, C, X,
                          tuple(map(tuple, groups)), tuple(map(tuple, ids)))


def list_lee_lee_files(root: Optional[Path] = None) -> List[Path]:
    """All instance files under root (default benchmark/lee_lee), sorted by name; [] if root is missing."""
    root = Path(root) if root else DEFAULT_LEE_LEE_DIR
    if not root.is_dir():
        return []
    return sorted((p for p in root.rglob("*.txt") if parse_lee_lee_name(p.name)), key=lambda p: p.name)

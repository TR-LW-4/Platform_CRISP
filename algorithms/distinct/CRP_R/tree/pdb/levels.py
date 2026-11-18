"""Pattern database over abstract states for Ku and Arthanari (2016)."""

from __future__ import annotations

import itertools
import pickle
from pathlib import Path
from typing import Dict, Iterator, Sequence, Tuple

State = Tuple[Tuple[int, ...], ...]
Levels = Dict[int, Dict[State, float]]

_FOLDER = Path(__file__).with_name("pdb_cache")


def abstract_key(stacks: Sequence[Sequence[int]]) -> State:
    """Relabel to ranks 1..r, drop empty stacks, sort stacks by base."""
    values = sorted(item for stack in stacks for item in stack)
    rank = {item: index + 1 for index, item in enumerate(values)}
    columns = [tuple(rank[item] for item in stack) for stack in stacks if stack]
    columns.sort(key=lambda column: column[0])
    return tuple(columns)


def load_pdb(depth: int, n_cols: int, m_tiers: int) -> Levels:
    levels: Levels = {0: {(): 0}}
    for k in range(1, depth + 1):
        path = _FOLDER / f"pdb_{n_cols}x{m_tiers}_{k}.pkl"
        if path.exists():
            with path.open("rb") as file:
                levels[k] = pickle.load(file)
            continue
        levels[k] = _build_level(k, n_cols, m_tiers, levels[k - 1])
        _FOLDER.mkdir(exist_ok=True)
        with path.open("wb") as file:
            pickle.dump(levels[k], file, protocol=pickle.HIGHEST_PROTOCOL)
    return levels


def _build_level(k, n_cols, m_tiers, previous) -> Dict[State, float]:
    table: Dict[State, float] = {}
    for state in _states(k, n_cols, m_tiers):
        _value(state, table, previous, n_cols, m_tiers)
    return table


def _value(state, table, previous, n_cols, m_tiers) -> float:
    known = table.get(state)
    if known is not None:
        return known
    columns = [list(column) for column in state]
    source = next(index for index, column in enumerate(columns) if 1 in column)
    if columns[source][-1] == 1:
        columns[source].pop()
        value = previous[abstract_key(columns)]
    else:
        blocker = columns[source].pop()
        value = float("inf")
        slots = len(columns) + (1 if len(columns) < n_cols else 0)
        for destination in range(slots):
            if destination == source:
                continue
            child = [list(column) for column in columns]
            if destination == len(child):
                child.append([blocker])
            elif len(child[destination]) < m_tiers:
                child[destination].append(blocker)
            else:
                continue
            value = min(value, 1 + _value(abstract_key(child), table, previous, n_cols, m_tiers))
    table[state] = value
    return value


def _states(r, n_cols, m_tiers) -> Iterator[State]:
    ranks = range(1, r + 1)
    for bases_count in range(-(-r // m_tiers), min(r, n_cols) + 1):
        for bases in itertools.combinations(ranks, bases_count):
            chosen = set(bases)
            rest = [item for item in ranks if item not in chosen]
            for tops in _fill(rest, bases_count, m_tiers - 1):
                yield tuple((base,) + top for base, top in zip(bases, tops))


def _fill(units, columns, height) -> Iterator[Tuple[Tuple[int, ...], ...]]:
    if columns == 0:
        if not units:
            yield ()
        return
    low = max(0, len(units) - (columns - 1) * height)
    for size in range(low, min(height, len(units)) + 1):
        for top in itertools.permutations(units, size):
            used = set(top)
            left = [item for item in units if item not in used]
            for others in _fill(left, columns - 1, height):
                yield (top,) + others

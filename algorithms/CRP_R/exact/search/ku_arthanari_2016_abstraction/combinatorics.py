"""
Ku & Arthanari (2016) — enumeration formulas for the abstraction method.

Reference
---------
D. Ku, T. S. Arthanari, "On the abstraction method for the container
relocation problem", Computers & Operations Research 68 (2016) 110-122,
Section 4.1 ("Enumeration of stacking configurations").

This module implements the paper's combinatorial results in isolation
(pure counting functions, no search) so that:
  (a) the abstraction search engine (``abstraction_core.py``) can size
      its pattern database / warn the user before spending memory, and
  (b) the results are independently checkable against the paper's own
      worked example (Table 2: r=8 units on an n=5, m=3 stack) --
      see ``test_abstraction.py``.

Definitions / equations reproduced
-----------------------------------
Definition 3   nHr        -- r-combinations with unlimited repetition of
                              an n-element set (Eq. before Lemma 1).
Lemma 1        g(n,r,m,i) -- number of stacking profiles with >= i
                              columns overloaded (>= m+1 units each).
Corollary 2    illegal(n,r,m) -- |union of illegal profiles| via
                              inclusion-exclusion on Lemma 1's g(i).
Theorem 1      C(n,r,m)   -- number of *stacking profiles* (shapes,
                              column identity forgotten) of r units on
                              an n x m stack (Eq. C = nHr - illegal).
Theorem 2      P(n,r,m)   -- number of *stacking configurations*
                              (distinct units 1..r, column identity
                              kept) = C(n,r,m) * r!  (Eq. 3).
Theorem 3      A(n,r,m)   -- number of *abstract states* (column
                              identity forgotten, i.e. the size of the
                              PDB's index space at r units).
"""

from __future__ import annotations

import math
from typing import Dict


def _comb(n: int, r: int) -> int:
    """
    C(n, r), with the paper's convention C(n,r) = 0 for r < 0 or r > n.
    Hand-rolled (iterative multiplicative formula) instead of
    ``math.comb`` for Python 3.7 compatibility (``math.comb`` needs 3.8+,
    and this platform targets 3.7 -- see other ``exact/search/*`` modules).
    """
    if r < 0 or n < 0 or r > n:
        return 0
    r = min(r, n - r)
    result = 1
    for k in range(r):
        result = result * (n - k) // (k + 1)
    return result


def num_combinations_with_repetition(n: int, r: int) -> int:
    """
    Definition 3:  nHr = C(n + r - 1, r), the number of r-combinations
    with unlimited repetition of an n-element set.  nH0 = 1 (r=0: the
    empty selection, needed as the base case for every formula below).
    """
    if r < 0:
        return 0
    if r == 0:
        return 1
    return _comb(n + r - 1, r)


def overloaded_profile_count(n: int, r: int, m: int, i: int) -> int:
    """
    Lemma 1:  g(i) = C(n,i) * nH_{r(i)} if r(i) = r - i*(m+1) >= 0, else 0.

    Number of stacking profiles in which *at least* i specific columns
    (chosen in C(n,i) ways) are overloaded with >= m+1 units each, after
    "pre-loading" those i columns with exactly m+1 units and distributing
    the remaining r(i) units among all n columns without a height limit.
    """
    if i <= 0:
        return 0
    r_i = r - i * (m + 1)
    if r_i < 0:
        return 0
    return _comb(n, i) * num_combinations_with_repetition(n, r_i)


def num_illegal_profiles(n: int, r: int, m: int) -> int:
    """
    Corollary 2:  |union_{i=1}^{n} S_i| = sum_{i=1}^{floor(r/(m+1))}
    (-1)^{i+1} * g(i), the number of *illegal* stacking profiles (i.e.
    at least one column holds more than m units), via inclusion-exclusion
    over Lemma 1's g(i).
    """
    if m + 1 <= 0:
        return 0
    max_i = r // (m + 1)
    total = 0
    for i in range(1, max_i + 1):
        term = overloaded_profile_count(n, r, m, i)
        total += term if (i % 2 == 1) else -term
    return total


def num_stacking_profiles(n: int, r: int, m: int) -> int:
    """
    Theorem 1:  C(n,r,m) = nHr - (number of illegal profiles).

    The number of *stacking profiles* of r units on an n-column, m-tier
    stack: shapes only (column order / identity not yet fixed, units not
    yet distinguished) -- e.g. Fig. 6's two profiles of 9 units on a
    4 x 3 stack.
    """
    if r == 0:
        return 1
    if n <= 0:
        return 0
    return num_combinations_with_repetition(n, r) - num_illegal_profiles(n, r, m)


def num_stacking_configurations(n: int, r: int, m: int) -> int:
    """
    Theorem 2 (Eq. 3):  P(n,r,m) = C(n,r,m) * r!

    The number of *stacking configurations* of r *distinct* units
    (sequenced 1..r) on an n x m stack -- i.e. the size of the ORIGINAL
    (non-abstracted) state space at r units, column identity included.
    """
    if r == 0:
        return 1
    return num_stacking_profiles(n, r, m) * math.factorial(r)


def num_abstract_states(n: int, r: int, m: int) -> int:
    """
    Theorem 3:  A(n,r,m) = sum_{i=ceil(r/m)}^{min(r,n)} C(r,i) *
    P(i, r-i, m-1).

    The number of *abstract states* of r units on an n x m stack: column
    identity forgotten and columns canonically sorted ascending by base
    value (Section 3.4's phi projection) -- i.e. the size of the PDB's
    index space at r units.  ``i`` ranges over the possible number of
    "base slots" (occupied columns): choose which i of the r units sit
    on the floor (``C(r,i)`` ways), then arrange the remaining r-i units
    on top of those i base slots, each with one fewer tier available
    (``P(i, r-i, m-1)``, Theorem 2 on the (i x (m-1)) sub-stack).

    Matches the paper's worked example exactly (Table 2, r=8, n=5, m=3):
    20,160 (i=3) + 31,920 (i=4) + 10,080 (i=5) = 62,160.
    """
    if r == 0:
        return 1
    if n <= 0 or m <= 0:
        return 0
    lo = -(-r // m)  # ceil(r / m)
    hi = min(r, n)
    total = 0
    for i in range(lo, hi + 1):
        total += _comb(r, i) * num_stacking_configurations(i, r - i, m - 1)
    return total


def abstract_state_breakdown(n: int, r: int, m: int) -> Dict[int, int]:
    """
    Same summation as :func:`num_abstract_states`, but returns the
    per-``i`` (number of base slots) contribution -- reproduces Table 2's
    row-by-row breakdown (one dict entry per "i baseslots" scenario).
    """
    if r == 0:
        return {0: 1}
    lo = -(-r // m)
    hi = min(r, n)
    return {
        i: _comb(r, i) * num_stacking_configurations(i, r - i, m - 1)
        for i in range(lo, hi + 1)
    }


def reduction_factor(n: int, r: int, m: int) -> float:
    """
    |original state space| / |abstract state space| (Table 3's ratio
    column).  Returns ``inf`` if the abstract space is empty (r > 0 but
    the stack cannot hold r units, e.g. r > n*m).
    """
    original = num_stacking_configurations(n, r, m)
    abstracted = num_abstract_states(n, r, m)
    if abstracted == 0:
        return float("inf") if original else 0.0
    return original / abstracted

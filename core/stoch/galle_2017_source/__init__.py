"""
Shared infrastructure ported from the reference implementation of the
Stochastic Container Relocation Problem (SCRP), originally published at
`https://github.com/vgalle/StochasticCRP` (Galle et al., "The Stochastic
Container Relocation Problem", 2017/2018) and vendored locally at
`/data/liuw2/StochasticCRP-master`.

This package is shared *infrastructure* (bay representation, lower
bounds, retrieval policies, tree search) consumed by several CRP-Stoch
algorithm packages — following the same pattern as
:mod:`core.stoch.policy_tree` and :mod:`core.stoch.batch_layout`.

Modules
-------
bay_utils    : source bay representation, abstraction, unveiling,
               instance generation/loading, preprocessing
bounds       : Blocking lower bound, rolling lower bound,
               bounds-difference helper for PBFSA sampling
heuristics   : EG / EM / ERI / L / Rand retrieval policies and
               Monte-Carlo heuristic evaluators
astar        : deterministic CRP tail solver used once the full
               retrieval order is known
tree_search  : PBFS_Online and PBFSA (exact if error_gap=0) tree search

Consumers
---------
- ``algorithms/CRP_Stoch/heuristic/galle_2017_eg/``       (EG)
- ``algorithms/CRP_Stoch/heuristic/ku_arthanari_2016_eri/`` (ERI)
- ``algorithms/CRP_Stoch/exact/galle_2017_pbfs/``          (PBFS / PBFSA)
"""

from __future__ import annotations

from .bay_utils import (
    abstract_bay,
    batch_start_labels,
    env_to_source_bay,
    generate_incomplete_config,
    preprocess_bay,
    read_input_file,
    unveil_containers,
    unveil_containers_online,
)
from .bounds import (
    blocking_lower_bound,
    bounds_difference,
    rolling_lower_bound,
)
from .heuristics import (
    HEURISTIC_IDS,
    run_heuristic_batch,
    run_heuristic_online,
)
from .astar import astar
from .tree_search import pbfs_online, pbfsa

__all__ = [
    "abstract_bay",
    "astar",
    "batch_start_labels",
    "blocking_lower_bound",
    "bounds_difference",
    "env_to_source_bay",
    "generate_incomplete_config",
    "HEURISTIC_IDS",
    "pbfs_online",
    "pbfsa",
    "preprocess_bay",
    "read_input_file",
    "rolling_lower_bound",
    "run_heuristic_batch",
    "run_heuristic_online",
    "unveil_containers",
    "unveil_containers_online",
]

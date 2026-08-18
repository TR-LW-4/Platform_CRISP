"""
core.benchmarks — adapters for published benchmark instance sets.

Each adapter parses an external layout file, derives the yard geometry,
and builds a ProblemConfig, so that instances from different sources are
evaluated under the same protocol.

caserta       : Caserta ``data{H}-{W}-{id}.dat``
zhu           : Zhu ``H-S-N/`` folders (same body format as Caserta)
zhu_duplicate : Zhu duplicate-priority instances

Submodules are re-exported wholesale so callers may use either
``from core.benchmarks import parse_caserta_dat`` or
``from core.benchmarks.caserta import parse_caserta_dat``.
"""

from core.benchmarks.caserta import *        # noqa: F401,F403
from core.benchmarks.zhu import *            # noqa: F401,F403
from core.benchmarks.zhu_duplicate import *  # noqa: F401,F403

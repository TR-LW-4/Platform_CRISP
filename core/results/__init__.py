"""
core.results — experiment result persistence and aggregation.

store     : save / load / list individual runs (JSON under results/)
aggregate : mean / std per benchmark size class for one algorithm
compare   : PlatEMO-style class x algorithm comparison tables

Submodules are re-exported wholesale so callers may use either
``from core.results import save_run`` or ``from core.results.store import save_run``.
"""

from core.results.store import *      # noqa: F401,F403
from core.results.aggregate import *  # noqa: F401,F403
from core.results.compare import *    # noqa: F401,F403

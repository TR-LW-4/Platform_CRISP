"""
Deprecated location — moved to :mod:`core.results.compare`.

Kept so existing imports keep working; new code should import from
``core.results.compare``.
"""

from core.results.compare import *  # noqa: F401,F403
from core.results.compare import _DEFAULT_DIR, _folder_for_algorithm, _safe_name  # noqa: F401

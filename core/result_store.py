"""
Deprecated location — moved to :mod:`core.results.store`.

Kept so existing imports keep working; new code should import from
``core.results.store``.
"""

from core.results.store import *  # noqa: F401,F403
from core.results.store import _DEFAULT_DIR, _result_path, _safe_token  # noqa: F401

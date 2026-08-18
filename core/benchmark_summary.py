"""
Deprecated location — moved to :mod:`core.results.aggregate`.

Kept so existing imports keep working; new code should import from
``core.results.aggregate``.
"""

from core.results.aggregate import *  # noqa: F401,F403
from core.results.aggregate import (  # noqa: F401
    _class_from_layout_path,
    _layout_path_from_run,
    _mean_std,
    _parse_caserta_filename,
    _parse_zhu_folder_name,
)

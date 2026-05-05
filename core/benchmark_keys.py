"""
Keys for loading fixed benchmark layouts into ``CRP-R`` (Caserta ``.dat``, Zhu ``.txt``, same body format).

Historically the extra field was named ``caserta_dat_path``; code still accepts it when reading.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# Preferred key on ``ProblemConfig.extra`` (absolute path to layout file).
LAYOUT_FILE_EXTRA_KEY = "layout_file_path"

# Legacy key (still read for backward compatibility).
_LEGACY_LAYOUT_EXTRA_KEY = "caserta_dat_path"


def layout_path_from_extra(extra: Dict[str, Any]) -> Optional[str]:
    """Return layout file path string, or ``None``."""
    if not extra:
        return None
    p = extra.get(LAYOUT_FILE_EXTRA_KEY) or extra.get(_LEGACY_LAYOUT_EXTRA_KEY)
    return str(p) if p else None

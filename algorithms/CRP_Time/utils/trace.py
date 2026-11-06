"""
Terminal progress lines, printed only when CRISP_TRACE_LAYOUT is set (as
core/layout_trace.py).

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import sys


def trace_layout(msg: str) -> None:
    if os.environ.get("CRISP_TRACE_LAYOUT"):
        print(f"[CRISP_TRACE_LAYOUT] {msg}", file=sys.stderr, flush=True)

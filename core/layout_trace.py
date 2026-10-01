"""Terminal progress lines, printed only when CRISP_TRACE_LAYOUT is set."""

from __future__ import annotations

import os
import sys


def trace_layout(msg: str) -> None:
    if os.environ.get("CRISP_TRACE_LAYOUT"):
        print(f"[CRISP_TRACE_LAYOUT] {msg}", file=sys.stderr, flush=True)

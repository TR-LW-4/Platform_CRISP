"""
Optional stderr tracing for Caserta / Zhu layout loading.

Enable layout loading tracing::

    export CRISP_TRACE_LAYOUT=1
    python main.py web
    # or: python main.py layout-run ...

Enable LA-N ``build_lan_plan`` move tracing (stderr)::

    export CRISP_TRACE_LAN=1
    python main.py layout-run --problem CRP-R --algo "LA-N Look-Ahead" ...

Typical order per benchmark instance:

1. ``problem_config_for_caserta_dat`` (or Zhu alias) — builds ``ProblemConfig`` and sets ``extra["layout_file_path"]`` (legacy: ``caserta_dat_path``).
2. ``CRP_R.reset`` → ``_build_episode`` — reads that path from ``config.extra``.
3. ``apply_caserta_file_to_yard`` — parses the same file again and places containers on the yard.

Batch: the workbench / batch runner calls ``train()`` once per layout file (paths from
``collect_paths_*``). Each ``train()`` does a single ``reset()`` that loads exactly
one file. Eighty files ⇒ eighty outer iterations ⇒ eighty loads.
"""

from __future__ import annotations

import os
import sys


def trace_layout(msg: str) -> None:
    if os.environ.get("CRISP_TRACE_LAYOUT"):
        print(f"[CRISP_TRACE_LAYOUT] {msg}", file=sys.stderr, flush=True)


def trace_lan(msg: str) -> None:
    """LA-N ``planner.build_lan_plan`` move-by-move trace."""
    if os.environ.get("CRISP_TRACE_LAN"):
        print(f"[CRISP_TRACE_LAN] {msg}", file=sys.stderr, flush=True)

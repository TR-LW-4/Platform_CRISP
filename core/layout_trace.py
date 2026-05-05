"""
Optional stderr tracing for Caserta / Zhu layout loading.

Enable::

    export CRISP_TRACE_LAYOUT=1
    streamlit run gui/app.py

Typical order per benchmark instance:

1. ``problem_config_for_caserta_dat`` (or Zhu alias) — builds ``ProblemConfig`` and sets ``extra["layout_file_path"]`` (legacy: ``caserta_dat_path``).
2. ``CRP_R.reset`` → ``_build_episode`` — reads that path from ``config.extra``.
3. ``apply_caserta_file_to_yard`` — parses the same file again and places containers on the yard.

Batch: the GUI outer loop calls ``train()`` once per layout file (paths from ``collect_paths_*``, session queue only stores which groups you picked). With ``num_eval_seeds=1``, inner seed loop runs once; **each reset loads exactly one file**. Eighty files ⇒ eighty outer iterations ⇒ eighty loads (not “stored 80 paths then popped”).
"""

from __future__ import annotations

import os
import sys


def trace_layout(msg: str) -> None:
    if os.environ.get("CRISP_TRACE_LAYOUT"):
        print(f"[CRISP_TRACE_LAYOUT] {msg}", file=sys.stderr, flush=True)

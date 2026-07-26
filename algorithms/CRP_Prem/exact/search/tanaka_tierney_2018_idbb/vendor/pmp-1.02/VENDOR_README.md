# Vendored source: `pmp-1.02`

Verbatim copy (byte-identical except `Makefile`, see below) of the reference
C implementation released by Shunji Tanaka and Kevin Tierney for:

> S. Tanaka, K. Tierney, "Solving real-world sized container pre-marshalling
> problems with an iterative deepening branch-and-bound algorithm",
> European Journal of Operational Research 264 (2018) 165-180.

This is the paper referred to elsewhere in this repository as
**Tanaka & Tierney (2018)**. It is the direct predecessor to, and shares
its lower bounds / dominance rules with, the branch-and-bound approach in
the *later* 2019 paper by Tanaka, Tierney, Parreno-Torres, Alvarez-Valdes
& Ruiz -- but this specific vendored code implements the *2018* IDBB
algorithm only (single-threaded, iterative-deepening-over-relocation-count
branch and bound with the Bortfeldt & Forster lower bound plus the two
"improved" tightenings `IMPROVED_BF_LOWER_BOUND1`/`2` enabled by default in
`solve.c`, dominance rules, and a greedy upper-bound heuristic).

Each source file carries the original copyright/BSD-style license header;
nothing below has been changed apart from:

* `Makefile`: dropped `-march=native` (replaced with a portable `-O3`) so a
  binary built on one machine cannot crash with `SIGILL` if later copied to
  or executed on a different CPU. No algorithmic code is touched.

Do not hand-edit the `.c`/`.h` files in this directory; if a fix is ever
needed, prefer patching `../bridge.py` (the Python-side wrapper) instead, to
keep this directory an easily-diffable copy of the upstream release.

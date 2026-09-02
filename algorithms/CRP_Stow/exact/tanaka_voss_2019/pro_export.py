"""
Yard → Tanaka–Voß .pro export for TanakaBB2019Stow.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from problems.CRP_Stow import CRP_Stow


def _vs_to_letter(vs: int) -> str:
    """Vessel stack index → uppercase letter: 0→'A', 1→'B', …"""
    return chr(ord('A') + vs)


def env_to_pro_string(env: "CRP_Stow") -> str:
    """
    Convert the CURRENT yard state of a CRP_Stow environment to a .pro string
    that can be passed directly to the brpsp_bb binary.

    Assumes env.reset() has already been called.  Note: auto-retrieved
    containers (cdd==0 at reset) are no longer in the yard; the generated
    .pro reflects the post-auto-retrieval state.  The solver handles this
    correctly — free retrievals do not affect the relocation count.

    Parameters
    ----------
    env : CRP_Stow
        A reset CRP_Stow environment.

    Returns
    -------
    str
        Complete .pro file contents.
    """
    n_stacks = env._n_stacks
    VS       = env.config.num_groups
    YT       = env.config.max_tiers

    # Build per-stack container lists (bottom → top).
    stacks: List[List[str]] = []
    N = 0
    for i in range(n_stacks):
        key = env._idx_to_stack(i)
        stk = env.yard.stacks.get(key)
        tokens: List[str] = []
        if stk and not stk.is_empty:
            for c in stk.containers:           # bottom → top
                tokens.append(f"{_vs_to_letter(c.group)}_{c.priority}")
            N += len(tokens)
        stacks.append(tokens)

    # Update vessel_loaded accounting for already-retrieved containers.
    # H_v for the .pro is the TOTAL vessel-stack capacity (not remaining).
    H_v = list(env._vessel_max_tier)

    lines: List[str] = []
    lines.append("Number Of Containers")
    lines.append(str(N))
    lines.append("Number Of Yard Stacks")
    lines.append(str(n_stacks))
    lines.append("MaxTier of Yard Stacks")
    lines.append(str(YT))
    lines.append("Number of Vessel Stacks")
    lines.append(str(VS))
    lines.append("Max Tier of Vessel Stacks")
    for j in range(VS):
        lines.append(str(H_v[j] if j < len(H_v) else 0))
    lines.append("Yard Bay")
    for i, tokens in enumerate(stacks):
        lines.append(f"Stack {i}:")
        lines.append(" ".join(tokens))   # empty line for empty stacks

    return "\n".join(lines) + "\n"

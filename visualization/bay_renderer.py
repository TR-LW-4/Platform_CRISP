"""
Yard / vessel bay visualisation using matplotlib.

render_yard(yard, config)   → RGB numpy array  (for gym render)
yard_figure(yard, config)   → matplotlib Figure (for Streamlit)
render_vessel(vessel_state) → matplotlib Figure
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# Colour palette per group (up to 16 groups)
_GROUP_COLOURS = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2",
    "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7",
    "#9C755F", "#BAB0AC", "#D62728", "#1F77B4",
    "#FF7F0E", "#2CA02C", "#9467BD", "#8C564B",
]
_EMPTY_COLOUR  = "#EEEEEE"
_BORDER_COLOUR = "#AAAAAA"


def _group_colour(group_id: int) -> str:
    return _GROUP_COLOURS[group_id % len(_GROUP_COLOURS)]


def yard_figure(
    yard_snapshot: Dict[Tuple[int, int], List[int]],
    num_bays:   int,
    num_rows:   int,
    max_tiers:  int,
    title:      str  = "Yard State",
    highlight:  Optional[Tuple[int, int]] = None,   # (bay, row) to highlight
):
    """
    Draw the yard as a 2-D grid of stacks.

    Layout: bays along X-axis, rows stacked side-by-side.
    Each column = one (bay, row) stack; tiers grow upward.

    Returns a matplotlib Figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    n_cols  = num_bays * num_rows
    n_rows  = max_tiers

    fig, ax = plt.subplots(figsize=(max(6, n_cols * 0.9), max(4, n_rows * 0.8)))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")

    col_idx = 0
    for bay in range(1, num_bays + 1):
        for row in range(1, num_rows + 1):
            key       = (bay, row)
            stack_ids = yard_snapshot.get(key, [])

            # Draw empty cells
            for tier in range(1, max_tiers + 1):
                y     = tier - 1
                x     = col_idx
                is_hl = highlight == key
                fc    = "#FFF9C4" if is_hl else _EMPTY_COLOUR
                rect  = mpatches.FancyBboxPatch(
                    (x + 0.05, y + 0.05), 0.9, 0.9,
                    boxstyle="round,pad=0.02",
                    facecolor=fc, edgecolor=_BORDER_COLOUR, linewidth=0.8,
                )
                ax.add_patch(rect)

            # Draw containers
            for tier_idx, container_id in enumerate(stack_ids):
                y   = tier_idx
                x   = col_idx
                # container_id here is actually group+1 from group_snapshot
                grp = max(0, int(container_id) - 1)
                fc  = _group_colour(grp)
                rect = mpatches.FancyBboxPatch(
                    (x + 0.05, y + 0.05), 0.9, 0.9,
                    boxstyle="round,pad=0.02",
                    facecolor=fc, edgecolor="#333333", linewidth=1.0,
                )
                ax.add_patch(rect)
                ax.text(
                    x + 0.5, y + 0.5, str(grp),
                    ha="center", va="center",
                    fontsize=7, fontweight="bold", color="white",
                )

            # Column label
            ax.text(
                col_idx + 0.5, -0.4,
                f"B{bay}R{row}",
                ha="center", va="center",
                fontsize=6, color="#555555",
            )
            col_idx += 1

    # Bay separators
    for bay in range(1, num_bays + 1):
        x = (bay - 1) * num_rows
        ax.axvline(x, color="#888888", linewidth=1.2, linestyle="--")

    # Legend
    groups_present = sorted({
        max(0, int(g) - 1)
        for ids in yard_snapshot.values()
        for g in ids
        if g > 0
    })
    legend_patches = [
        mpatches.Patch(color=_group_colour(g), label=f"Group {g}")
        for g in groups_present
    ]
    if legend_patches:
        ax.legend(
            handles=legend_patches,
            loc="upper right",
            fontsize=7,
            framealpha=0.7,
        )

    fig.tight_layout()
    return fig


def vessel_figure(
    vessel_state: np.ndarray,   # shape (N, 5): bay/row/tier/occupied/group
    vessel_bays:  int,
    vessel_rows:  int,
    vessel_tiers: int,
    title:        str = "Vessel State",
    current_slot: Optional[int] = None,
):
    """Draw vessel bay as a 2-D grid (similar to yard_figure)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    n_cols = vessel_bays * vessel_rows
    n_rows = vessel_tiers

    fig, ax = plt.subplots(figsize=(max(5, n_cols * 1.0), max(3, n_rows * 0.9)))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")

    col_map: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = {}
    for i, slot in enumerate(vessel_state):
        b, r, t, occ, grp = int(slot[0]), int(slot[1]), int(slot[2]), int(slot[3]), int(slot[4])
        col_map.setdefault((b, r), []).append((t, occ, grp, i))

    col_idx = 0
    for bay in range(1, vessel_bays + 1):
        for row in range(1, vessel_rows + 1):
            cells = sorted(col_map.get((bay, row), []), key=lambda x: x[0])
            for tier, occ, grp, slot_i in cells:
                y    = tier - 1
                x    = col_idx
                is_cur = (slot_i == current_slot)
                if occ:
                    fc = _group_colour(grp)
                elif is_cur:
                    fc = "#FFEE58"
                else:
                    fc = "#F5F5F5"
                border = "#FF0000" if is_cur else "#555555"
                rect = mpatches.FancyBboxPatch(
                    (x + 0.05, y + 0.05), 0.9, 0.9,
                    boxstyle="round,pad=0.02",
                    facecolor=fc, edgecolor=border,
                    linewidth=1.5 if is_cur else 0.8,
                )
                ax.add_patch(rect)
                if occ:
                    ax.text(x + 0.5, y + 0.5, str(grp),
                            ha="center", va="center",
                            fontsize=7, color="white", fontweight="bold")
            ax.text(col_idx + 0.5, -0.4, f"B{bay}R{row}",
                    ha="center", va="center", fontsize=6, color="#555555")
            col_idx += 1

    for bay in range(1, vessel_bays + 1):
        ax.axvline((bay - 1) * vessel_rows, color="#888888",
                   linewidth=1.2, linestyle="--")

    fig.tight_layout()
    return fig


def render_yard(yard, config) -> np.ndarray:
    """Convert yard to RGB numpy array (for gym render)."""
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = yard_figure(
        yard.group_snapshot(),
        config.num_bays, config.num_rows, config.max_tiers,
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80)
    plt.close(fig)
    buf.seek(0)
    import imageio
    img = imageio.imread(buf)
    return img[:, :, :3]


def metrics_figure(
    history: List[float],
    label:   str  = "Metric",
    title:   str  = "Training Progress",
    lower_is_better: bool = True,
):
    """Line chart of a metric over training steps/episodes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(history, color="#4E79A7", linewidth=1.5)
    if len(history) > 10:
        # Smoothed line
        k   = max(1, len(history) // 20)
        sm  = np.convolve(history, np.ones(k) / k, mode="valid")
        ax.plot(range(k - 1, len(history)), sm,
                color="#E15759", linewidth=2.0, label="Smoothed")
    ax.set_xlabel("Step", fontsize=9)
    ax.set_ylabel(label, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    if lower_is_better and history:
        best = min(history)
        ax.axhline(best, color="#59A14F", linewidth=1.0,
                   linestyle="--", label=f"Best: {best:.2f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig

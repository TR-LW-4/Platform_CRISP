"""
Acceleration-aware crane-time model for ParrenoTorresAlvarezValdesRuizTierney2020CPMPCT.

------------------------------- Copyright --------------------------------
Copyright (c) 2026 LIACS, Leiden University.
Platform_CRISP is free for research use. Publications that use this
platform or its code should acknowledge "Platform_CRISP" and cite the
corresponding original algorithm paper.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


@dataclass
class CraneParams:
    # From the paper's RTG calibration (TR-E 2020).
    vmax_v_loaded: float = 0.5
    vmax_v_unloaded: float = 1.0
    vmax_r_loaded: float = 1.16
    vmax_r_unloaded: float = 2.16
    d_v: float = 2.65
    d_r: float = 2.50
    # Geometry
    container_h: float = 2.591
    container_w: float = 2.438
    margin_w: float = 0.300
    top_sep: float = 2.000
    start_hsep: float = 1.000


def twistlock_time(level: int, max_tiers: int) -> float:
    # b_h = 5 * (H - h + 1)
    return float(5 * (max_tiers - level + 1))


def travel_time(distance: float, vmax: float, d: float) -> float:
    if distance <= 0.0:
        return 0.0
    # a = vmax^2 / (2d)
    a = (vmax * vmax) / max(1e-9, 2.0 * d)
    if distance < 2.0 * d:
        return 2.0 * sqrt(2.0 * a * distance) / max(1e-9, a)
    return (2.0 * vmax / max(1e-9, a)) + (distance - 2.0 * d) / max(1e-9, vmax)


def vertical_distance(level: int, max_tiers: int, p: CraneParams) -> float:
    # distance_v(h) = vsep + (H-h+1) * ch
    return p.top_sep + (max_tiers - level + 1) * p.container_h


def horizontal_distance(src_stack: int, dst_stack: int, p: CraneParams) -> float:
    """
    stack indices are zero-based.
    src_stack < 0 means the initial point outside bay.
    """
    if src_stack < 0:
        s = dst_stack + 1
        return (s - 1) * (p.margin_w + p.container_w) + p.start_hsep
    return abs(dst_stack - src_stack) * (p.margin_w + p.container_w)


def move_time(
    prev_dst_stack: int,
    src_stack: int,
    src_level: int,
    dst_stack: int,
    dst_level: int,
    max_tiers: int,
    p: CraneParams,
) -> float:
    # Horizontal unloaded to source stack
    c0 = travel_time(
        horizontal_distance(prev_dst_stack, src_stack, p),
        p.vmax_r_unloaded,
        p.d_r,
    )
    # Vertical unloaded to source level
    v0 = travel_time(
        vertical_distance(src_level, max_tiers, p),
        p.vmax_v_unloaded,
        p.d_v,
    )
    # Vertical loaded hoist from source level
    v1_src = travel_time(
        vertical_distance(src_level, max_tiers, p),
        p.vmax_v_loaded,
        p.d_v,
    )
    # Horizontal loaded to destination stack
    c1 = travel_time(
        horizontal_distance(src_stack, dst_stack, p),
        p.vmax_r_loaded,
        p.d_r,
    )
    # Vertical loaded down to destination level
    v1_dst = travel_time(
        vertical_distance(dst_level, max_tiers, p),
        p.vmax_v_loaded,
        p.d_v,
    )
    # Vertical unloaded return to top lane from destination level
    v0_dst = travel_time(
        vertical_distance(dst_level, max_tiers, p),
        p.vmax_v_unloaded,
        p.d_v,
    )
    return c0 + v0 + twistlock_time(src_level, max_tiers) + v1_src + c1 + v1_dst + v0_dst


def min_move_time(max_tiers: int, p: CraneParams) -> float:
    # Adjacent stacks, source/destination at top tier.
    c12 = travel_time(p.container_w + p.margin_w, p.vmax_r_loaded, p.d_r)
    vH0 = travel_time(vertical_distance(max_tiers, max_tiers, p), p.vmax_v_unloaded, p.d_v)
    vH1 = travel_time(vertical_distance(max_tiers, max_tiers, p), p.vmax_v_loaded, p.d_v)
    return c12 + vH0 + twistlock_time(max_tiers, max_tiers) + vH1 + c12 + vH1 + vH0

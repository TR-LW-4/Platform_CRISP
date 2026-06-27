"""
de Melo da Silva et al. (2018) — CRP-D grouped-priority MIP solvers.

Public symbols
--------------
DeMeloSilva2018BRPGrouped  : unrestricted BRP (CRP-D), Gurobi MIP.
DeMeloSilva2018RBRPGrouped : restricted r-BRP (CRP-D), Gurobi MIP.
"""

from .algorithm import DeMeloSilva2018BRPGrouped, DeMeloSilva2018RBRPGrouped

__all__ = ["DeMeloSilva2018BRPGrouped", "DeMeloSilva2018RBRPGrouped"]

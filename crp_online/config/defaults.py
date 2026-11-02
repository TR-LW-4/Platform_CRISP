"""Default configuration parameters, kinematic constants, and terminal sets."""

# Kinematic constants (Verified from C++ ContainerYard line 205)
GANTRY_SPEED: float = 3.5      # m/s
TROLLEY_SPEED: float = 1.2     # m/s
GANTRY_ACCELERATION: float = 40.0
PICK_UP_TIME: float = 30.0

# Crane default positions (truck at stack -1)
TRUCK_STACK_INDEX: int = -1

# Protected division threshold
DIVISION_EPSILON: float = 1e-9

# Sentinels
INT_MAX_SENTINEL: int = 2147483647
UNKNOWN_CONTAINER_SENTINEL: int = -1

# Terminal sets defined in Table 2 of Đurasević et al. (2026)
CORE_TERMINALS: tuple[str, ...] = ("SH", "AVG", "EMP", "CUR", "RI", "DIFF")

EXTENDED_TERMINALS: tuple[str, ...] = (
    "DUR", "MIN", "REM", "NEXT", "EMPTY", "WL", "NL", "DSM"
)

ONLINE_TERMINALS_HORIZON: tuple[str, ...] = (
    "NUNC", "NPER", "NRAT"
)

ONLINE_TERMINALS_GROUPS: tuple[str, ...] = (
    "NUNC", "NPER", "NRAT", "NGRP", "GRPS"
)

# Experimental terminal sets for Limited Horizon (Total: C=6, C+E=14, C+O=9, C+E+O=17)
HORIZON_TERMINAL_SETS: dict[str, tuple[str, ...]] = {
    "C": CORE_TERMINALS,
    "C+E": CORE_TERMINALS + EXTENDED_TERMINALS,
    "C+O": CORE_TERMINALS + ONLINE_TERMINALS_HORIZON,
    "C+E+O": CORE_TERMINALS + EXTENDED_TERMINALS + ONLINE_TERMINALS_HORIZON,
}

# Experimental terminal sets for Container Groups (Total: C=6, C+E=14, C+O=11, C+E+O=19)
GROUP_TERMINAL_SETS: dict[str, tuple[str, ...]] = {
    "C": CORE_TERMINALS,
    "C+E": CORE_TERMINALS + EXTENDED_TERMINALS,
    "C+O": CORE_TERMINALS + ONLINE_TERMINALS_GROUPS,
    "C+E+O": CORE_TERMINALS + EXTENDED_TERMINALS + ONLINE_TERMINALS_GROUPS,
}

# Genetic Programming Defaults (Table 3)
GP_POPULATION_SIZE: int = 1000
GP_MAX_TREE_DEPTH: int = 5
GP_TOURNAMENT_SIZE: int = 3
GP_MUTATION_RATE: float = 0.3
GP_CROSSOVER_RATE: float = 1.0
GP_TOTAL_EVALUATIONS: int = 50000

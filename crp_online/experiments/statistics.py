import numpy as np
from typing import Dict, Sequence, Tuple


def compute_distribution_stats(values: Sequence[float], num_bootstrap: int = 1000) -> Dict[str, float]:
    """Computes sample statistics and bootstrap confidence intervals.
    
    Returns:
        dict containing 'mean', 'std', 'median', 'q25', 'q75', 'iqr', 'ci95_low', 'ci95_high'.
    """
    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return {}

    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    median_val = float(np.median(arr))
    q25 = float(np.percentile(arr, 25))
    q75 = float(np.percentile(arr, 75))
    iqr = q75 - q25

    # Bootstrap 95% confidence interval for the median
    if n > 1:
        rng = np.random.default_rng(42)
        indices = rng.integers(0, n, size=(num_bootstrap, n))
        boot_medians = np.median(arr[indices], axis=1)
        ci_low = float(np.percentile(boot_medians, 2.5))
        ci_high = float(np.percentile(boot_medians, 97.5))
    else:
        ci_low = median_val
        ci_high = median_val

    return {
        "mean": mean_val,
        "std": std_val,
        "median": median_val,
        "q25": q25,
        "q75": q75,
        "iqr": iqr,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


def is_equivalent(
    actual: float,
    target: float,
    rel_tol: float = 0.02,
) -> bool:
    """Checks whether the actual value falls within the relative tolerance band of target.
    
    Default tolerance is 2.0% matching the Tier 4 Acceptance Contract in design_document.md.
    """
    if target == 0.0:
        return actual == 0.0
    return abs(actual - target) / target <= rel_tol

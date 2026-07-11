"""Statistics for comparing controllers.

All controllers are run on the same held-out seeds, so each seed is a paired
sample: the same cars arrive at the same times for every controller. Paired
differences remove most of the traffic-to-traffic noise.
Confidence intervals are percentile bootstrap intervals (Efron and Tibshirani, 1993).
"""
import numpy as np

N_BOOT = 10_000


def bootstrap_ci(values, level=0.95, seed=0):
    """Mean and its bootstrap confidence interval."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, (N_BOOT, len(values))).mean(axis=1)
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(values.mean()), float(lo), float(hi)


def paired_difference(a, b, level=0.95, seed=0):
    """Mean of (a - b) per seed with its bootstrap CI. CI entirely below 0 means a is reliably lower."""
    return bootstrap_ci(np.asarray(a, dtype=float) - np.asarray(b, dtype=float), level, seed)


def percent_change(new, old):
    return 100.0 * (new - old) / old if old else float('nan')

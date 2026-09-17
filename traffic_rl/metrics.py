"""Statistics for comparing controllers.

All controllers are run on the same held-out seeds, so each seed is a paired
sample: the same cars arrive at the same times for every controller. Paired
differences remove most of the traffic-to-traffic noise.
Confidence intervals are percentile bootstrap intervals (Efron and Tibshirani, 1993).

A learner is trained several times (independent training runs), and a single
run can be lucky or unlucky. The *_runs functions take a (runs x seeds) array
and resample at two levels, first training runs and then traffic seeds, so
their intervals cover both sources of randomness (a hierarchical bootstrap).
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


def two_level_ci(x, level=0.95, seed=0):
    """Mean of a (runs x seeds) array and its two-level bootstrap CI. Each draw picks
    runs with replacement, then seeds with replacement (the same seeds for every run,
    so paired differences stay paired)."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    rng = np.random.default_rng(seed)
    n_runs, n_seeds = x.shape
    runs = rng.integers(n_runs, size=(N_BOOT, n_runs))
    seeds = rng.integers(n_seeds, size=(N_BOOT, n_seeds))
    means = x[runs[:, :, None], seeds[:, None, :]].mean(axis=(1, 2))
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(x.mean()), float(lo), float(hi)


def bootstrap_ci_runs(values, level=0.95, seed=0):
    """Mean and CI of values from several training runs: shape (runs, seeds).
    A 1-D array (a controller that needs no training) is treated as a single run."""
    return two_level_ci(values, level, seed)


def paired_difference_runs(a, b, level=0.95, seed=0, pair_runs=True):
    """Mean of (a - b) with its two-level CI, paired by traffic seed.
    a: (runs, seeds) for a trained learner.
    b: (seeds,) for a baseline; or (runs, seeds) for the same runs before training
    (pair_runs=True: run k before vs run k after); or (runs, seeds) for another
    learner, whose runs are unrelated (pair_runs=False: runs resampled separately)."""
    a = np.atleast_2d(np.asarray(a, dtype=float))
    b = np.asarray(b, dtype=float)
    if b.ndim == 1 or pair_runs:
        return two_level_ci(a - b, level, seed)
    rng = np.random.default_rng(seed)
    seeds = rng.integers(a.shape[1], size=(N_BOOT, a.shape[1]))
    ra = rng.integers(a.shape[0], size=(N_BOOT, a.shape[0]))
    rb = rng.integers(b.shape[0], size=(N_BOOT, b.shape[0]))
    diffs = (a[ra[:, :, None], seeds[:, None, :]].mean(axis=(1, 2))
             - b[rb[:, :, None], seeds[:, None, :]].mean(axis=(1, 2)))
    lo, hi = np.percentile(diffs, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(a.mean() - b.mean()), float(lo), float(hi)

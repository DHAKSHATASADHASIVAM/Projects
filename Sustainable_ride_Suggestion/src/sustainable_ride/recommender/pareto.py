"""Pareto dominance over the (cost, time, CO2) objective space.

A weighted score alone is not a satisfying answer to "which ride should I
take?", because the weights are made up. Someone who says they care equally
about all three objectives has not really told you their utility function, and
small changes to arbitrary weights can flip the ranking.

The Pareto frontier is weight-free. An option is dominated when some other
option is at least as good on every objective and strictly better on one --
which means no rational traveller would ever choose it, whatever their
preferences. That is a much stronger statement than "it scored lower", and it
is the honest way to narrow a decision before applying any subjective trade-off.

So the engine reports both: the frontier (defensible, weight-independent) and
the weighted ranking within it (useful, and clearly labelled as preference-based).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def dominates(a: Sequence[float], b: Sequence[float],
              tolerance: float = 1e-9) -> bool:
    """Whether ``a`` Pareto-dominates ``b``. All objectives are minimised.

    Dominance requires being no worse on every objective and strictly better on
    at least one. The tolerance stops floating-point dust from registering as a
    real advantage -- without it, a 10^-15 difference in predicted CO2 would
    "dominate" an otherwise identical option.
    """
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    if a.shape != b.shape:
        raise ValueError(f"objective vectors differ in length: {a.shape} vs {b.shape}")

    no_worse = np.all(a <= b + tolerance)
    strictly_better = np.any(a < b - tolerance)
    return bool(no_worse and strictly_better)


def pareto_frontier(objectives: Sequence[Sequence[float]],
                    tolerance: float = 1e-9) -> list[int]:
    """Indices of the non-dominated rows in an objective matrix.

    O(n^2), which is entirely fine: n is the number of travel modes, so it is 3.
    """
    matrix = np.asarray(objectives, dtype="float64")
    if matrix.ndim != 2:
        raise ValueError("objectives must be a 2-D array of shape (n_options, n_objectives)")

    n = matrix.shape[0]
    frontier = []
    for i in range(n):
        if not any(dominates(matrix[j], matrix[i], tolerance)
                   for j in range(n) if j != i):
            frontier.append(i)
    return frontier


def normalise(values: Sequence[float]) -> np.ndarray:
    """Min-max normalise to [0, 1], where 0 is best (all objectives minimised).

    When every option ties -- three modes with identical CO2, say -- min-max
    would divide by zero. Returning all zeros is the right answer there: a
    dimension on which nothing differs should contribute nothing to the ranking,
    rather than blowing up or arbitrarily favouring one option.
    """
    values = np.asarray(values, dtype="float64")
    if values.size == 0:
        return values

    lo, hi = np.nanmin(values), np.nanmax(values)
    if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(hi, lo):
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def weighted_scores(cost: Sequence[float], time: Sequence[float],
                    co2: Sequence[float], weights: dict[str, float]) -> np.ndarray:
    """Combine normalised objectives into one score. Lower is better.

    Weights are normalised to sum to 1 so that a user passing ``{cost: 2, time: 1,
    co2: 1}`` gets the intuitive result rather than a score on a different scale
    from everyone else's.
    """
    w_cost = float(weights.get("cost", 0.0))
    w_time = float(weights.get("time", 0.0))
    w_co2 = float(weights.get("co2", 0.0))

    total = w_cost + w_time + w_co2
    if total <= 0:
        raise ValueError("At least one of the cost/time/co2 weights must be positive")
    w_cost, w_time, w_co2 = w_cost / total, w_time / total, w_co2 / total

    return (w_cost * normalise(cost)
            + w_time * normalise(time)
            + w_co2 * normalise(co2))

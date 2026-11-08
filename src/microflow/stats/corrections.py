"""Multiple-comparison correction. Pure numerics.

A family is the set of tests that answer the same question across the run: one p-value column
(Welch, Mann-Whitney or permutation) across every measure and comparison. Each family is corrected
separately. Missing p-values (NaN, from tests that could not be computed) are not part of the family
and stay missing.
"""

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

Rule = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values (false discovery rate)."""
    return _adjust(p_values, _benjamini_hochberg)


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values (family-wise error rate)."""
    return _adjust(p_values, _holm)


def adjust(p_values: Sequence[float], method: str) -> list[float]:
    if method == "bh":
        return benjamini_hochberg(p_values)
    if method == "holm":
        return holm(p_values)
    if method == "none":
        return [float(value) for value in p_values]
    raise ValueError(f"unknown correction {method!r}; use bh, holm or none")


def _adjust(p_values: Sequence[float], rule: Rule) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    finite = np.flatnonzero(np.isfinite(values))
    result = np.full(values.shape, np.nan)
    if finite.size:
        result[finite] = rule(values[finite])
    return [float(value) for value in result]


def _benjamini_hochberg(p: NDArray[np.float64]) -> NDArray[np.float64]:
    count = p.size
    order = np.argsort(p, kind="stable")
    ranked = p[order] * count / np.arange(1, count + 1)
    smoothed = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(count)
    adjusted[order] = np.minimum(smoothed, 1.0)
    return adjusted


def _holm(p: NDArray[np.float64]) -> NDArray[np.float64]:
    count = p.size
    order = np.argsort(p, kind="stable")
    factors = count - np.arange(count)
    ranked = p[order] * factors
    smoothed = np.maximum.accumulate(ranked)
    adjusted = np.empty(count)
    adjusted[order] = np.minimum(smoothed, 1.0)
    return adjusted

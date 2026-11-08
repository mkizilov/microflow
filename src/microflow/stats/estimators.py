"""Estimators for comparing groups of biological samples. Numerical code only; no I/O.

Every function here takes sample-level values, one per biological sample. Objects never reach these
functions, which is what keeps the sample size honest.

Interval estimates use the percentile bootstrap, resampling samples with replacement within each
group. Effect sizes use Hedges' g. Two-sided p-values come from Welch's t-test, the Mann-Whitney U
test, and a two-sided permutation test on the difference in means. Where a value cannot be computed
(too few samples, zero variance), the function returns NaN rather than a guess.
"""

import math
import warnings
from collections.abc import Iterator
from itertools import combinations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

Values = NDArray[np.float64]


def mean_and_sd(values: Values) -> tuple[float, float]:
    """Mean and sample standard deviation (ddof 1). The SD is NaN for fewer than two values."""
    if values.size == 0:
        return float("nan"), float("nan")
    sd = float(values.std(ddof=1)) if values.size > 1 else float("nan")
    return float(values.mean()), sd


def bootstrap_mean_interval(
    values: Values, *, rng: np.random.Generator, resamples: int, confidence: float
) -> tuple[float, float]:
    if values.size < 2:
        return float("nan"), float("nan")
    draws = values[rng.integers(0, values.size, size=(resamples, values.size))].mean(axis=1)
    return _interval(draws, confidence)


def bootstrap_difference_interval(
    first: Values,
    second: Values,
    *,
    rng: np.random.Generator,
    resamples: int,
    confidence: float,
) -> tuple[float, float]:
    """Interval for mean(first) - mean(second), resampling each group independently."""
    if first.size < 2 or second.size < 2:
        return float("nan"), float("nan")
    draws_a = first[rng.integers(0, first.size, size=(resamples, first.size))].mean(axis=1)
    draws_b = second[rng.integers(0, second.size, size=(resamples, second.size))].mean(axis=1)
    return _interval(draws_a - draws_b, confidence)


def hedges_g(first: Values, second: Values) -> float:
    """Hedges' g: the standardised mean difference, pooled SD, with the small-sample correction."""
    scale = _hedges_scale(first.size, second.size)
    if scale is None:
        return float("nan")
    pooled = _pooled_sd(first.std(ddof=1), first.size, second.std(ddof=1), second.size)
    if not pooled > 0:
        return float("nan")
    return scale * float((first.mean() - second.mean()) / pooled)


def bootstrap_hedges_interval(
    first: Values,
    second: Values,
    *,
    rng: np.random.Generator,
    resamples: int,
    confidence: float,
) -> tuple[float, float, float]:
    """Interval for Hedges' g, and the fraction of resamples that gave a finite value. Resamples
    where a group has zero variance have no finite g and are dropped."""
    scale = _hedges_scale(first.size, second.size)
    if scale is None:
        return float("nan"), float("nan"), 0.0
    draws_a = first[rng.integers(0, first.size, size=(resamples, first.size))]
    draws_b = second[rng.integers(0, second.size, size=(resamples, second.size))]
    with np.errstate(divide="ignore", invalid="ignore"):
        pooled = np.sqrt(
            ((first.size - 1) * draws_a.var(axis=1, ddof=1)
             + (second.size - 1) * draws_b.var(axis=1, ddof=1))
            / (first.size + second.size - 2)
        )
        g = scale * (draws_a.mean(axis=1) - draws_b.mean(axis=1)) / pooled
    finite = g[np.isfinite(g)]
    usable = finite.size / resamples
    if finite.size < 2:
        return float("nan"), float("nan"), usable
    low, high = _interval(finite, confidence)
    return low, high, usable


def welch_p(first: Values, second: Values) -> float:
    """Two-sided p-value of Welch's t-test (unequal variances)."""
    if first.size < 2 or second.size < 2:
        return float("nan")
    if first.var(ddof=1) == 0 and second.var(ddof=1) == 0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = stats.ttest_ind(first, second, equal_var=False)
    return float(result.pvalue)


def mann_whitney(first: Values, second: Values) -> tuple[float, float]:
    """Two-sided Mann-Whitney U p-value, and the rank-biserial correlation (+1 when every value of
    ``first`` exceeds every value of ``second``, -1 for the reverse)."""
    if first.size < 1 or second.size < 1:
        return float("nan"), float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = stats.mannwhitneyu(first, second, alternative="two-sided")
    rank_biserial = 2 * float(result.statistic) / (first.size * second.size) - 1
    return float(result.pvalue), rank_biserial


def permutation_p(
    first: Values,
    second: Values,
    *,
    rng: np.random.Generator,
    resamples: int,
    max_exact: int,
) -> tuple[float, str]:
    """Two-sided permutation p-value for the difference in means, with the group labels shuffled
    among the samples. Returns the p-value and how it was obtained: ``exact`` enumerates every
    split,
    ``monte_carlo`` draws random splits and uses the (k + 1) / (B + 1) estimator, which never
    reports
    zero."""
    na, nb = first.size, second.size
    if na < 1 or nb < 1:
        return float("nan"), "not_applicable"
    pooled = np.concatenate([first, second])
    total = pooled.sum()
    observed = abs(float(first.mean() - second.mean()))
    tolerance = 1e-12 * max(1.0, observed)
    splits = math.comb(na + nb, na)
    if splits <= max_exact:
        count = 0
        for index in _splits(na + nb, na):
            selected = pooled[list(index)].sum()
            difference = selected / na - (total - selected) / nb
            if abs(difference) >= observed - tolerance:
                count += 1
        return count / splits, "exact"
    count = 0
    for _ in range(resamples):
        shuffled = rng.permutation(pooled)
        difference = abs(shuffled[:na].mean() - shuffled[na:].mean())
        if difference >= observed - tolerance:
            count += 1
    return (count + 1) / (resamples + 1), "monte_carlo"


def _splits(n: int, size: int) -> Iterator[tuple[int, ...]]:
    return combinations(range(n), size)


def _interval(draws: Values, confidence: float) -> tuple[float, float]:
    tail = (1 - confidence) / 2
    low, high = np.quantile(draws, [tail, 1 - tail])
    return float(low), float(high)


def _hedges_scale(na: int, nb: int) -> float | None:
    if na < 2 or nb < 2:
        return None
    return 1 - 3 / (4 * (na + nb) - 9)


def _pooled_sd(sd_a: float, na: int, sd_b: float, nb: int) -> float:
    return math.sqrt(((na - 1) * sd_a**2 + (nb - 1) * sd_b**2) / (na + nb - 2))

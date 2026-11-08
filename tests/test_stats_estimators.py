import math

import numpy as np
import pytest
from scipy import stats

from microflow.stats.estimators import (
    bootstrap_difference_interval,
    bootstrap_hedges_interval,
    bootstrap_mean_interval,
    hedges_g,
    mann_whitney,
    mean_and_sd,
    permutation_p,
    welch_p,
)


def test_mean_and_sample_standard_deviation_of_known_values() -> None:
    mean, sd = mean_and_sd(np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]))
    assert mean == pytest.approx(5.0)
    # Population SD is 2; the sample SD (ddof 1) is sqrt(32 / 7).
    assert sd == pytest.approx(math.sqrt(32 / 7))


def test_a_single_value_has_a_mean_but_no_standard_deviation() -> None:
    mean, sd = mean_and_sd(np.array([3.0]))
    assert mean == 3.0
    assert math.isnan(sd)
    assert all(math.isnan(v) for v in mean_and_sd(np.array([])))


def test_hedges_g_matches_its_closed_form_for_a_small_pair() -> None:
    # Groups [1, 2] and [3, 4]: means 1.5 and 3.5. Each sample variance is 0.5 (ddof 1), so the
    # pooled SD is sqrt(0.5). The small-sample factor is 1 - 3 / (4 * 4 - 9) = 1 - 3/7 = 4/7.
    # g = (4/7) * (1.5 - 3.5) / sqrt(0.5) = -(8/7) * sqrt(2) = -1.6165.
    first = np.array([1.0, 2.0])
    second = np.array([3.0, 4.0])
    expected = -(8 / 7) * math.sqrt(2)
    assert hedges_g(first, second) == pytest.approx(expected)
    assert expected == pytest.approx(-1.6165, abs=1e-4)


def test_hedges_g_is_the_mean_difference_in_pooled_sd_with_bias_correction() -> None:
    first = np.array([10.0, 12.0, 14.0, 16.0])
    second = np.array([8.0, 9.0, 10.0, 11.0])
    raw = (first.mean() - second.mean()) / math.sqrt(
        ((first.size - 1) * first.var(ddof=1) + (second.size - 1) * second.var(ddof=1))
        / (first.size + second.size - 2)
    )
    correction = 1 - 3 / (4 * (first.size + second.size) - 9)
    assert hedges_g(first, second) == pytest.approx(correction * raw)
    assert hedges_g(first, second) > 0


def test_hedges_g_is_undefined_without_two_samples_per_group_or_any_spread() -> None:
    assert math.isnan(hedges_g(np.array([1.0]), np.array([2.0, 3.0])))
    assert math.isnan(hedges_g(np.array([1.0, 1.0]), np.array([1.0, 1.0])))


def test_welch_p_is_the_unequal_variance_t_test_and_undefined_for_constant_groups() -> None:
    first = np.array([1.0, 2.0, 3.0, 4.0])
    second = np.array([2.0, 3.0, 4.0, 9.0])
    # Unequal variances, so Welch's test must differ from the equal-variance test.
    welch = stats.ttest_ind(first, second, equal_var=False).pvalue
    student = stats.ttest_ind(first, second, equal_var=True).pvalue
    assert welch_p(first, second) == pytest.approx(welch)
    assert welch != pytest.approx(student)
    assert math.isnan(welch_p(np.array([1.0, 1.0]), np.array([1.0, 1.0])))


def test_mann_whitney_detects_complete_separation_with_rank_biserial_one() -> None:
    first = np.array([4.0, 5.0, 6.0])
    second = np.array([1.0, 2.0, 3.0])
    p_value, rank_biserial = mann_whitney(first, second)
    assert rank_biserial == pytest.approx(1.0)
    assert p_value == pytest.approx(stats.mannwhitneyu(first, second).pvalue)
    assert mann_whitney(second, first)[1] == pytest.approx(-1.0)


def test_exact_permutation_p_is_the_proportion_of_splits_at_least_as_extreme() -> None:
    # Complete separation of 3 versus 3: the observed split and the one that swaps the two groups
    # reach the observed absolute difference. Of C(6, 3) = 20 splits, that is 2 / 20.
    rng = np.random.default_rng(0)
    p_value, method = permutation_p(
        np.array([4.0, 5.0, 6.0]), np.array([1.0, 2.0, 3.0]), rng=rng, resamples=100, max_exact=100
    )
    assert method == "exact"
    assert p_value == pytest.approx(2 / 20)


def test_monte_carlo_permutation_p_never_reports_zero() -> None:
    rng = np.random.default_rng(0)
    p_value, method = permutation_p(
        np.arange(10.0) + 100, np.arange(10.0), rng=rng, resamples=200, max_exact=10
    )
    assert method == "monte_carlo"
    assert p_value >= 1 / 201


def test_permutation_p_is_reproducible_for_a_seed() -> None:
    first = np.array([1.0, 2.5, 3.1, 4.0, 5.2, 6.1])
    second = np.array([2.0, 3.0, 2.8, 4.4])
    values = [
        permutation_p(first, second, rng=np.random.default_rng(7), resamples=500, max_exact=5)
        for _ in range(2)
    ]
    assert values[0] == values[1]


def test_bootstrap_interval_of_a_constant_sample_collapses_to_the_constant() -> None:
    low, high = bootstrap_mean_interval(
        np.full(5, 3.0), rng=np.random.default_rng(0), resamples=500, confidence=0.95
    )
    assert low == high == pytest.approx(3.0)


def test_bootstrap_intervals_bracket_the_estimate_and_widen_with_confidence() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    narrow = bootstrap_mean_interval(
        values, rng=np.random.default_rng(1), resamples=2000, confidence=0.5
    )
    wide = bootstrap_mean_interval(
        values, rng=np.random.default_rng(1), resamples=2000, confidence=0.95
    )
    assert narrow[0] <= values.mean() <= narrow[1]
    assert wide[1] - wide[0] > narrow[1] - narrow[0]


def test_bootstrap_difference_interval_covers_the_true_difference_for_clear_groups() -> None:
    first = np.array([10.0, 10.5, 9.8, 10.2, 10.1])
    second = np.array([5.0, 5.2, 4.9, 5.1, 4.8])
    low, high = bootstrap_difference_interval(
        first, second, rng=np.random.default_rng(2), resamples=2000, confidence=0.95
    )
    assert 4.5 < low < high < 5.8


def test_bootstrap_hedges_reports_how_many_resamples_gave_a_finite_value() -> None:
    # Groups of two identical values have zero spread, so no resample can give a finite g.
    constant = np.array([1.0, 1.0])
    low, high, usable = bootstrap_hedges_interval(
        constant, constant, rng=np.random.default_rng(3), resamples=300, confidence=0.9
    )
    assert usable == 0.0
    assert math.isnan(low) and math.isnan(high)
    varied = np.array([1.0, 2.0, 3.0])
    _, _, usable_varied = bootstrap_hedges_interval(
        varied, varied + 0.5, rng=np.random.default_rng(3), resamples=300, confidence=0.9
    )
    assert usable_varied > 0.9


def test_intervals_need_at_least_two_values() -> None:
    rng = np.random.default_rng(0)
    assert all(
        math.isnan(v)
        for v in bootstrap_mean_interval(np.array([1.0]), rng=rng, resamples=100, confidence=0.9)
    )

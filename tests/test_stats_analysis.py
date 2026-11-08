"""Analysis on the hand-checkable fixture study.

The fixture has four samples with per-sample mean areas c1 = 10 and c2 = 12 (control), and t1 = 16
and t2 = 18 (treated). Each sample has two images, and the images hold different numbers of
objects, so pooling objects would give different answers. These tests check the sample-level
analysis.
"""

import math
from pathlib import Path

import pytest
from scipy import stats

from microflow.stats.analysis import ComparisonRow, DescriptiveRow, run_analysis
from microflow.stats.data import ObjectTable, read_measurements
from microflow.stats.design import StatsDesign
from stats_fixtures import AREAS, design_mapping, write_dictionary, write_measurements


def _study(
    tmp_path: Path,
    areas: dict[str, list[float]] = AREAS,
    *,
    measures: list[str] | None = None,
    **overrides: object,
) -> tuple[StatsDesign, ObjectTable]:
    write_measurements(tmp_path, areas)
    write_dictionary(tmp_path)
    mapping = design_mapping(tmp_path, **overrides)
    if measures is not None:
        mapping["measures"] = measures
    design = StatsDesign.model_validate(mapping)
    table = read_measurements(
        design.measurements, measures=design.measures, dictionary=design.data_dictionary
    )
    return design, table


def _descriptive(rows: tuple[DescriptiveRow, ...], group: str) -> DescriptiveRow:
    return next(row for row in rows if row.group == group)


def _comparison(tmp_path: Path, **overrides: object) -> ComparisonRow:
    design, table = _study(tmp_path, **overrides)
    return run_analysis(design, table).comparisons[0]


def test_descriptive_statistics_summarise_the_two_sample_means(tmp_path: Path) -> None:
    design, table = _study(tmp_path)
    control = _descriptive(run_analysis(design, table).descriptive, "control")
    # Sample means 10 and 12: mean 11, sample SD sqrt(2), standard error 1.
    assert control.samples == 2
    assert control.sample_mean == pytest.approx(11.0)
    assert control.sample_sd == pytest.approx(math.sqrt(2))
    assert control.sem == pytest.approx(1.0)
    assert control.sample_median == pytest.approx(11.0)
    assert control.items == 4
    assert control.objects == 6


def test_object_counts_do_not_inflate_the_sample_size(tmp_path: Path) -> None:
    areas = dict(AREAS)
    areas["c1_a"] = [8.0] * 50 + [10.0] * 50
    design, table = _study(tmp_path, areas)
    result = run_analysis(design, table)
    control = _descriptive(result.descriptive, "control")
    # Control objects: 100 in c1_a, 1 in c1_b and 3 in c2 = 104. The samples are still two.
    assert control.samples == 2
    assert control.objects == 104
    comparison = result.comparisons[0]
    assert comparison.samples_a == 2
    assert comparison.objects_a == 104


def test_the_comparison_is_the_welch_test_on_the_sample_means(tmp_path: Path) -> None:
    comparison = _comparison(tmp_path)
    expected = stats.ttest_ind([10.0, 12.0], [16.0, 18.0], equal_var=False).pvalue
    assert comparison.mean_a == pytest.approx(11.0)
    assert comparison.mean_b == pytest.approx(17.0)
    assert comparison.difference == pytest.approx(-6.0)
    assert comparison.p_welch == pytest.approx(expected)


def test_the_difference_interval_brackets_the_observed_difference(tmp_path: Path) -> None:
    comparison = _comparison(tmp_path)
    assert comparison.difference_ci_low <= comparison.difference <= comparison.difference_ci_high
    assert comparison.difference_ci_high < 0


def test_a_group_with_too_few_samples_is_reported_and_not_tested(tmp_path: Path) -> None:
    # Control has one sample holding all four control images, so it cannot be tested.
    groups = {
        "control": {"c1": ["c1_a", "c1_b", "c2_a", "c2_b"]},
        "treated": {"t1": ["t1_a", "t1_b"], "t2": ["t2_a", "t2_b"]},
    }
    design, table = _study(tmp_path, groups=groups)
    result = run_analysis(design, table)
    comparison = result.comparisons[0]
    assert "insufficient_replication" in comparison.flags
    assert math.isnan(comparison.p_welch)
    assert math.isnan(comparison.hedges_g)
    assert any("not tested" in warning for warning in result.warnings)


def test_a_sample_with_no_valid_objects_is_excluded_and_reported(tmp_path: Path) -> None:
    areas = dict(AREAS)
    areas["c1_a"] = [math.nan]
    areas["c1_b"] = [math.nan]
    design, table = _study(tmp_path, areas)
    result = run_analysis(design, table)
    control = _descriptive(result.descriptive, "control")
    assert control.samples == 1
    assert "excluded_samples:1" in control.flags
    excluded = [row for row in result.samples if row.sample == "c1"]
    assert len(excluded) == 1 and not excluded[0].included
    assert any("excluded" in warning for warning in result.warnings)


def test_median_aggregation_uses_each_sample_median(tmp_path: Path) -> None:
    design, table = _study(tmp_path, per_sample="median")
    values = {row.sample: row.value for row in run_analysis(design, table).samples}
    # c1 holds 8, 10 and 12 (median 10); c2 holds 11, 13 and 12 (median 12).
    assert values["c1"] == pytest.approx(10.0)
    assert values["c2"] == pytest.approx(12.0)


def test_corrections_are_applied_within_each_test_family(tmp_path: Path) -> None:
    design, table = _study(tmp_path)
    result = run_analysis(design, table)
    comparison = result.comparisons[0]
    # One comparison and one measure: each family holds a single p-value, so BH leaves it unchanged.
    assert comparison.p_welch_adjusted == pytest.approx(comparison.p_welch)
    assert comparison.p_mannwhitney_adjusted == pytest.approx(comparison.p_mannwhitney)
    assert result.families == {"welch": 1, "mannwhitney": 1, "permutation": 1}


def test_the_result_is_reproducible_for_the_same_design(tmp_path: Path) -> None:
    design, table = _study(tmp_path)
    first = run_analysis(design, table)
    second = run_analysis(design, table)
    assert first.descriptive == second.descriptive
    assert first.comparisons == second.comparisons


def test_bootstrap_intervals_do_not_depend_on_the_order_of_measures(tmp_path: Path) -> None:
    design, table = _study(tmp_path, measures=["area_um2", "eccentricity"])
    together = run_analysis(design, table)
    reversed_design = design.model_copy(update={"measures": ("eccentricity", "area_um2")})
    reversed_result = run_analysis(reversed_design, table)
    before = _descriptive(together.descriptive, "control")
    after = next(
        row
        for row in reversed_result.descriptive
        if row.measure == "area_um2" and row.group == "control"
    )
    assert before.mean_ci_low == after.mean_ci_low
    assert before.mean_ci_high == after.mean_ci_high


def test_pooling_objects_would_give_a_different_answer_than_the_sample_level_analysis(
    tmp_path: Path,
) -> None:
    # Control: two samples with one object each, and c2 has 60 objects. Object-level pooling would
    # weight the control mean towards 100; the sample-level analysis must use the sample means 1
    # and 100 (mean 50.5), and treated means 10 and 20 (mean 15).
    areas = {
        "c1_a": [1.0],
        "c1_b": [1.0],
        "c2_a": [100.0] * 30,
        "c2_b": [100.0] * 30,
        "t1_a": [10.0] * 40,
        "t1_b": [10.0] * 40,
        "t2_a": [20.0] * 40,
        "t2_b": [20.0] * 40,
    }
    design, table = _study(tmp_path, areas)
    comparison = run_analysis(design, table).comparisons[0]
    pooled_control = (2 * 1.0 + 60 * 100.0) / 62
    assert comparison.samples_a == 2
    assert comparison.mean_a == pytest.approx(50.5)
    assert comparison.mean_a != pytest.approx(pooled_control)
    assert comparison.mean_b == pytest.approx(15.0)

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from microflow.config import ConfigError
from microflow.stats.data import check_assignments, read_measurements, sample_values
from microflow.stats.design import StatsDesign, load_design
from microflow.stats.errors import StatsError
from stats_fixtures import (
    AREAS,
    design_mapping,
    write_design,
    write_dictionary,
    write_measurements,
)


def _design(**changes: Any) -> dict[str, Any]:
    return design_mapping(Path("."), **changes)


def test_a_valid_design_loads_and_resolves_paths_against_its_directory(tmp_path: Path) -> None:
    write_measurements(tmp_path)
    write_dictionary(tmp_path)
    design = load_design(write_design(tmp_path, design_mapping(tmp_path)))
    assert design.measurements == tmp_path / "measurements.csv"
    assert design.output_directory == tmp_path / "results"
    assert design.groups["control"].samples[0].sample == "c1"


def test_a_sample_cannot_belong_to_two_groups() -> None:
    groups = {
        "control": {"c1": ["c1_a"], "c2": ["c2_a"]},
        "treated": {"c1": ["t1_a"], "t2": ["t2_a"]},
    }
    with pytest.raises(ValidationError, match="more than one group"):
        StatsDesign.model_validate(_design(groups=groups))


def test_an_image_cannot_belong_to_two_samples() -> None:
    groups = {
        "control": {"c1": ["c1_a"], "c2": ["c1_a"]},
        "treated": {"t1": ["t1_a"], "t2": ["t2_a"]},
    }
    with pytest.raises(ValidationError, match="more than one sample"):
        StatsDesign.model_validate(_design(groups=groups))


def test_a_comparison_needs_two_declared_groups_and_no_self_comparison() -> None:
    with pytest.raises(ValidationError, match="at least two groups"):
        StatsDesign.model_validate(
            _design(groups={"control": {"c1": ["c1_a"]}}, comparisons=[["control", "control"]])
        )
    with pytest.raises(ValidationError, match="undeclared group"):
        StatsDesign.model_validate(_design(comparisons=[["control", "missing"]]))
    with pytest.raises(ValidationError, match="compares a group with itself"):
        StatsDesign.model_validate(_design(comparisons=[["control", "control"]]))


def test_the_same_comparison_listed_twice_in_either_order_is_refused() -> None:
    with pytest.raises(ValidationError, match="listed more than once"):
        StatsDesign.model_validate(
            _design(comparisons=[["control", "treated"], ["treated", "control"]])
        )


def test_sample_and_group_names_are_restricted_and_repeats_are_refused() -> None:
    with pytest.raises(ValidationError, match="String should match pattern"):
        StatsDesign.model_validate(
            _design(groups={"control": {"bad name": ["c1_a"]}, "treated": {"t1": ["t1_a"]}})
        )
    with pytest.raises(ValidationError, match="letters, digits"):
        StatsDesign.model_validate(
            _design(groups={"control/x": {"c1": ["c1_a"]}, "treated": {"t1": ["t1_a"]}})
        )
    with pytest.raises(ValidationError, match="listed once"):
        StatsDesign.model_validate(_design(measures=["area_um2", "area_um2"]))


def test_unknown_keys_and_bad_settings_are_reported_with_their_location(tmp_path: Path) -> None:
    bad_choice = tmp_path / "choice"
    bad_choice.mkdir()
    with pytest.raises(ConfigError, match="correction"):
        load_design(write_design(bad_choice, _design(correction="bonferroni")))
    unknown_key = tmp_path / "key"
    unknown_key.mkdir()
    with pytest.raises(ConfigError, match="surprise"):
        load_design(write_design(unknown_key, _design(surprise=1)))


def test_bootstrap_and_permutation_settings_have_limits() -> None:
    with pytest.raises(ValidationError):
        StatsDesign.model_validate(_design(bootstrap={"resamples": 10}))
    with pytest.raises(ValidationError):
        StatsDesign.model_validate(_design(bootstrap={"confidence": 1.0}))


def test_every_image_must_be_assigned_and_every_design_image_must_be_measured(
    tmp_path: Path,
) -> None:
    measurements = write_measurements(tmp_path)
    design = StatsDesign.model_validate(
        _design(groups={"control": {"c1": ["c1_a"]}, "treated": {"t1": ["t1_a"]}})
    )
    table = read_measurements(measurements, measures=["area_um2"])
    with pytest.raises(StatsError, match="belong to no sample"):
        check_assignments(table, design)

    partial_dir = tmp_path / "partial"
    partial_dir.mkdir()
    partial = dict(AREAS)
    del partial["c2_b"]
    measurements = write_measurements(partial_dir, partial)
    full = StatsDesign.model_validate(_design())
    with pytest.raises(StatsError, match="have no measurements"):
        check_assignments(read_measurements(measurements, measures=["area_um2"]), full)


def test_a_measurement_table_needs_an_item_column_and_the_requested_measures(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("name,area_um2\nx,1\n", encoding="utf-8")
    with pytest.raises(StatsError, match="'item' column"):
        read_measurements(path, measures=["area_um2"])
    path.write_text("item,area_um2\nx,1\n", encoding="utf-8")
    with pytest.raises(StatsError, match="no column for measure"):
        read_measurements(path, measures=["eccentricity"])


def test_a_non_numeric_value_is_reported_with_its_row(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("item,area_um2\nx,oops\n", encoding="utf-8")
    with pytest.raises(StatsError, match="row 2, area_um2: 'oops' is not a number"):
        read_measurements(path, measures=["area_um2"])


def test_a_row_with_the_wrong_field_count_is_refused_rather_than_read_as_missing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "short.csv"
    path.write_text("item,area_um2,eccentricity\nx,1\n", encoding="utf-8")
    with pytest.raises(StatsError, match="different number of fields"):
        read_measurements(path, measures=["area_um2", "eccentricity"])
    path.write_text("item,area_um2\nx,1,2\n", encoding="utf-8")
    with pytest.raises(StatsError, match="different number of fields"):
        read_measurements(path, measures=["area_um2"])


def test_missing_text_becomes_nan_and_is_excluded_from_its_sample(tmp_path: Path) -> None:
    path = tmp_path / "gaps.csv"
    path.write_text("item,area_um2\nc1_a,4\nc1_a,NA\nc1_a,6\n", encoding="utf-8")
    design = StatsDesign.model_validate(
        _design(groups={"control": {"c1": ["c1_a"]}, "treated": {"t1": ["t1_a"]}})
    )
    table = read_measurements(path, measures=["area_um2"])
    entry = sample_values(table, design, "area_um2")[0]
    assert entry.objects == 2
    assert entry.value == pytest.approx(5.0)


def test_infinite_values_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "inf.csv"
    path.write_text("item,area_um2\nx,inf\n", encoding="utf-8")
    with pytest.raises(StatsError, match="infinite"):
        read_measurements(path, measures=["area_um2"])

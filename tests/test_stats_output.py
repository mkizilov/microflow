import csv
import json
from pathlib import Path

import pytest

from microflow.provenance.guard import OutputConflictError
from microflow.stats.errors import StatsError
from microflow.stats.study import run_study
from stats_fixtures import design_mapping, write_design, write_dictionary, write_measurements


def _project(tmp_path: Path, **overrides: object) -> Path:
    write_measurements(tmp_path)
    write_dictionary(tmp_path)
    return write_design(tmp_path, design_mapping(tmp_path, **overrides))


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_a_run_writes_the_tables_and_a_summary_with_counts_and_inputs(tmp_path: Path) -> None:
    design = _project(tmp_path)
    output = run_study(design)
    directory = output.directory
    for name in ("descriptive.csv", "comparisons.csv", "sample_values.csv", "summary.json"):
        assert (directory / name).is_file(), name

    descriptive = _rows(directory / "descriptive.csv")
    assert {row["group"] for row in descriptive} == {"control", "treated"}
    control = next(row for row in descriptive if row["group"] == "control")
    assert control["samples"] == "2"
    assert float(control["sample_mean"]) == pytest.approx(11.0)
    assert control["unit"] == "um2"

    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    assert summary["analysis_id"] == output.identity
    assert summary["counts"]["samples"] == 4
    assert summary["counts"]["comparisons"] == 1
    assert len(summary["inputs"]["measurements_sha256"]) == 64
    assert summary["test_families"] == {"welch": 1, "mannwhitney": 1, "permutation": 1}


def test_missing_numbers_are_written_as_na_not_blank_or_zero(tmp_path: Path) -> None:
    # One control sample cannot be tested, so its statistics must be NA.
    groups = {
        "control": {"c1": ["c1_a", "c1_b", "c2_a", "c2_b"]},
        "treated": {"t1": ["t1_a", "t1_b"], "t2": ["t2_a", "t2_b"]},
    }
    design = _project(tmp_path, groups=groups)
    directory = run_study(design).directory
    comparison = _rows(directory / "comparisons.csv")[0]
    assert comparison["p_welch"] == "NA"
    assert comparison["hedges_g"] == "NA"
    assert "insufficient_replication" in comparison["flags"]


def test_the_same_inputs_give_identical_table_bytes(tmp_path: Path) -> None:
    design = _project(tmp_path)
    first = run_study(design).directory
    first_bytes = {
        name: (first / name).read_bytes() for name in ("descriptive.csv", "comparisons.csv")
    }
    run_study(design, overwrite=True)
    for name, content in first_bytes.items():
        assert (first / name).read_bytes() == content


def test_a_different_design_refuses_to_overwrite_existing_results(tmp_path: Path) -> None:
    design = _project(tmp_path)
    run_study(design)
    changed_design = write_design(
        tmp_path, design_mapping(tmp_path, per_sample="median"), name="changed.yaml"
    )
    with pytest.raises(OutputConflictError, match="different analysis"):
        run_study(changed_design)
    run_study(changed_design, overwrite=True)


def test_invalid_input_leaves_no_results_directory_behind(tmp_path: Path) -> None:
    # The design names an image that has no measurements, so the run must fail before any write.
    groups = {
        "control": {"c1": ["c1_a", "c1_b", "c2_a", "c2_b", "ghost"]},
        "treated": {"t1": ["t1_a", "t1_b"], "t2": ["t2_a", "t2_b"]},
    }
    design = _project(tmp_path, groups=groups)
    with pytest.raises(StatsError, match="have no measurements: ghost"):
        run_study(design)
    assert not (tmp_path / "results").exists()


def test_plots_are_written_only_when_requested(tmp_path: Path) -> None:
    design = _project(tmp_path)
    directory = run_study(design, plots=False).directory
    assert not (directory / "plots").exists()

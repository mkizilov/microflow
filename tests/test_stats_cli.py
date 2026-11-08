from pathlib import Path

from typer.testing import CliRunner

from microflow.cli import app
from stats_fixtures import design_mapping, write_design, write_dictionary, write_measurements

runner = CliRunner()


def _project(tmp_path: Path) -> Path:
    write_measurements(tmp_path)
    write_dictionary(tmp_path)
    return write_design(tmp_path, design_mapping(tmp_path))


def test_check_reports_sample_and_object_counts_without_running_tests(tmp_path: Path) -> None:
    result = runner.invoke(app, ["stats", "check", str(_project(tmp_path))])
    assert result.exit_code == 0, result.output
    assert "group control" in result.output
    assert "2 sample(s), 4 image(s), 6 object row(s)" in result.output
    assert not (tmp_path / "results").exists()


def test_run_prints_each_comparison_with_its_sample_sizes(tmp_path: Path) -> None:
    result = runner.invoke(app, ["stats", "run", str(_project(tmp_path)), "--no-plots"])
    assert result.exit_code == 0, result.output
    assert "area_um2: control vs treated (n = 2, 2)" in result.output
    assert "Hedges' g" in result.output
    assert (tmp_path / "results" / "summary.json").is_file()


def test_run_refuses_a_different_analysis_without_overwrite(tmp_path: Path) -> None:
    first = _project(tmp_path)
    assert runner.invoke(app, ["stats", "run", str(first), "--no-plots"]).exit_code == 0
    other = write_design(
        tmp_path, design_mapping(tmp_path, per_sample="median"), name="median.yaml"
    )
    refused = runner.invoke(app, ["stats", "run", str(other), "--no-plots"])
    assert refused.exit_code == 1
    assert "different analysis" in refused.output
    forced = runner.invoke(app, ["stats", "run", str(other), "--no-plots", "--overwrite"])
    assert forced.exit_code == 0, forced.output


def test_an_invalid_design_fails_with_its_location(tmp_path: Path) -> None:
    path = write_design(tmp_path, design_mapping(tmp_path, correction="bonferroni"))
    result = runner.invoke(app, ["stats", "run", str(path)])
    assert result.exit_code == 1
    assert "correction" in result.output


def test_example_writes_the_two_condition_dataset(tmp_path: Path) -> None:
    result = runner.invoke(app, ["stats", "example", str(tmp_path / "example")])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "example" / "measurements.csv").is_file()
    assert (tmp_path / "example" / "data_dictionary.csv").is_file()

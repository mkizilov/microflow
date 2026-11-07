from pathlib import Path

import pytest
from typer.testing import CliRunner

from microflow import __version__
from microflow.cli import app

runner = CliRunner()


def test_version_command_prints_package_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.output.strip() == __version__


def test_help_lists_sample_data_command() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "make-sample-data" in result.output


def test_make_sample_data_writes_images_and_calibration(tmp_path: Path) -> None:
    out_dir = tmp_path / "synthetic"
    result = runner.invoke(app, ["make-sample-data", "--out-dir", str(out_dir), "--count", "2"])
    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in out_dir.iterdir()) == [
        "synthetic_00.calibration.json",
        "synthetic_00.tif",
        "synthetic_01.calibration.json",
        "synthetic_01.tif",
    ]


@pytest.mark.parametrize("count", ["0", "-1"])
def test_make_sample_data_rejects_non_positive_count(tmp_path: Path, count: str) -> None:
    result = runner.invoke(app, ["make-sample-data", "--out-dir", str(tmp_path), "--count", count])
    assert result.exit_code != 0

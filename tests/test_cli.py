from pathlib import Path

import pytest
from typer.testing import CliRunner

from microflow import __version__
from microflow.cli import app
from microflow.sample_data import write_sample_dataset

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


EXAMPLE_CONFIG = Path(__file__).resolve().parents[1] / "examples" / "synthetic.yaml"


def test_validate_config_accepts_example_configuration() -> None:
    result = runner.invoke(app, ["validate-config", str(EXAMPLE_CONFIG)])
    assert result.exit_code == 0, result.output
    assert result.output.startswith("valid:")
    assert "3 images" in result.output


def test_validate_config_reports_invalid_configuration(tmp_path: Path) -> None:
    config_file = tmp_path / "bad.yaml"
    config_file.write_text("dataset:\n  images: []\n", encoding="utf-8")
    result = runner.invoke(app, ["validate-config", str(config_file)])
    assert result.exit_code == 1
    assert "invalid:" in result.output
    assert "dataset.images" in result.output


def test_validate_config_check_images_reports_per_image_metadata(tmp_path: Path) -> None:
    write_sample_dataset(tmp_path / "data", seed=0, count=1)
    config_file = tmp_path / "run.yaml"
    config_file.write_text(
        "dataset:\n"
        "  images:\n"
        "    - name: synthetic_00\n"
        "      path: data/synthetic_00.tif\n"
        "      calibration: {pixel_size: 0.5, unit: um}\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["validate-config", str(config_file), "--check-images"])
    assert result.exit_code == 0, result.output
    assert "synthetic_00: axes=YX shape=(256, 256) dtype=uint16" in result.output


def test_validate_config_check_images_fails_for_missing_image(tmp_path: Path) -> None:
    config_file = tmp_path / "run.yaml"
    config_file.write_text(
        "dataset:\n"
        "  images:\n"
        "    - name: absent\n"
        "      path: absent.tif\n"
        "      calibration: {pixel_size: 0.5, unit: um}\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["validate-config", str(config_file), "--check-images"])
    assert result.exit_code == 1
    assert "absent: cannot read image" in result.output


@pytest.mark.parametrize("count", ["0", "-1"])
def test_make_sample_data_rejects_non_positive_count(tmp_path: Path, count: str) -> None:
    result = runner.invoke(app, ["make-sample-data", "--out-dir", str(tmp_path), "--count", count])
    assert result.exit_code != 0

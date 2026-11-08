import csv
import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner, Result

from microflow.cli import app
from microflow.image_io import write_image

runner = CliRunner()


@pytest.fixture
def measure_inputs(tmp_path: Path) -> tuple[Path, Path]:
    labels = np.zeros((40, 40), dtype=np.int32)
    labels[5:15, 3:13] = 1
    labels[30, 30:33] = 2
    labels_path = tmp_path / "labels.tif"
    raw_path = tmp_path / "raw.tif"
    write_image(labels_path, labels)
    write_image(raw_path, np.full((40, 40), 200, dtype=np.uint16))
    return labels_path, raw_path


def _measure(tmp_path: Path, args: list[str]) -> tuple[Result, Path]:
    output = tmp_path / "measurements.csv"
    result = runner.invoke(app, ["measure", *args, "--output", str(output)])
    return result, output


def test_measure_writes_one_row_per_object_in_calibrated_units(
    tmp_path: Path, measure_inputs: tuple[Path, Path]
) -> None:
    labels_path, raw_path = measure_inputs
    result, output = _measure(
        tmp_path,
        [str(labels_path), "--raw", str(raw_path), "--pixel-size", "0.5", "--unit", "um"],
    )
    assert result.exit_code == 0, result.output
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["label"] for row in rows] == ["1", "2"]
    assert float(rows[0]["area_um2"]) == pytest.approx(25.0)
    assert float(rows[1]["area_um2"]) == pytest.approx(0.75)
    assert float(rows[0]["raw_c0_mean"]) == pytest.approx(200.0)
    assert float(rows[0]["raw_c0_integrated_intensity_um2"]) == pytest.approx(200 * 25)


def test_corrected_image_adds_its_own_columns(
    tmp_path: Path, measure_inputs: tuple[Path, Path]
) -> None:
    labels_path, raw_path = measure_inputs
    corrected_path = tmp_path / "corrected.tif"
    write_image(corrected_path, np.full((40, 40), 100, dtype=np.uint16))
    corrected_path.with_suffix(".provenance.json").write_text(
        json.dumps({"stages": [{"stage": "input"}, {"stage": "flat_field"}]}), encoding="utf-8"
    )
    result, output = _measure(
        tmp_path,
        [
            str(labels_path),
            "--raw",
            str(raw_path),
            "--corrected",
            str(corrected_path),
            "--pixel-size",
            "0.5",
            "--unit",
            "um",
        ],
    )
    assert result.exit_code == 0, result.output
    with output.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert float(row["raw_c0_mean"]) == pytest.approx(200.0)
    assert float(row["corrected_c0_mean"]) == pytest.approx(100.0)


def test_normalised_corrected_image_is_refused(
    tmp_path: Path, measure_inputs: tuple[Path, Path]
) -> None:
    labels_path, raw_path = measure_inputs
    corrected_path = tmp_path / "normalised.tif"
    write_image(corrected_path, np.full((40, 40), 100, dtype=np.uint16))
    corrected_path.with_suffix(".provenance.json").write_text(
        json.dumps({"stages": [{"stage": "normalize"}]}), encoding="utf-8"
    )
    result, output = _measure(
        tmp_path,
        [str(labels_path), "--raw", str(raw_path), "--corrected", str(corrected_path)],
    )
    assert result.exit_code == 1
    assert "normalise" in result.output
    assert not output.exists()


def test_corrected_image_without_sidecar_is_reported_as_unverified(
    tmp_path: Path, measure_inputs: tuple[Path, Path]
) -> None:
    labels_path, raw_path = measure_inputs
    corrected_path = tmp_path / "unknown.tif"
    write_image(corrected_path, np.full((40, 40), 100, dtype=np.uint16))
    result, output = _measure(
        tmp_path,
        [str(labels_path), "--raw", str(raw_path), "--corrected", str(corrected_path)],
    )
    assert result.exit_code == 0, result.output
    assert "cannot be verified" in result.output
    assert output.exists()


def test_uncalibrated_measurement_reports_pixel_units(
    tmp_path: Path, measure_inputs: tuple[Path, Path]
) -> None:
    labels_path, raw_path = measure_inputs
    result, output = _measure(tmp_path, [str(labels_path), "--raw", str(raw_path)])
    assert result.exit_code == 0, result.output
    header = output.read_text(encoding="utf-8").splitlines()[0]
    assert "area_px2" in header
    assert "raw_c0_integrated_intensity_px2" in header


def test_require_calibration_refuses_uncalibrated_input(
    tmp_path: Path, measure_inputs: tuple[Path, Path]
) -> None:
    labels_path, raw_path = measure_inputs
    result, output = _measure(
        tmp_path, [str(labels_path), "--raw", str(raw_path), "--require-calibration"]
    )
    assert result.exit_code == 1
    assert "no pixel calibration" in result.output
    assert not output.exists()


def test_labels_must_be_a_single_plane(tmp_path: Path, measure_inputs: tuple[Path, Path]) -> None:
    _, raw_path = measure_inputs
    stack_path = tmp_path / "stack.tif"
    write_image(stack_path, np.zeros((2, 40, 40), dtype=np.int32), axes="ZYX")
    result, _ = _measure(tmp_path, [str(stack_path), "--raw", str(raw_path)])
    assert result.exit_code == 1
    assert "expected YX" in result.output

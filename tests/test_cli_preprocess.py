import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
from typer.testing import CliRunner

from microflow.cli import app
from microflow.image_io import write_image

runner = CliRunner()

STEPS = (
    "steps:\n"
    "  - kind: subtract_constant\n"
    "    value: 10\n"
    "  - kind: gaussian\n"
    "    sigma: {value: 1, unit: um}\n"
)


@pytest.fixture
def input_tif(tmp_path: Path) -> Path:
    path = tmp_path / "input.tif"
    write_image(path, (np.arange(64, dtype=np.uint16).reshape(8, 8) + 20).astype(np.uint16))
    return path


def test_preprocess_writes_float64_output_and_provenance(tmp_path: Path, input_tif: Path) -> None:
    steps = tmp_path / "steps.yaml"
    steps.write_text(STEPS, encoding="utf-8")
    output = tmp_path / "out" / "result.tif"
    output.parent.mkdir()
    result = runner.invoke(
        app,
        [
            "preprocess",
            str(input_tif),
            "--output",
            str(output),
            "--steps",
            str(steps),
            "--pixel-size",
            "0.5",
            "--unit",
            "um",
        ],
    )
    assert result.exit_code == 0, result.output
    written = tifffile.imread(output)
    assert written.dtype == np.float64
    assert written.shape == (8, 8)

    provenance = json.loads(output.with_suffix(".provenance.json").read_text(encoding="utf-8"))
    assert provenance["source_dtype"] == "uint16"
    assert provenance["output_dtype"] == "float64"
    assert provenance["calibration"] == {"pixel_size": 0.5, "unit": "um"}
    stages = [stage["stage"] for stage in provenance["stages"]]
    assert stages == ["input", "subtract_constant", "gaussian", "output"]
    assert provenance["stages"][2]["effect"] == {"sigma_px": 2.0}


def test_preprocess_writes_before_after_preview(tmp_path: Path, input_tif: Path) -> None:
    output = tmp_path / "result.tif"
    preview = tmp_path / "preview.png"
    result = runner.invoke(
        app, ["preprocess", str(input_tif), "--output", str(output), "--preview", str(preview)]
    )
    assert result.exit_code == 0, result.output
    assert preview.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_preprocess_without_calibration_fails_for_length_steps(
    tmp_path: Path, input_tif: Path
) -> None:
    steps = tmp_path / "steps.yaml"
    steps.write_text(STEPS, encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "preprocess",
            str(input_tif),
            "--output",
            str(tmp_path / "out.tif"),
            "--steps",
            str(steps),
        ],
    )
    assert result.exit_code == 1
    assert "needs pixel calibration" in result.output
    assert not (tmp_path / "out.tif").exists()


def test_pixel_size_requires_unit(tmp_path: Path, input_tif: Path) -> None:
    result = runner.invoke(
        app,
        ["preprocess", str(input_tif), "--output", str(tmp_path / "o.tif"), "--pixel-size", "0.5"],
    )
    assert result.exit_code == 1
    assert "needs --unit" in result.output


def test_invalid_unit_is_reported(tmp_path: Path, input_tif: Path) -> None:
    result = runner.invoke(
        app,
        [
            "preprocess",
            str(input_tif),
            "--output",
            str(tmp_path / "o.tif"),
            "--pixel-size",
            "0.5",
            "--unit",
            "furlong",
        ],
    )
    assert result.exit_code == 1
    assert "invalid calibration" in result.output


def test_invalid_steps_file_is_reported(tmp_path: Path, input_tif: Path) -> None:
    steps = tmp_path / "steps.yaml"
    steps.write_text("steps:\n  - kind: sharpen\n", encoding="utf-8")
    result = runner.invoke(
        app,
        ["preprocess", str(input_tif), "--output", str(tmp_path / "o.tif"), "--steps", str(steps)],
    )
    assert result.exit_code == 1
    assert "Input tag 'sharpen'" in result.output

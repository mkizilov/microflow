import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
from click.testing import Result
from typer.testing import CliRunner

from microflow.cli import app
from microflow.image_io import write_image
from microflow.sample_data import Disc, make_phantom

runner = CliRunner()

DISCS = [Disc(centre=(30, 30), radius=8), Disc(centre=(30, 95), radius=6)]


@pytest.fixture
def phantom_files(tmp_path: Path) -> tuple[Path, Path]:
    image, truth = make_phantom(DISCS, seed=4)
    image_path = tmp_path / "phantom.tif"
    truth_path = tmp_path / "truth.tif"
    write_image(image_path, image.astype(np.uint16))
    write_image(truth_path, truth)
    return image_path, truth_path


def _segment(tmp_path: Path, args: list[str]) -> tuple[Result, Path]:
    output = tmp_path / "labels.tif"
    result = runner.invoke(app, ["segment", *args, "--output", str(output)])
    return result, output


def test_segment_writes_int32_labels_and_sidecar(
    tmp_path: Path, phantom_files: tuple[Path, Path]
) -> None:
    image_path, _ = phantom_files
    result, output = _segment(tmp_path, [str(image_path), "--pixel-size", "1", "--unit", "um"])
    assert result.exit_code == 0, result.output
    labels = tifffile.imread(output)
    assert labels.dtype == np.int32
    assert labels.shape == (128, 128)
    assert labels.max() == 2
    record = json.loads(output.with_suffix(".segmentation.json").read_text(encoding="utf-8"))
    assert record["object_counts"] == [2]
    assert record["scores"] is None
    assert record["settings"]["method"] == "otsu"


def test_segment_reports_dice_and_iou_against_reference(
    tmp_path: Path, phantom_files: tuple[Path, Path]
) -> None:
    image_path, truth_path = phantom_files
    result, output = _segment(
        tmp_path,
        [str(image_path), "--reference", str(truth_path), "--pixel-size", "1", "--unit", "um"],
    )
    assert result.exit_code == 0, result.output
    assert "Dice" in result.output
    record = json.loads(output.with_suffix(".segmentation.json").read_text(encoding="utf-8"))
    assert record["scores"]["dice"] > 0.99
    assert record["scores"]["iou"] > 0.98


def test_segment_without_calibration_is_an_error(
    tmp_path: Path, phantom_files: tuple[Path, Path]
) -> None:
    result, output = _segment(tmp_path, [str(phantom_files[0])])
    assert result.exit_code == 1
    assert "needs pixel calibration" in result.output
    assert not output.exists()


def test_segment_rejects_invalid_config_file(
    tmp_path: Path, phantom_files: tuple[Path, Path]
) -> None:
    config = tmp_path / "seg.yaml"
    config.write_text("method: kmeans\n", encoding="utf-8")
    result, _ = _segment(
        tmp_path,
        [str(phantom_files[0]), "--config", str(config), "--pixel-size", "1", "--unit", "um"],
    )
    assert result.exit_code == 1
    assert "method" in result.output


def test_segment_reports_reference_shape_mismatch(
    tmp_path: Path, phantom_files: tuple[Path, Path]
) -> None:
    wrong = tmp_path / "wrong.tif"
    write_image(wrong, np.zeros((10, 10), dtype=np.int32))
    result, _ = _segment(
        tmp_path,
        [str(phantom_files[0]), "--reference", str(wrong), "--pixel-size", "1", "--unit", "um"],
    )
    assert result.exit_code == 1
    assert "shapes differ" in result.output

import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from microflow.cli import app
from microflow.image_io import write_image
from microflow.loading import load_image
from microflow.ml.cli import _single_channel
from microflow.ml.config import TrainingConfig
from microflow.ml.manifest import read_manifest
from microflow.provenance.hashing import sha256_file

runner = CliRunner()
CALIBRATION = ["--pixel-size", "1.0", "--unit", "um"]


def _labels(*objects: tuple[int, int, int, int, int]) -> np.ndarray:
    labels = np.zeros((10, 10), dtype=np.int32)
    for label, top, bottom, left, right in objects:
        labels[top:bottom, left:right] = label
    return labels


def test_fingerprint_prints_the_checksum_of_a_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"weights")
    result = runner.invoke(app, ["ml", "fingerprint", str(checkpoint)])
    assert result.exit_code == 0, result.output
    assert hashlib.sha256(b"weights").hexdigest() in result.output


def test_fingerprint_of_a_missing_file_fails(tmp_path: Path) -> None:
    result = runner.invoke(app, ["ml", "fingerprint", str(tmp_path / "absent.pt")])
    assert result.exit_code == 1
    assert "no such file" in result.output


def test_compare_reports_agreement_and_writes_json(tmp_path: Path) -> None:
    labels = _labels((1, 0, 2, 0, 2), (2, 5, 8, 5, 8))
    write_image(tmp_path / "ml.tif", labels, "YX")
    write_image(tmp_path / "classical.tif", labels, "YX")
    report = tmp_path / "comparison.json"
    result = runner.invoke(
        app,
        [
            "ml",
            "compare",
            str(tmp_path / "ml.tif"),
            str(tmp_path / "classical.tif"),
            "--output",
            str(report),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Dice 1.0000  IoU 1.0000" in result.output
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["matched_objects"] == payload["objects_a"] == payload["objects_b"] == 2


def test_compare_reports_disagreement_by_score(tmp_path: Path) -> None:
    write_image(tmp_path / "ml.tif", _labels((1, 0, 4, 0, 4)), "YX")
    write_image(tmp_path / "classical.tif", _labels((1, 2, 6, 2, 6)), "YX")
    result = runner.invoke(
        app, ["ml", "compare", str(tmp_path / "ml.tif"), str(tmp_path / "classical.tif")]
    )
    assert result.exit_code == 0, result.output
    assert "Dice 0.2500" in result.output


def test_compare_refuses_label_images_of_different_shapes(tmp_path: Path) -> None:
    write_image(tmp_path / "ml.tif", np.zeros((10, 10), np.int32), "YX")
    write_image(tmp_path / "classical.tif", np.zeros((8, 8), np.int32), "YX")
    result = runner.invoke(
        app, ["ml", "compare", str(tmp_path / "ml.tif"), str(tmp_path / "classical.tif")]
    )
    assert result.exit_code == 1
    assert "differ" in result.output


def test_segment_without_pytorch_says_how_to_install_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "image.tif"
    write_image(image, np.ones((8, 8), np.uint16))
    monkeypatch.setitem(sys.modules, "torch", None)
    result = runner.invoke(
        app,
        [
            "ml",
            "segment",
            str(tmp_path / "absent.pt"),
            str(image),
            "--output",
            str(tmp_path / "o.tif"),
        ],
    )
    assert result.exit_code == 1
    assert "PyTorch is required" in result.output


def test_single_channel_selection_is_explicit() -> None:
    stack = np.zeros((2, 4, 4))
    with pytest.raises(ValueError, match="choose one with --channel"):
        _single_channel(stack, "CYX", None)
    picked, axes = _single_channel(stack, "CYX", 1)
    assert picked.shape == (4, 4)
    assert axes == "YX"
    with pytest.raises(ValueError, match="no channel axis"):
        _single_channel(np.zeros((4, 4)), "YX", 1)
    _, plain = _single_channel(np.zeros((4, 4)), "YX", None)
    assert plain == "YX"


@pytest.mark.ml
def test_ml_segment_writes_the_classical_contract_and_a_report(
    tmp_path: Path, ml_manifest: Path, ml_config: Callable[..., TrainingConfig]
) -> None:
    pytest.importorskip("torch")
    from microflow.ml.train import train

    trained = train(ml_config(ml_manifest, training={"epochs": 1, "batch_size": 4}))
    checkpoint = trained.output_directory / "best.pt"
    record = read_manifest(ml_manifest)[0]

    ml_dir = tmp_path / "ml"
    ml_dir.mkdir()
    ml_out = ml_dir / "labels.tif"
    ml_run = runner.invoke(
        app,
        [
            "ml",
            "segment",
            str(checkpoint),
            str(record.image),
            "--output",
            str(ml_out),
            "--tile-size",
            "32",
            "--overlap",
            "8",
            *CALIBRATION,
            "--reference",
            str(record.mask),
        ],
    )
    assert ml_run.exit_code == 0, ml_run.output

    classical_dir = tmp_path / "classical"
    classical_dir.mkdir()
    classical_out = classical_dir / "labels.tif"
    classical_run = runner.invoke(
        app,
        [
            "segment",
            str(record.image),
            "--output",
            str(classical_out),
            *CALIBRATION,
            "--reference",
            str(record.mask),
        ],
    )
    assert classical_run.exit_code == 0, classical_run.output

    ml_sidecar = json.loads(ml_out.with_suffix(".segmentation.json").read_text(encoding="utf-8"))
    classical_sidecar = json.loads(
        classical_out.with_suffix(".segmentation.json").read_text(encoding="utf-8")
    )
    contract = {"source", "axes", "object_counts", "thresholds", "settings", "scores"}
    assert set(ml_sidecar) == set(classical_sidecar) == contract
    assert ml_sidecar["axes"] == classical_sidecar["axes"] == "YX"
    assert ml_sidecar["scores"]["dice"] >= 0.0

    labels = load_image(ml_out).data
    assert labels.dtype == np.int32
    assert labels.shape == load_image(record.image).data.shape

    report = json.loads(ml_out.with_suffix(".ml.json").read_text(encoding="utf-8"))
    assert report["checkpoint"]["sha256"] == sha256_file(checkpoint)
    assert report["tiling"]["overlap"] == 8
    assert report["tiling"]["overlap_sufficient"] is False
    assert report["device"]["used"] == "cpu"
    assert report["device"]["fallback_used"] is False
    assert len(report["planes"]) == 1


@pytest.mark.ml
def test_inspect_validates_and_describes_a_checkpoint(
    tmp_path: Path, ml_manifest: Path, ml_config: Callable[..., TrainingConfig]
) -> None:
    pytest.importorskip("torch")
    from microflow.ml.train import train

    trained = train(ml_config(ml_manifest, training={"epochs": 1, "batch_size": 4}))
    result = runner.invoke(app, ["ml", "inspect", str(trained.output_directory / "best.pt")])
    assert result.exit_code == 0, result.output
    assert "compatible with this MicroFlow version" in result.output
    assert trained.training_id in result.output

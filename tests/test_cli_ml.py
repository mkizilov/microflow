import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from microflow.cli import app
from microflow.ml.synthetic import make_synthetic_dataset

runner = CliRunner()


def _write_config(root: Path, manifest: Path) -> Path:
    path = root / "train.yaml"
    path.write_text(
        f"manifest: {manifest}\n"
        f"output_directory: {root / 'out'}\n"
        "model: {base_channels: 4, depth: 2}\n"
        "patches: {size: 32, per_image: 2}\n"
        "training: {epochs: 1, batch_size: 4}\n",
        encoding="utf-8",
    )
    return path


def test_make_dataset_and_splits_run_without_pytorch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    data = tmp_path / "data"
    made = runner.invoke(app, ["ml", "make-dataset", str(data), "--samples", "4", "--size", "32"])
    assert made.exit_code == 0, made.output
    assert (data / "manifest.yaml").is_file()

    config = _write_config(tmp_path, data / "manifest.yaml")
    shown = runner.invoke(app, ["ml", "splits", str(config)])
    assert shown.exit_code == 0, shown.output
    assert "4 sample(s), 8 image(s)" in shown.output
    for name in ("train", "validation", "test"):
        assert name in shown.output


def test_training_without_pytorch_gives_an_install_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = make_synthetic_dataset(tmp_path / "data", samples=3, size=32)
    config = _write_config(tmp_path, manifest)
    monkeypatch.setitem(sys.modules, "torch", None)
    result = runner.invoke(app, ["ml", "train", str(config)])
    assert result.exit_code == 1
    assert "PyTorch is required" in result.output
    assert "uv sync --extra ml" in result.output


def test_an_unknown_device_is_refused_before_anything_else(tmp_path: Path) -> None:
    config = _write_config(tmp_path, tmp_path / "missing-manifest.yaml")
    result = runner.invoke(app, ["ml", "train", str(config), "--device", "tpu"])
    assert result.exit_code == 1
    assert "--device must be one of cpu, cuda, auto" in result.output


@pytest.mark.ml
def test_train_and_evaluate_from_the_command_line(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    manifest = make_synthetic_dataset(tmp_path / "data", samples=4, images_per_sample=2, size=32)
    config = _write_config(tmp_path, manifest)

    trained = runner.invoke(app, ["ml", "train", str(config)])
    assert trained.exit_code == 0, trained.output
    assert "best epoch" in trained.output

    best = tmp_path / "out" / "best.pt"
    evaluated = runner.invoke(app, ["ml", "evaluate", str(config), str(best)])
    assert evaluated.exit_code == 0, evaluated.output
    assert "test:" in evaluated.output

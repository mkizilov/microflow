import os
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from microflow.batch.store import LOCK_FILE
from microflow.cli import app
from microflow.image_io import write_image
from microflow.sample_data import Disc, make_phantom

runner = CliRunner()

DISCS = [Disc(centre=(30, 30), radius=8), Disc(centre=(30, 95), radius=6)]


def _phantom(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image, _ = make_phantom(DISCS, seed=seed)
    write_image(path, image.astype(np.uint16))


def _batch_yaml(tmp_path: Path, images: list[str], extra: str = "") -> Path:
    lines = ["name: cli-batch", "dataset:", "  images:"]
    for name in images:
        lines += [
            f"    - name: {name}",
            f"      path: data/{name}.tif",
            "      calibration: {pixel_size: 1.0, unit: um}",
        ]
    lines += [
        "segmentation:",
        "  min_object_area: {value: 9, unit: um}",
        "output_directory: out",
        "workers: 1",
    ]
    path = tmp_path / "batch.yaml"
    path.write_text("\n".join(lines) + "\n" + extra, encoding="utf-8")
    return path


def test_batch_command_runs_then_resumes_from_cache(tmp_path: Path) -> None:
    _phantom(tmp_path / "data" / "a.tif", seed=1)
    _phantom(tmp_path / "data" / "b.tif", seed=2)
    config = _batch_yaml(tmp_path, ["a", "b"])

    first = runner.invoke(app, ["batch", str(config)])
    assert first.exit_code == 0, first.output
    assert "[1/2] a: ok" in first.output
    assert "[2/2] b: ok" in first.output
    assert "status: complete" in first.output

    second = runner.invoke(app, ["batch", str(config)])
    assert second.exit_code == 0, second.output
    assert "a: ok (cached)" in second.output
    assert "resumed 2" in second.output


def test_batch_command_exits_with_three_when_an_image_fails(tmp_path: Path) -> None:
    _phantom(tmp_path / "data" / "a.tif", seed=1)
    config = _batch_yaml(tmp_path, ["a", "missing"])
    result = runner.invoke(app, ["batch", str(config)])
    assert result.exit_code == 3, result.output
    assert "missing: failed" in result.output
    assert "status: completed_with_failures" in result.output


def test_batch_command_rejects_an_invalid_configuration(tmp_path: Path) -> None:
    config = _batch_yaml(tmp_path, ["a"], extra="segmentation_method: kmeans\n")
    result = runner.invoke(app, ["batch", str(config)])
    assert result.exit_code == 1
    assert "error:" in result.output
    assert "segmentation_method" in result.output


def test_batch_command_refuses_a_directory_held_by_a_live_run(tmp_path: Path) -> None:
    _phantom(tmp_path / "data" / "a.tif", seed=1)
    config = _batch_yaml(tmp_path, ["a"])
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / LOCK_FILE).write_text(str(os.getpid()), encoding="utf-8")
    result = runner.invoke(app, ["batch", str(config)])
    assert result.exit_code == 1
    assert "another batch run" in result.output

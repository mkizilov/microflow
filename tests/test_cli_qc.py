from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner, Result

from microflow.cli import app
from microflow.image_io import write_image
from microflow.reporting.figures import FIGURE_TITLES
from microflow.sample_data import Disc, make_phantom

runner = CliRunner()

DISCS = [Disc(centre=(30, 30), radius=8), Disc(centre=(30, 95), radius=6)]


@pytest.fixture
def phantom_tif(tmp_path: Path) -> Path:
    image, _ = make_phantom(DISCS, seed=4)
    path = tmp_path / "raw.tif"
    write_image(path, image.astype(np.uint16))
    return path


def _segment(tmp_path: Path, raw: Path) -> Path:
    labels = tmp_path / "labels.tif"
    result = runner.invoke(
        app,
        ["segment", str(raw), "--output", str(labels), "--pixel-size", "1", "--unit", "um"],
    )
    assert result.exit_code == 0, result.output
    return labels


def _qc(args: list[str]) -> Result:
    return runner.invoke(app, ["qc-report", *args])


def test_qc_report_writes_html_and_png_figures(tmp_path: Path, phantom_tif: Path) -> None:
    labels = _segment(tmp_path, phantom_tif)
    output = tmp_path / "report.html"
    png_dir = tmp_path / "figures"
    result = _qc(
        [
            str(phantom_tif),
            "--labels",
            str(labels),
            "--output",
            str(output),
            "--png-dir",
            str(png_dir),
        ]
    )
    assert result.exit_code == 0, result.output
    page = output.read_text(encoding="utf-8")
    assert "Quality control" in page
    assert "No suspicious results were flagged." in page
    written = sorted(path.name for path in png_dir.iterdir())
    assert written == sorted(f"{name}.png" for name in FIGURE_TITLES if name != "preprocessed")


def test_qc_report_accepts_a_preprocessed_image(tmp_path: Path, phantom_tif: Path) -> None:
    labels = _segment(tmp_path, phantom_tif)
    preprocessed = tmp_path / "pre.tif"
    write_image(preprocessed, np.ones((128, 128)), "YX")
    output = tmp_path / "report.html"
    result = _qc(
        [
            str(phantom_tif),
            "--labels",
            str(labels),
            "--preprocessed",
            str(preprocessed),
            "--output",
            str(output),
        ]
    )
    assert result.exit_code == 0, result.output
    assert output.read_text(encoding="utf-8").count("data:image/png;base64,") == 6


def test_fail_on_flags_exits_with_status_two_and_still_writes_the_report(tmp_path: Path) -> None:
    raw = tmp_path / "raw.tif"
    write_image(raw, np.full((32, 32), 100, dtype=np.uint16))
    labels = tmp_path / "labels.tif"
    write_image(labels, np.zeros((32, 32), dtype=np.int32))
    output = tmp_path / "report.html"
    result = _qc([str(raw), "--labels", str(labels), "--output", str(output), "--fail-on-flags"])
    assert result.exit_code == 2
    assert "flag empty_mask" in result.output
    assert output.exists()


def test_qc_config_thresholds_are_applied(tmp_path: Path, phantom_tif: Path) -> None:
    labels = _segment(tmp_path, phantom_tif)
    config = tmp_path / "qc.yaml"
    config.write_text("max_foreground_fraction: 0.01\n", encoding="utf-8")
    output = tmp_path / "report.html"
    result = _qc(
        [
            str(phantom_tif),
            "--labels",
            str(labels),
            "--output",
            str(output),
            "--qc-config",
            str(config),
        ]
    )
    assert result.exit_code == 0, result.output
    assert "excessive_coverage" in result.output


def test_invalid_qc_config_is_reported(tmp_path: Path, phantom_tif: Path) -> None:
    labels = _segment(tmp_path, phantom_tif)
    config = tmp_path / "qc.yaml"
    config.write_text("max_foreground_fraction: 1.5\n", encoding="utf-8")
    output = tmp_path / "report.html"
    result = _qc(
        [
            str(phantom_tif),
            "--labels",
            str(labels),
            "--output",
            str(output),
            "--qc-config",
            str(config),
        ]
    )
    assert result.exit_code == 1
    assert "max_foreground_fraction" in result.output
    assert not output.exists()


def test_labels_on_a_different_grid_are_refused(tmp_path: Path, phantom_tif: Path) -> None:
    labels = tmp_path / "small.tif"
    write_image(labels, np.zeros((16, 16), dtype=np.int32))
    result = _qc([str(phantom_tif), "--labels", str(labels), "--output", str(tmp_path / "r.html")])
    assert result.exit_code == 1
    assert "share" in result.output

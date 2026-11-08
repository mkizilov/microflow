import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from microflow.cli import app
from microflow.image_io import write_image
from microflow.sample_data import Disc, make_phantom

runner = CliRunner()

DISCS = [Disc(centre=(30, 30), radius=8), Disc(centre=(30, 95), radius=6)]


def _phantom(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image, _ = make_phantom(DISCS, seed=seed)
    write_image(path, image.astype(np.uint16))


def _write_config(root: Path, min_area: int) -> Path:
    lines = ["name: cli-provenance", "dataset:", "  images:"]
    for name in ("a", "b"):
        lines += [
            f"    - name: {name}",
            f"      path: data/{name}.tif",
            "      calibration: {pixel_size: 1.0, unit: um}",
        ]
    lines += [
        "segmentation:",
        f"  min_object_area: {{value: {min_area}, unit: um}}",
        "output_directory: out",
        "workers: 1",
    ]
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"batch-{min_area}.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _prepare(tmp_path: Path) -> None:
    _phantom(tmp_path / "data" / "a.tif", seed=1)
    _phantom(tmp_path / "data" / "b.tif", seed=2)


def test_export_command_writes_csv_parquet_and_the_dictionary(tmp_path: Path) -> None:
    _prepare(tmp_path)
    config = _write_config(tmp_path, 9)
    assert runner.invoke(app, ["batch", str(config)]).exit_code == 0

    result = runner.invoke(
        app, ["export", str(tmp_path / "out"), "--output", str(tmp_path / "exp")]
    )
    assert result.exit_code == 0, result.output
    assert "exported" in result.output
    expected = ("measurements.csv", "measurements.parquet", "data_dictionary.csv")
    for name in (*expected, "provenance.json"):
        assert (tmp_path / "exp" / name).exists(), name


def test_export_command_refuses_a_directory_holding_another_result_set(tmp_path: Path) -> None:
    _prepare(tmp_path)
    config = _write_config(tmp_path, 9)
    assert runner.invoke(app, ["batch", str(config)]).exit_code == 0

    target = tmp_path / "exp"
    target.mkdir()
    record = json.dumps({"result_set_id": "other"})
    (target / "provenance.json").write_text(record, encoding="utf-8")
    (target / "measurements.csv").write_text("other\n", encoding="utf-8")
    result = runner.invoke(
        app,
        ["export", str(tmp_path / "out"), "--output", str(target), "--format", "csv"],
    )
    assert result.exit_code == 1
    assert "different analysis or data set" in result.output
    assert (target / "measurements.csv").read_text(encoding="utf-8") == "other\n"


def test_batch_command_refuses_a_changed_configuration_until_overwrite(tmp_path: Path) -> None:
    _prepare(tmp_path)
    assert runner.invoke(app, ["batch", str(_write_config(tmp_path, 9))]).exit_code == 0

    changed = _write_config(tmp_path, 4)
    refused = runner.invoke(app, ["batch", str(changed)])
    assert refused.exit_code == 1
    assert "different analysis" in refused.output

    accepted = runner.invoke(app, ["batch", str(changed), "--overwrite"])
    assert accepted.exit_code == 0, accepted.output
    assert "status: complete" in accepted.output


def test_compare_configs_reports_whether_the_analysis_changed(tmp_path: Path) -> None:
    nine = str(_write_config(tmp_path, 9))
    same = runner.invoke(app, ["compare-configs", nine, nine])
    assert same.exit_code == 0, same.output
    assert "analysis identity: same" in same.output
    assert "no differences" in same.output

    four = str(_write_config(tmp_path / "x", 4))
    different = runner.invoke(app, ["compare-configs", nine, four])
    assert different.exit_code == 0, different.output
    assert "analysis identity: different" in different.output
    assert "segmentation.min_object_area.value: 9.0 -> 4.0  [changes analysis]" in different.output


def test_cache_status_command_names_each_image(tmp_path: Path) -> None:
    _prepare(tmp_path)
    config = _write_config(tmp_path, 9)
    assert runner.invoke(app, ["batch", str(config)]).exit_code == 0

    result = runner.invoke(app, ["cache-status", str(config)])
    assert result.exit_code == 0, result.output
    assert "output directory matches" in result.output
    assert "a: reusable (valid)" in result.output
    assert "b: reusable (valid)" in result.output

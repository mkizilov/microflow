import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from microflow.batch import BatchConfig, analysis_id, run_batch
from microflow.ml.config import TrainingConfig
from microflow.ml.manifest import read_manifest
from microflow.plugins.api import ENTRY_POINT_GROUPS, PluginError, PluginSpec
from microflow.plugins.registry import configure, discover
from microflow.provenance.hashing import sha256_file

PLUGIN = "ml-unet"
CALIBRATION = {"pixel_size": 1.0, "unit": "um"}


def _options(checkpoint: Path, digest: str, **extra: object) -> dict[str, object]:
    return {"checkpoint": str(checkpoint), "checkpoint_sha256": digest, **extra}


def _dataset_images(manifest: Path) -> list[dict[str, Any]]:
    return [
        {"name": record.image.stem, "path": record.image, "calibration": CALIBRATION}
        for record in read_manifest(manifest)
    ]


def _segmentation(checkpoint: Path, digest: str, **options: object) -> dict[str, Any]:
    return {
        "method": "plugin",
        "plugin": {"name": PLUGIN, "options": _options(checkpoint, digest, **options)},
        "min_object_area": {"value": 4, "unit": "um"},
    }


def test_the_trained_model_segmenter_is_registered_by_the_package() -> None:
    names = {item.name: item for item in discover("segmenter")}
    assert PLUGIN in names
    assert names[PLUGIN].distribution == "microflow"
    assert names[PLUGIN].target == "microflow.ml.plugin:UNetSegmenter"
    assert ENTRY_POINT_GROUPS["segmenter"] == "microflow.segmenters"


def test_the_checkpoint_digest_is_verified_before_the_file_is_opened(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"not a checkpoint")
    configured = configure(
        "segmenter",
        PluginSpec(name=PLUGIN, options=_options(checkpoint, "0" * 64)),
    )
    with pytest.raises(PluginError, match="has SHA-256"):
        configured.plugin.segment(np.zeros((8, 8)), None, configured.options)


def test_a_missing_checkpoint_is_reported_by_name(tmp_path: Path) -> None:
    configured = configure(
        "segmenter",
        PluginSpec(name=PLUGIN, options=_options(tmp_path / "absent.pt", "0" * 64)),
    )
    with pytest.raises(PluginError, match="does not exist"):
        configured.plugin.segment(np.zeros((8, 8)), None, configured.options)


def test_the_digest_must_be_a_lower_case_sha256_value(tmp_path: Path) -> None:
    with pytest.raises(PluginError, match="invalid options"):
        configure("segmenter", PluginSpec(name=PLUGIN, options=_options(tmp_path / "m.pt", "ABC")))


def test_without_pytorch_the_plugin_says_how_to_install_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"model")
    digest = sha256_file(checkpoint)
    monkeypatch.setitem(sys.modules, "torch", None)
    configured = configure(
        "segmenter", PluginSpec(name=PLUGIN, options=_options(checkpoint, digest))
    )
    with pytest.raises(PluginError, match="PyTorch is required"):
        configured.plugin.segment(np.zeros((8, 8)), None, configured.options)


def test_the_checkpoint_digest_is_part_of_the_analysis_identity(tmp_path: Path) -> None:
    images = [{"name": "img", "path": tmp_path / "img.tif", "calibration": CALIBRATION}]

    def config_with(digest: str) -> BatchConfig:
        return BatchConfig.model_validate(
            {
                "dataset": {"images": images},
                "segmentation": _segmentation(tmp_path / "m.pt", digest),
                "output_directory": tmp_path / "out",
            }
        )

    first = analysis_id(config_with("a" * 64))
    assert analysis_id(config_with("a" * 64)) == first
    assert analysis_id(config_with("b" * 64)) != first


def test_a_batch_run_segments_with_the_trained_model_through_the_plugin(
    ml_manifest: Path, ml_config: Callable[..., TrainingConfig], tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    from microflow.ml.train import train

    trained = train(ml_config(ml_manifest, training={"epochs": 1, "batch_size": 4}))
    checkpoint = trained.output_directory / "best.pt"
    config = BatchConfig.model_validate(
        {
            "name": "ml-batch",
            "dataset": {"images": _dataset_images(ml_manifest)},
            "segmentation": _segmentation(
                checkpoint,
                sha256_file(checkpoint),
                device="cpu",
                tile_size=32,
                overlap=8,
            ),
            "output_directory": tmp_path / "out",
            "workers": 1,
        }
    )
    manifest = run_batch(config)
    assert manifest.status == "complete"
    assert manifest.provenance.algorithm_versions["plugin:segmenter:ml-unet"] == "1.0.0"

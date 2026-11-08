from pathlib import Path

import numpy as np
import pytest

from microflow.config import ConfigError, ImageRecord, MicroFlowConfig
from microflow.dataset import inspect_dataset, inspect_image
from microflow.image_io import write_image
from microflow.units import Calibration


def _record(path: Path, **fields: object) -> ImageRecord:
    return ImageRecord.model_validate({"name": "img", "path": path, **fields})


def test_heterogeneous_images_are_described_individually(tmp_path: Path) -> None:
    two_channel = tmp_path / "two.tif"
    write_image(two_channel, np.zeros((2, 8, 8), dtype=np.uint16))
    single_plane = tmp_path / "single.tif"
    write_image(single_plane, np.zeros((16, 12), dtype=np.float32))
    z_stack = tmp_path / "stack.tif"
    write_image(z_stack, np.zeros((5, 4, 6), dtype=np.uint8))

    config = MicroFlowConfig.model_validate(
        {
            "dataset": {
                "images": [
                    {
                        "name": "two",
                        "path": str(two_channel),
                        "axes": "CYX",
                        "channels": ["DAPI", "GFP"],
                        "calibration": {"pixel_size": 0.3, "unit": "um"},
                    },
                    {
                        "name": "single",
                        "path": str(single_plane),
                        "calibration": {"pixel_size": 0.3, "unit": "um"},
                    },
                    {
                        "name": "stack",
                        "path": str(z_stack),
                        "axes": "ZYX",
                        "calibration": {"pixel_size": 1.0, "unit": "mm"},
                    },
                ]
            }
        }
    )
    metadata = inspect_dataset(config)

    assert [m.shape for m in metadata] == [(2, 8, 8), (16, 12), (5, 4, 6)]
    assert [m.dtype for m in metadata] == ["uint16", "float32", "uint8"]
    assert metadata[0].channels == ("DAPI", "GFP")
    assert metadata[0].axis_size("C") == 2
    assert metadata[1].channels is None
    assert metadata[2].axis_size("Z") == 5
    assert metadata[2].calibration == Calibration(pixel_size=1.0, unit="mm")


def test_axes_must_match_image_dimensions(tmp_path: Path) -> None:
    path = tmp_path / "plane.tif"
    write_image(path, np.zeros((8, 8), dtype=np.uint8))
    with pytest.raises(ConfigError, match="describe 3 dimensions but"):
        inspect_image(_record(path, axes="CYX"))


def test_channel_names_must_match_channel_axis(tmp_path: Path) -> None:
    path = tmp_path / "two.tif"
    write_image(path, np.zeros((2, 8, 8), dtype=np.uint8))
    with pytest.raises(ConfigError, match="3 channel names given but the image has 2"):
        inspect_image(_record(path, axes="CYX", channels=["a", "b", "c"]))


def test_channels_without_channel_axis_must_be_single(tmp_path: Path) -> None:
    path = tmp_path / "plane.tif"
    write_image(path, np.zeros((8, 8), dtype=np.uint8))
    with pytest.raises(ConfigError, match="2 channel names given but the image has 1"):
        inspect_image(_record(path, channels=["a", "b"]))


def test_missing_image_file_is_reported_with_image_name(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="img: cannot read image"):
        inspect_image(_record(tmp_path / "absent.tif"))


def test_invalid_image_file_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "not-a-tiff.tif"
    path.write_bytes(b"this is not a tiff file")
    with pytest.raises(ConfigError, match="cannot read image"):
        inspect_image(_record(path))

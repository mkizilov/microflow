"""Typed per-image metadata, inspected from image headers and checked against the manifest.

Images in one dataset may differ in shape, dimensionality, channel count and dtype; each
image is described on its own.
"""

from pathlib import Path

import tifffile
from pydantic import BaseModel, ConfigDict

from microflow.config import ConfigError, ImageRecord, MicroFlowConfig
from microflow.image_io import read_shape_and_dtype
from microflow.units import Calibration


class ImageMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    path: Path
    axes: str
    shape: tuple[int, ...]
    dtype: str
    channels: tuple[str, ...] | None
    calibration: Calibration | None

    def axis_size(self, axis: str) -> int:
        return self.shape[self.axes.index(axis)]


def inspect_image(record: ImageRecord) -> ImageMetadata:
    try:
        shape, dtype = read_shape_and_dtype(record.path)
    except (OSError, tifffile.TiffFileError) as exc:
        raise ConfigError(f"{record.name}: cannot read image {record.path}: {exc}") from exc

    if len(shape) != len(record.axes):
        raise ConfigError(
            f"{record.name}: axes {record.axes!r} describe {len(record.axes)} dimensions "
            f"but {record.path} has shape {shape}"
        )

    channels = tuple(record.channels) if record.channels is not None else None
    if channels is not None:
        if "C" in record.axes:
            channel_count = shape[record.axes.index("C")]
        else:
            channel_count = 1
        if len(channels) != channel_count:
            raise ConfigError(
                f"{record.name}: {len(channels)} channel names given but the image has "
                f"{channel_count} channel(s) in {record.path}"
            )

    return ImageMetadata(
        name=record.name,
        path=record.path,
        axes=record.axes,
        shape=shape,
        dtype=dtype,
        channels=channels,
        calibration=record.calibration,
    )


def inspect_dataset(config: MicroFlowConfig) -> list[ImageMetadata]:
    return [inspect_image(record) for record in config.dataset.images]

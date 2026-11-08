"""Image and calibration file input/output."""

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import tifffile

from microflow.units import Calibration


def read_image(path: Path) -> npt.NDArray[Any]:
    return np.asarray(tifffile.imread(path))


def write_image(path: Path, image: npt.NDArray[Any], axes: str | None = None) -> None:
    """Write an image. Pass ``axes`` for stacks so the file records what each dimension means."""
    metadata = {"axes": axes} if axes else {}
    tifffile.imwrite(path, image, metadata=metadata)


def read_shape_and_dtype(path: Path) -> tuple[tuple[int, ...], str]:
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        return tuple(series.shape), str(series.dtype)


def calibration_path(image_path: Path) -> Path:
    return image_path.with_suffix(".calibration.json")


def read_calibration(path: Path) -> Calibration:
    return Calibration.model_validate_json(path.read_text(encoding="utf-8"))


def write_calibration(path: Path, calibration: Calibration) -> None:
    path.write_text(calibration.model_dump_json(indent=2) + "\n", encoding="utf-8")

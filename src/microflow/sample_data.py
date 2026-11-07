"""Deterministic synthetic images for demonstrations and tests."""

from pathlib import Path

import numpy as np
import numpy.typing as npt

from microflow.image_io import calibration_path, write_calibration, write_image
from microflow.units import Calibration

SAMPLE_PIXEL_SIZE_UM = 0.5


def make_synthetic_image(
    seed: int = 0, shape: tuple[int, int] = (256, 256), n_blobs: int = 12
) -> npt.NDArray[np.uint16]:
    rng = np.random.default_rng(seed)
    image: npt.NDArray[np.float64] = rng.normal(100.0, 5.0, shape)
    yy, xx = np.indices(shape, dtype=np.float64)
    for _ in range(n_blobs):
        cy, cx = rng.uniform(0, shape[0]), rng.uniform(0, shape[1])
        sigma = rng.uniform(3.0, 8.0)
        amplitude = rng.uniform(200.0, 800.0)
        image += amplitude * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2))
    return np.clip(image, 0, np.iinfo(np.uint16).max).astype(np.uint16)


def write_sample_dataset(out_dir: Path, seed: int = 0, count: int = 3) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    calibration = Calibration(pixel_size=SAMPLE_PIXEL_SIZE_UM, unit="um")
    paths: list[Path] = []
    for index in range(count):
        image_path = out_dir / f"synthetic_{index:02d}.tif"
        write_image(image_path, make_synthetic_image(seed + index))
        write_calibration(calibration_path(image_path), calibration)
        paths.append(image_path)
    return paths

"""Deterministic synthetic images for demonstrations and tests."""

from collections.abc import Sequence
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Disc:
    """A filled disc with a known centre (row, column) and radius, in pixels."""

    centre: tuple[float, float]
    radius: float


def make_phantom(
    discs: Sequence[Disc],
    shape: tuple[int, int] = (128, 128),
    *,
    background: float | npt.ArrayLike = 100.0,
    contrast: float = 300.0,
    noise_sd: float = 5.0,
    seed: int = 0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
    """Image with known discs, and ground-truth labels (disc i is labelled i, 0 is background).

    Where discs overlap, each shared pixel belongs to the disc whose centre is nearest, so the
    ground truth is well defined. The foreground is the union of the discs.
    """
    rng = np.random.default_rng(seed)
    image = np.broadcast_to(np.asarray(background, dtype=np.float64), shape)
    image = image + rng.normal(0.0, noise_sd, shape)
    rows, cols = np.indices(shape, dtype=np.float64)
    truth = np.zeros(shape, dtype=np.int32)
    nearest = np.full(shape, np.inf)
    for number, disc in enumerate(discs, start=1):
        distance = np.hypot(rows - disc.centre[0], cols - disc.centre[1])
        claim = (distance <= disc.radius) & (distance < nearest)
        truth[claim] = number
        nearest[claim] = distance[claim]
    image[truth > 0] += contrast
    return image, truth


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

from pathlib import Path

import numpy as np

from microflow.image_io import calibration_path, read_calibration, read_image
from microflow.sample_data import SAMPLE_PIXEL_SIZE_UM, make_synthetic_image, write_sample_dataset
from microflow.units import Calibration


def test_synthetic_image_is_deterministic_for_same_seed() -> None:
    first = make_synthetic_image(seed=7)
    second = make_synthetic_image(seed=7)
    np.testing.assert_array_equal(first, second)


def test_synthetic_image_differs_between_seeds() -> None:
    assert not np.array_equal(make_synthetic_image(seed=1), make_synthetic_image(seed=2))


def test_synthetic_image_has_expected_shape_and_dtype() -> None:
    image = make_synthetic_image(seed=0, shape=(64, 32))
    assert image.shape == (64, 32)
    assert image.dtype == np.uint16


def test_sample_dataset_round_trips_with_units(tmp_path: Path) -> None:
    paths = write_sample_dataset(tmp_path, seed=3, count=2)
    assert len(paths) == 2
    for index, path in enumerate(paths):
        np.testing.assert_array_equal(read_image(path), make_synthetic_image(seed=3 + index))
        assert read_calibration(calibration_path(path)) == Calibration(
            pixel_size=SAMPLE_PIXEL_SIZE_UM, unit="um"
        )

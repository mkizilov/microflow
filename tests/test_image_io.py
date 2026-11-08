from pathlib import Path

import numpy as np

from microflow.image_io import write_image
from microflow.loading import load_image


def test_stack_written_with_axes_reads_back_with_the_same_axes(tmp_path: Path) -> None:
    data = np.arange(2 * 4 * 5, dtype=np.uint16).reshape(2, 4, 5)
    path = tmp_path / "stack.tif"
    write_image(path, data, "CYX")
    loaded = load_image(path)
    assert loaded.axes == "CYX"
    np.testing.assert_array_equal(loaded.data, data)


def test_plane_written_without_axes_still_reads_as_yx(tmp_path: Path) -> None:
    path = tmp_path / "plane.tif"
    write_image(path, np.zeros((4, 5), dtype=np.int32))
    assert load_image(path).axes == "YX"

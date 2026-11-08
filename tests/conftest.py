from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tifffile


@pytest.fixture
def write_tiff(tmp_path: Path) -> Callable[..., Path]:
    """Write a synthetic TIFF under tmp_path. Keyword arguments go to ``tifffile.imwrite``."""

    def _write(name: str, data: np.ndarray, **kwargs: Any) -> Path:
        path = tmp_path / name
        tifffile.imwrite(path, data, **kwargs)
        return path

    return _write


@pytest.fixture
def stack_cyx() -> np.ndarray:
    """Deterministic (2, 8, 8) uint16 stack whose values differ per channel and pixel."""
    return (np.arange(2 * 8 * 8, dtype=np.uint16).reshape(2, 8, 8) * 37).astype(np.uint16)

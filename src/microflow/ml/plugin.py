"""The ``ml-unet`` segmenter plugin: a trained U-Net inside a batch analysis.

It is registered under the ``microflow.segmenters`` entry point group. The batch pipeline uses it
like any other plugin: ``segmentation: {method: plugin, plugin: {name: ml-unet, options: ...}}``.

The module imports nothing that needs PyTorch, so listing plugins stays cheap. The checkpoint's
SHA-256 is checked on every call, before the file is opened, and the same digest is part of the
analysis identity through the options. The plugin returns labels. The batch pipeline then applies
the configured minimum area and border rules, as it does for every segmenter.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from microflow.ml.runtime import MLError, require_torch
from microflow.plugins.api import PLUGIN_API_VERSION, PluginError
from microflow.provenance.hashing import sha256_file
from microflow.units import Calibration

if TYPE_CHECKING:
    from microflow.ml.predict import TiledSegmenter


class UNetOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    checkpoint: str = Field(min_length=1, description="Path to a checkpoint written by ml train.")
    checkpoint_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 of the checkpoint file; the run refuses any other file.",
    )
    device: Literal["auto", "cpu", "cuda"] = "auto"
    tile_size: int = Field(default=256, ge=8)
    overlap: int | None = Field(default=None, ge=0)
    threshold: float | None = Field(default=None, gt=0, lt=1)
    fallback_to_cpu: bool = True


class UNetSegmenter:
    name = "ml-unet"
    version = "1.0.0"
    api_version = PLUGIN_API_VERSION
    kind = "segmenter"
    options_model = UNetOptions

    def segment(
        self,
        plane: NDArray[np.float64],
        calibration: Calibration | None,
        options: UNetOptions,
    ) -> NDArray[np.int32]:
        segmenter = _segmenter(options)
        cal = None if calibration is None else (float(calibration.pixel_size), calibration.unit)
        return segmenter.segment_plane(plane, calibration=cal).labels


_CACHE: dict[tuple[str, str, str, int, int | None, float | None, bool], "TiledSegmenter"] = {}


def _verified_checkpoint(options: UNetOptions) -> Path:
    path = Path(options.checkpoint)
    if not path.is_file():
        raise PluginError(f"ml-unet: checkpoint {path} does not exist")
    digest = sha256_file(path)
    if digest != options.checkpoint_sha256:
        raise PluginError(
            f"ml-unet: checkpoint {path} has SHA-256 {digest}, but the configuration expects "
            f"{options.checkpoint_sha256}; the file changed, or the wrong file is named"
        )
    return path


def _segmenter(options: UNetOptions) -> "TiledSegmenter":
    path = _verified_checkpoint(options)
    key = (
        str(path),
        options.checkpoint_sha256,
        options.device,
        options.tile_size,
        options.overlap,
        options.threshold,
        options.fallback_to_cpu,
    )
    if key not in _CACHE:
        try:
            require_torch()
            from microflow.ml.predict import TiledSegmenter, load_model

            loaded = load_model(path)
            _CACHE[key] = TiledSegmenter(
                loaded,
                device=options.device,
                tile_size=options.tile_size,
                overlap=options.overlap,
                threshold=options.threshold,
                fallback_to_cpu=options.fallback_to_cpu,
            )
        except MLError as exc:
            raise PluginError(f"ml-unet: {exc}") from exc
    return _CACHE[key]

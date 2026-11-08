"""Loading trained models and running them on planes, on CPU or CUDA. Requires PyTorch.

Model loading validates the checkpoint before any weights reach the network (see ``model_spec``).
The network is kept on CPU. The accelerated copy, if any, is a separate copy, so the CPU model is
always available for fallback. Inference runs with gradients off and the network in evaluation mode,
so the same input gives the same output on the same machine.
"""

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from microflow.ml.checkpoint import load_checkpoint
from microflow.ml.fallback import FallbackRunner
from microflow.ml.inference import resolve_device
from microflow.ml.model_spec import ModelSpec, validate_metadata
from microflow.ml.pipeline import PlaneResult, segment_plane
from microflow.ml.runtime import CheckpointError, MLError
from microflow.ml.tiling import receptive_radius
from microflow.ml.unet import UNet, build_unet
from microflow.provenance.hashing import sha256_file

_DEVICE_FAILURE_TOKENS = ("cuda", "out of memory", "cudnn", "cublas")


@dataclass(frozen=True)
class LoadedModel:
    spec: ModelSpec
    model: UNet
    path: Path
    sha256: str


def load_model(path: Path) -> LoadedModel:
    payload = load_checkpoint(path)
    spec = validate_metadata(
        payload, source=str(path), runtime_torch_version=str(torch.__version__)
    )
    model = build_unet(spec.architecture)
    try:
        model.load_state_dict(payload["model_state"])
    except (KeyError, RuntimeError) as exc:
        raise CheckpointError(
            f"{path}: the weights do not match the declared architecture: {exc}"
        ) from exc
    model.eval()
    return LoadedModel(spec=spec, model=model, path=path, sha256=sha256_file(path))


def _is_device_failure(exc: Exception) -> bool:
    return isinstance(exc, RuntimeError) and any(
        token in str(exc).lower() for token in _DEVICE_FAILURE_TOKENS
    )


def _forward(
    model: UNet, window: NDArray[np.float64], device: torch.device
) -> NDArray[np.float32]:
    with torch.no_grad():
        tensor = torch.as_tensor(window, dtype=torch.float32, device=device)
        logits: torch.Tensor = model(tensor[None, None])
        probabilities = torch.sigmoid(logits)[0, 0]
    result: NDArray[np.float32] = probabilities.cpu().numpy()
    return result


class TiledSegmenter:
    """Segments planes by tiled inference with one trained model.

    ``device`` is ``auto``, ``cpu`` or ``cuda``. ``auto`` uses CUDA when it is present. With
    ``fallback_to_cpu`` (the default), a recoverable CUDA failure moves the rest of that run to CPU,
    and each affected plane's quality report lists ``device_fallback``. An explicit ``cuda`` with no
    CUDA device is an error, not a silent change of device.
    """

    def __init__(
        self,
        loaded: LoadedModel,
        *,
        device: str = "auto",
        tile_size: int = 256,
        overlap: int | None = None,
        threshold: float | None = None,
        fallback_to_cpu: bool = True,
    ) -> None:
        depth = loaded.spec.architecture["depth"]
        multiple = 2**depth
        if tile_size < multiple or tile_size % multiple:
            raise MLError(
                f"tile_size must be a positive multiple of {multiple} for this depth-{depth} "
                f"model; got {tile_size}"
            )
        self.radius = receptive_radius(depth)
        self.overlap = self.radius if overlap is None else overlap
        if self.overlap < 0 or 2 * self.overlap >= tile_size:
            raise MLError(
                f"overlap {self.overlap} must be at least 0 and less than half the tile size "
                f"({tile_size})"
            )
        self.tile_size = tile_size
        self.multiple = multiple
        self.threshold = loaded.spec.threshold if threshold is None else threshold
        if not 0 < self.threshold < 1:
            raise MLError(f"threshold must be between 0 and 1; got {self.threshold}")
        self._loaded = loaded
        self._device = resolve_device(device)
        self._fallback_enabled = fallback_to_cpu and self._device.type != "cpu"
        self._accelerated = (
            copy.deepcopy(loaded.model).to(self._device)
            if self._device.type != "cpu"
            else loaded.model
        )

    @property
    def device(self) -> str:
        return str(self._device)

    def segment_plane(
        self,
        plane: NDArray[np.float64],
        *,
        calibration: tuple[float, str] | None = None,
        min_area_px: float = 0.0,
        exclude_border: bool = False,
    ) -> PlaneResult:
        runner = FallbackRunner(
            self._primary_window,
            self._cpu_window if self._fallback_enabled else None,
            is_recoverable=_is_device_failure,
        )
        return segment_plane(
            plane,
            normalization=self._loaded.spec.normalization,
            predict=runner,
            multiple=self.multiple,
            tile_size=self.tile_size,
            overlap=self.overlap,
            receptive_radius=self.radius,
            threshold=self.threshold,
            min_area_px=min_area_px,
            exclude_border=exclude_border,
            calibration=calibration,
            training_calibrations=self._loaded.spec.training_calibrations,
            fell_back=lambda: runner.fell_back,
        )

    def _primary_window(self, window: NDArray[np.float64]) -> NDArray[np.float32]:
        return _forward(self._accelerated, window, self._device)

    def _cpu_window(self, window: NDArray[np.float64]) -> NDArray[np.float32]:
        return _forward(self._loaded.model, window, torch.device("cpu"))

"""Whole-image prediction and scoring, shared by validation during training and by evaluation.
Requires PyTorch.

Images are padded with zeros, which is the normalised mean, up to a multiple of ``2**depth``. The
logits are cropped back to the original size before the loss and the overlap scores are computed,
so padding never contributes to a score.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from microflow.ml.losses import segmentation_loss
from microflow.ml.metrics import score_probabilities
from microflow.ml.preparation import ImagePair, Normalization
from microflow.ml.runtime import MLError
from microflow.ml.unet import UNet


@dataclass(frozen=True)
class ImageResult:
    sample: str
    image_path: Path
    loss: float
    dice: float
    iou: float
    predicted_pixels: int
    truth_pixels: int


def pad_to_multiple(image: NDArray[np.float64], multiple: int) -> NDArray[np.float64]:
    rows, cols = image.shape
    padded: NDArray[np.float64] = np.pad(
        image, ((0, (-rows) % multiple), (0, (-cols) % multiple)), mode="constant"
    )
    return padded


def predict_logits(
    model: UNet, image: NDArray[np.float64], device: torch.device, multiple: int
) -> torch.Tensor:
    rows, cols = image.shape
    batch = torch.as_tensor(pad_to_multiple(image, multiple), dtype=torch.float32, device=device)
    with torch.no_grad():
        logits: torch.Tensor = model(batch[None, None])
    cropped: torch.Tensor = logits[0, 0, :rows, :cols]
    return cropped


def score_pairs(
    model: UNet,
    pairs: Sequence[ImagePair],
    normalization: Normalization,
    *,
    device: torch.device,
    multiple: int,
    threshold: float,
    bce_weight: float,
) -> tuple[ImageResult, ...]:
    """Score each image as a whole. Puts the model in evaluation mode; callers restore training
    mode themselves."""
    model.eval()
    results: list[ImageResult] = []
    with torch.no_grad():
        for pair in pairs:
            logits = predict_logits(model, normalization.apply(pair.image), device, multiple)
            target = torch.as_tensor(pair.mask, dtype=torch.float32, device=device)
            loss = segmentation_loss(logits[None, None], target[None, None], bce_weight)
            probabilities: NDArray[np.float32] = torch.sigmoid(logits).cpu().numpy()
            scores = score_probabilities(probabilities, pair.mask, threshold)
            results.append(
                ImageResult(
                    sample=pair.sample,
                    image_path=pair.image_path,
                    loss=float(loss.item()),
                    dice=scores.dice,
                    iou=scores.iou,
                    predicted_pixels=scores.predicted_pixels,
                    truth_pixels=scores.truth_pixels,
                )
            )
    return tuple(results)


def resolve_device(name: str) -> torch.device:
    cuda_available = torch.cuda.is_available()
    if name == "auto":
        return torch.device("cuda" if cuda_available else "cpu")
    if name == "cuda" and not cuda_available:
        raise MLError("device 'cuda' was requested but CUDA is not available; use 'cpu' or 'auto'")
    return torch.device(name)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)

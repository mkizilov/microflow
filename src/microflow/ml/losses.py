"""Loss for binary segmentation: binary cross-entropy plus soft Dice. Requires PyTorch."""

import torch
from torch.nn import functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """1 - soft Dice, computed per image over channel, height and width, then averaged."""
    probabilities = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (probabilities * target).sum(dim=dims)
    totals = probabilities.sum(dim=dims) + target.sum(dim=dims)
    dice: torch.Tensor = (2 * intersection + eps) / (totals + eps)
    return (1 - dice).mean()


def segmentation_loss(
    logits: torch.Tensor, target: torch.Tensor, bce_weight: float
) -> torch.Tensor:
    bce: torch.Tensor = F.binary_cross_entropy_with_logits(logits, target)
    return bce_weight * bce + (1 - bce_weight) * soft_dice_loss(logits, target)

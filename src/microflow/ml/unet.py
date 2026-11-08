"""A small U-Net for binary segmentation. Requires PyTorch."""

from collections.abc import Mapping

import torch
from torch import nn


class _DoubleConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    """Encoder-decoder with skip connections. Each level halves the resolution and doubles the
    channels. The input height and width must be multiples of ``2**depth``; the caller pads."""

    def __init__(self, *, base_channels: int, depth: int, in_channels: int = 1) -> None:
        super().__init__()
        self.depth = depth
        widths = [base_channels * 2**level for level in range(depth + 1)]
        self.encoders = nn.ModuleList()
        channels = in_channels
        for width in widths:
            self.encoders.append(_DoubleConv(channels, width))
            channels = width
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for width in reversed(widths[:-1]):
            self.upsamples.append(nn.ConvTranspose2d(channels, width, kernel_size=2, stride=2))
            self.decoders.append(_DoubleConv(2 * width, width))
            channels = width
        self.head = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        multiple = 2**self.depth
        if x.shape[-2] % multiple or x.shape[-1] % multiple:
            raise ValueError(
                f"input height and width must be multiples of {multiple}; got {tuple(x.shape[-2:])}"
            )
        skips: list[torch.Tensor] = []
        for level, encoder in enumerate(self.encoders):
            x = encoder(x)
            if level < self.depth:
                skips.append(x)
                x = self.pool(x)
        for upsample, decoder, skip in zip(
            self.upsamples, self.decoders, reversed(skips), strict=True
        ):
            x = upsample(x)
            x = decoder(torch.cat([x, skip], dim=1))
        logits: torch.Tensor = self.head(x)
        return logits


def build_unet(architecture: Mapping[str, int]) -> UNet:
    return UNet(
        base_channels=architecture["base_channels"],
        depth=architecture["depth"],
        in_channels=architecture["in_channels"],
    )

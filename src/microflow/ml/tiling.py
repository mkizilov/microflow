"""Tiled inference: predict large planes window by window, without seams. Torch-free.

A plane is first padded with zeros (the normalised mean) to a multiple of ``2**depth``. The padded
plane is then cut into windows of ``tile_size``. Neighbouring windows overlap by ``overlap`` pixels,
and each window is predicted whole. Every pixel is taken from exactly one window, the one in which
it is a core pixel, so nothing is averaged and each pixel has one probability.

Each core lies at least ``overlap`` pixels inside its window, except where the window meets the
edge of the padded plane. If ``overlap`` covers the network's receptive field, every core pixel sees
the same context as in whole-plane prediction, and the two results are equal.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Box = tuple[int, int, int, int]
"""Top, bottom, left and right, with bottom and right exclusive."""


def receptive_radius(depth: int) -> int:
    """Upper bound, in pixels, on how far an output pixel can depend on input pixels.

    For a U-Net with ``depth`` downsamplings, each 3 x 3 convolution adds one step of its level's
    stride to the radius, and each 2 x 2 pooling or transposed convolution adds at most one step.
    Summing along the deepest path gives ``8 * 2**depth - 6``. This is a bound, not the exact
    support, and it is checked against the measured gradient support in the test suite.
    """
    if depth < 1:
        raise ValueError(f"depth must be at least 1; got {depth}")
    return 8 * 2**depth - 6


def pad_to_multiple(image: NDArray[np.float64], multiple: int) -> NDArray[np.float64]:
    """Pad at the bottom and right with zeros so both sides are multiples of ``multiple``."""
    rows, cols = image.shape
    padded: NDArray[np.float64] = np.pad(
        image, ((0, (-rows) % multiple), (0, (-cols) % multiple)), mode="constant"
    )
    return padded


@dataclass(frozen=True)
class Tile:
    window: Box
    core: Box


@dataclass(frozen=True)
class TilePlan:
    shape: tuple[int, int]
    tile_size: int
    overlap: int
    tiles: tuple[Tile, ...]


def plan_tiles(
    shape: tuple[int, int], *, tile_size: int, overlap: int, multiple: int = 1
) -> TilePlan:
    if multiple < 1 or tile_size < multiple or tile_size % multiple:
        raise ValueError(f"tile_size must be a positive multiple of {multiple}; got {tile_size}")
    if overlap < 0:
        raise ValueError(f"overlap cannot be negative; got {overlap}")
    if 2 * overlap >= tile_size:
        raise ValueError(
            f"overlap {overlap} leaves no core pixels in a tile of {tile_size}; "
            "it must be less than half the tile size"
        )
    rows = _spans(shape[0], tile_size, overlap)
    cols = _spans(shape[1], tile_size, overlap)
    tiles = tuple(
        Tile(
            window=(row_window[0], row_window[1], col_window[0], col_window[1]),
            core=(row_window[2], row_window[3], col_window[2], col_window[3]),
        )
        for row_window in rows
        for col_window in cols
    )
    return TilePlan(shape=(shape[0], shape[1]), tile_size=tile_size, overlap=overlap, tiles=tiles)


def _spans(length: int, tile: int, overlap: int) -> list[tuple[int, int, int, int]]:
    """Per axis: (window start, window end, core start, core end)."""
    if length <= tile:
        return [(0, length, 0, length)]
    step = tile - 2 * overlap
    spans: list[tuple[int, int, int, int]] = []
    for core_start in range(0, length, step):
        core_end = min(length, core_start + step)
        window_start = min(max(core_start - overlap, 0), length - tile)
        spans.append((window_start, window_start + tile, core_start, core_end))
    return spans


def predict_tiled(
    image: NDArray[np.float64],
    predict: Callable[[NDArray[np.float64]], NDArray[np.float32]],
    plan: TilePlan,
) -> NDArray[np.float32]:
    """Predict every window and copy each window's core into the output. Windows are visited in
    plan order, so the result does not depend on how fast each one runs."""
    if image.shape != plan.shape:
        raise ValueError(f"the image shape {image.shape} does not match the plan {plan.shape}")
    output = np.zeros(plan.shape, dtype=np.float32)
    covered = np.zeros(plan.shape, dtype=bool)
    for tile in plan.tiles:
        top, bottom, left, right = tile.window
        window = image[top:bottom, left:right]
        probabilities = np.asarray(predict(window))
        if probabilities.shape != window.shape:
            raise ValueError(
                f"the predictor returned shape {probabilities.shape} for a window of shape "
                f"{window.shape}"
            )
        core_top, core_bottom, core_left, core_right = tile.core
        output[core_top:core_bottom, core_left:core_right] = probabilities[
            core_top - top : core_bottom - top, core_left - left : core_right - left
        ]
        covered[core_top:core_bottom, core_left:core_right] = True
    if not covered.all():
        raise RuntimeError("the tiles do not cover the plane; this is a bug in the tile plan")
    return output

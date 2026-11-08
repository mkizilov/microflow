"""Comparing two label images, such as an ML segmentation and a classical one. Torch-free.

Objects are matched one to one. Pairs are taken in order of decreasing IoU, and a pair is accepted
when its IoU reaches ``match_iou`` and neither object is already matched. Ties are broken by label
number, so the result is deterministic.

Label numbers restart in every plane, so stacks are compared plane by plane. Pixel-level Dice
and IoU are computed over the whole stack.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from microflow.segmentation.metrics import score_segmentation


@dataclass(frozen=True)
class LabelComparison:
    dice: float
    iou: float
    objects_a: int
    objects_b: int
    matched_objects: int
    match_iou: float
    planes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "dice": self.dice,
            "iou": self.iou,
            "objects_a": self.objects_a,
            "objects_b": self.objects_b,
            "matched_objects": self.matched_objects,
            "match_iou": self.match_iou,
            "planes": self.planes,
        }


def compare_label_stacks(
    labels_a: NDArray[Any],
    labels_b: NDArray[Any],
    axes: str,
    *,
    match_iou: float = 0.5,
) -> LabelComparison:
    if not 0 < match_iou <= 1:
        raise ValueError(f"match_iou must be in (0, 1]; got {match_iou}")
    a = np.asarray(labels_a, dtype=np.int64)
    b = np.asarray(labels_b, dtype=np.int64)
    if a.shape != b.shape:
        raise ValueError(f"label shapes differ: {a.shape} and {b.shape}")
    if "Y" not in axes or "X" not in axes:
        raise ValueError(f"the labels need Y and X axes; got {axes!r}")
    spatial = (axes.index("Y"), axes.index("X"))
    rows, cols = a.shape[spatial[0]], a.shape[spatial[1]]
    planes_a = np.moveaxis(a, spatial, (-2, -1)).reshape(-1, rows, cols)
    planes_b = np.moveaxis(b, spatial, (-2, -1)).reshape(-1, rows, cols)
    matched = 0
    objects_a = 0
    objects_b = 0
    for plane_a, plane_b in zip(planes_a, planes_b, strict=True):
        objects_a += _object_count(plane_a)
        objects_b += _object_count(plane_b)
        matched += _matched_objects(plane_a, plane_b, match_iou)
    overall = score_segmentation(a > 0, b > 0)
    return LabelComparison(
        dice=overall.dice,
        iou=overall.iou,
        objects_a=objects_a,
        objects_b=objects_b,
        matched_objects=matched,
        match_iou=match_iou,
        planes=int(planes_a.shape[0]),
    )


def _object_count(plane: NDArray[np.int64]) -> int:
    return int(np.unique(plane[plane > 0]).size)


def _matched_objects(a: NDArray[np.int64], b: NDArray[np.int64], match_iou: float) -> int:
    both = (a > 0) & (b > 0)
    if not both.any():
        return 0
    pairs, shared = np.unique(np.stack([a[both], b[both]], axis=1), axis=0, return_counts=True)
    area_a = np.bincount(a.ravel())
    area_b = np.bincount(b.ravel())
    union = area_a[pairs[:, 0]] + area_b[pairs[:, 1]] - shared
    iou = shared / union
    order = sorted(
        range(len(pairs)),
        key=lambda index: (-iou[index], int(pairs[index, 0]), int(pairs[index, 1])),
    )
    used_a: set[int] = set()
    used_b: set[int] = set()
    matched = 0
    for index in order:
        if iou[index] < match_iou:
            break
        left, right = int(pairs[index, 0]), int(pairs[index, 1])
        if left in used_a or right in used_b:
            continue
        used_a.add(left)
        used_b.add(right)
        matched += 1
    return matched

"""Classical segmentation operations on one two-dimensional plane.

Masks are boolean arrays. Label images are int32 with 0 as background and objects numbered
1..N in raster order of their first pixel. Sizes are in pixels here; the pipeline converts
physical units. Invalid (NaN) pixels are never foreground.
"""

import numpy as np
import numpy.typing as npt
from scipy import ndimage
from skimage import feature, filters
from skimage import segmentation as sk_segmentation

Plane = npt.NDArray[np.float64]
Mask = npt.NDArray[np.bool_]
Labels = npt.NDArray[np.int32]

_EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)


def foreground_mask(
    plane: Plane, method: str, threshold: float | None, block_size: int, offset: float
) -> tuple[Mask, float | None]:
    """Foreground mask, and the global intensity level used (None for adaptive thresholding)."""
    valid = np.isfinite(plane)
    if not valid.any():
        return np.zeros(plane.shape, dtype=bool), None
    if method == "adaptive":
        filled = np.where(valid, plane, float(np.median(plane[valid])))
        local = filters.threshold_local(filled, block_size=block_size, offset=-offset)
        return valid & (filled > local), None
    if method == "otsu":
        values = plane[valid]
        if values.min() == values.max():
            level = float(values.max())
        else:
            level = float(filters.threshold_otsu(values))
    elif threshold is None:
        raise ValueError("the threshold method needs a threshold value")
    else:
        level = threshold
    return valid & (plane > level), level


def disk(radius_px: float) -> Mask:
    radius = round(radius_px)
    rows, cols = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return rows * rows + cols * cols <= radius * radius


def open_mask(mask: Mask, radius_px: float) -> Mask:
    """Erosion then dilation. Outside the image counts as foreground while eroding, so objects
    that touch the image edge keep their edge pixels."""
    structure = disk(radius_px)
    eroded = ndimage.binary_erosion(mask, structure=structure, border_value=1)
    return ndimage.binary_dilation(eroded, structure=structure)


def close_mask(mask: Mask, radius_px: float) -> Mask:
    """Dilation then erosion, with the same image-edge convention as :func:`open_mask`."""
    structure = disk(radius_px)
    dilated = ndimage.binary_dilation(mask, structure=structure)
    return ndimage.binary_erosion(dilated, structure=structure, border_value=1)


def label_objects(mask: Mask) -> Labels:
    labels, _ = ndimage.label(mask, structure=_EIGHT_CONNECTED)
    return labels.astype(np.int32)


def split_touching(mask: Mask, components: Labels, min_distance_px: float) -> Labels:
    """Split merged objects with a marker-based watershed on the distance transform.

    Markers are the distance-transform peaks at least ``min_distance_px`` apart. Any object
    without a peak receives its deepest point as a marker, so no object is lost.
    """
    distance = ndimage.distance_transform_edt(mask)
    peaks = feature.peak_local_max(
        distance,
        min_distance=max(1, round(min_distance_px)),
        labels=components,
        exclude_border=False,
    )
    markers = np.zeros(mask.shape, dtype=np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1, dtype=np.int32)
    covered = set(np.unique(components[tuple(peaks.T)]).tolist())
    missing = [number for number in range(1, int(components.max()) + 1) if number not in covered]
    next_id = len(peaks)
    if missing:
        for position in ndimage.maximum_position(distance, components, missing):
            next_id += 1
            markers[position] = next_id
    return sk_segmentation.watershed(-distance, markers, mask=mask).astype(np.int32)


def remove_small(labels: Labels, min_area_px: float) -> Labels:
    """Remove objects with fewer than ``min_area_px`` pixels (the minimum itself is kept)."""
    areas = np.bincount(labels.ravel())
    keep = areas >= min_area_px
    keep[0] = False
    return np.where(keep[labels], labels, 0).astype(np.int32)


def remove_border(labels: Labels) -> Labels:
    edge = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    touching = np.unique(edge[edge != 0])
    return np.where(np.isin(labels, touching), 0, labels).astype(np.int32)


def relabel_in_raster_order(labels: Labels) -> Labels:
    values, first, inverse = np.unique(labels.ravel(), return_index=True, return_inverse=True)
    objects = np.flatnonzero(values)
    ordered = objects[np.argsort(first[objects])]
    new_ids = np.zeros(len(values), dtype=np.int32)
    new_ids[ordered] = np.arange(1, len(ordered) + 1, dtype=np.int32)
    return new_ids[inverse].reshape(labels.shape)

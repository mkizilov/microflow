"""Work for one image: load, preprocess, segment, measure and report.

``process_image`` runs in a worker process. An ordinary error for one image (an unreadable file,
a missing calibration, a bad channel name) becomes a failed :class:`ItemRecord`, so it cannot stop
the rest of the batch. Only interruptions and system failures escape.
"""

import csv
import dataclasses
import math
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from microflow.batch import store
from microflow.batch.records import ItemRecord, OutputFile
from microflow.config import QualityControlConfig, SegmentationConfig
from microflow.image_io import write_image
from microflow.loading import LoadedImage, load_array, load_image
from microflow.preprocessing import preprocess
from microflow.preprocessing.steps import Step
from microflow.quantification import IntensitySource, measure_plane
from microflow.reporting.figures import render_figures
from microflow.reporting.html import render_html
from microflow.reporting.qc import build_quality_report, inputs_from_images
from microflow.segmentation import segment_plane
from microflow.units import Calibration

TRACEBACK_LIMIT = 4000


@dataclass(frozen=True)
class ImageJob:
    name: str
    path: Path
    expected_axes: str | None
    channels: tuple[str, ...] | None
    calibration: Calibration | None
    item_dir: Path
    input_sha256: str
    config_fingerprint: str
    estimated_mb: float
    steps: tuple[Step, ...]
    segmentation: SegmentationConfig
    segmentation_channel: str | None
    quality: QualityControlConfig


@dataclass(frozen=True)
class _Outcome:
    outputs: dict[str, OutputFile]
    objects: int
    flags: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class _Planes:
    """Planes of an image: ``data`` is shaped (planes, channels, Y, X) in the order of ``names``."""

    lead_axes: str
    lead_shape: tuple[int, ...]
    names: tuple[str, ...]
    data: npt.NDArray[Any]


def process_image(job: ImageJob) -> ItemRecord:
    started = time.perf_counter()
    try:
        store.clear_item(job.item_dir)
        outcome = _run_image(job)
        record = ItemRecord(
            name=job.name,
            status="ok",
            input_sha256=job.input_sha256,
            config_fingerprint=job.config_fingerprint,
            estimated_mb=job.estimated_mb,
            objects=outcome.objects,
            flags=outcome.flags,
            warnings=outcome.warnings,
            elapsed_s=time.perf_counter() - started,
            outputs=outcome.outputs,
        )
    except Exception as exc:
        store.clear_item(job.item_dir)
        record = ItemRecord(
            name=job.name,
            status="failed",
            input_sha256=job.input_sha256,
            config_fingerprint=job.config_fingerprint,
            estimated_mb=job.estimated_mb,
            error=f"{type(exc).__name__}: {exc}",
            detail=traceback.format_exc()[-TRACEBACK_LIMIT:],
            elapsed_s=time.perf_counter() - started,
        )
    store.write_result(job.item_dir, record)
    return record


def _run_image(job: ImageJob) -> _Outcome:
    warnings: list[str] = []
    raw = _with_channels(
        load_image(job.path, axes=job.expected_axes, calibration=job.calibration), job.channels
    )
    channels = raw.channels or tuple(f"c{index}" for index in range(_channel_count(raw)))
    if raw.calibration is None:
        warnings.append("no pixel calibration: lengths and areas are in pixels")

    processed = preprocess(raw, job.steps)
    measure_corrected = not any(step.kind == "normalize" for step in job.steps)
    if not measure_corrected:
        warnings.append("normalize step present: corrected intensities are not measured")

    raw_planes = _planes(raw.data, raw.axes)
    processed_planes = _planes(processed.data, processed.axes)
    selected = _channel_index(channels, job.segmentation_channel)
    count = raw_planes.data.shape[0]
    height, width = raw_planes.data.shape[-2:]

    labels = np.zeros((count, height, width), dtype=np.int32)
    for index in range(count):
        plane = np.asarray(processed_planes.data[index, selected], dtype=np.float64)
        labels[index], _ = segment_plane(plane, job.segmentation, raw.calibration)
    objects = sum(int(labels[index].max()) for index in range(count))

    labels_axes = raw_planes.lead_axes + "YX"
    labels_stack = labels.reshape(raw_planes.lead_shape + (height, width))
    outputs: dict[str, OutputFile] = {
        "labels": store.write_output(
            job.item_dir,
            "labels.tif",
            lambda path: write_image(path, labels_stack, labels_axes),
        )
    }

    tables = []
    rows: list[dict[str, Any]] = []
    for index in range(count):
        sources = [
            IntensitySource(
                name="raw",
                planes=np.asarray(raw_planes.data[index], dtype=np.float64),
                channels=channels,
            )
        ]
        if measure_corrected:
            sources.append(
                IntensitySource(
                    name="corrected",
                    planes=np.asarray(processed_planes.data[index], dtype=np.float64),
                    channels=channels,
                )
            )
        table = measure_plane(labels[index], sources, raw.calibration)
        tables.append(table)
        rows.extend({"plane": raw_planes.names[index], **row} for row in table.rows())
    columns = ["plane", *tables[0].columns()]
    outputs["measurements"] = store.write_output(
        job.item_dir,
        "measurements.csv",
        lambda path: _write_csv(path, columns, rows),
    )

    inputs = inputs_from_images(
        load_array(
            _selected_stack(raw_planes, selected, height, width),
            axes=labels_axes,
            calibration=raw.calibration,
        ),
        load_array(labels_stack, axes=labels_axes),
        job.quality,
        preprocessed=load_array(
            _selected_stack(processed_planes, selected, height, width), axes=labels_axes
        ),
        title=f"Quality control: {job.name}",
    )
    report = build_quality_report(inputs)
    page = render_html(report, render_figures(inputs, report))
    outputs["qc_report"] = store.write_output(
        job.item_dir,
        "qc_report.html",
        lambda path: _write_text(path, page),
    )

    flags = tuple(flag.code for flag in report.flags)
    if flags:
        warnings.append(f"{len(flags)} QC flag(s): {', '.join(sorted(set(flags)))}")
    return _Outcome(
        outputs=outputs,
        objects=objects,
        flags=flags,
        warnings=tuple(warnings),
    )


def _with_channels(image: LoadedImage, channels: tuple[str, ...] | None) -> LoadedImage:
    if channels is None:
        return image
    expected = _channel_count(image)
    if len(channels) != expected:
        raise ValueError(f"{len(channels)} channel name(s) given for {expected} channel(s)")
    return dataclasses.replace(image, channels=tuple(channels))


def _channel_count(image: LoadedImage) -> int:
    return image.axis_size("C") if "C" in image.axes else 1


def _channel_index(channels: tuple[str, ...], selector: str | None) -> int:
    if selector is None:
        return 0
    if selector not in channels:
        raise ValueError(f"segmentation channel {selector!r} is not one of {list(channels)}")
    return channels.index(selector)


def _planes(data: npt.ArrayLike, axes: str) -> _Planes:
    """Arrange an image as planes of (channels, Y, X), in the order of its non-spatial axes."""
    lead = [index for index, axis in enumerate(axes) if axis not in "CYX"]
    inner = ([axes.index("C")] if "C" in axes else []) + [axes.index("Y"), axes.index("X")]
    moved = np.transpose(np.asarray(data), lead + inner)
    lead_shape = tuple(int(size) for size in moved.shape[: len(lead)])
    stacked = moved.reshape((math.prod(lead_shape),) + moved.shape[len(lead) :])
    if "C" not in axes:
        stacked = stacked[:, np.newaxis]
    lead_axes = "".join(axes[index] for index in lead)
    names = tuple(
        "".join(
            f"{axis}{position}"
            for axis, position in zip(lead_axes, position_index, strict=True)
        )
        or "image"
        for position_index in np.ndindex(lead_shape)
    )
    return _Planes(lead_axes=lead_axes, lead_shape=lead_shape, names=names, data=stacked)


def _selected_stack(planes: _Planes, channel: int, height: int, width: int) -> npt.NDArray[Any]:
    return planes.data[:, channel].reshape(planes.lead_shape + (height, width))


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")

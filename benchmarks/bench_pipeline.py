"""Benchmark the core classical pipeline on synthetic images: segmentation and measurement.

Run from the repository root with the dependencies installed:

    uv run python benchmarks/bench_pipeline.py --sizes 256 512 --repeats 3 \
      --output benchmarks/out/pipeline.json

For each image size the script generates one synthetic image (fixed seed, in memory, nothing
written to disk), then times classical segmentation and per-object measurement. Each timing is
repeated and the median is reported, with the peak Python allocation measured separately by
tracemalloc (which slows execution, so it is never mixed into the timings). The results are JSON,
with the software and platform recorded so that numbers are compared only like with like.

Argument checks run before the scientific stack is imported, so a bad size fails at once.

No numbers are reported in this repository. They must be produced on the machine that will be
quoted; see docs/benchmarks.md.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from typing import Any

MAX_SIZE = 2048
MIN_SIZE = 64
DISC_DENSITY_PER_PIXEL = 14 / (128 * 128)


def _timed(function: Callable[[], Any], repeats: int) -> tuple[float, float]:
    runs = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        runs.append(time.perf_counter() - start)
    return statistics.median(runs), min(runs)


def _peak_bytes(function: Callable[[], Any]) -> int:
    tracemalloc.start()
    try:
        function()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak


def benchmark(size: int, repeats: int) -> dict[str, Any]:
    import numpy as np

    from microflow.config import SegmentationConfig
    from microflow.quantification import IntensitySource, measure_plane
    from microflow.sample_data import Disc, make_phantom
    from microflow.segmentation.pipeline import segment_plane
    from microflow.units import Calibration

    pixel_size = Calibration(pixel_size=0.5, unit="um")
    config = SegmentationConfig()
    rng = np.random.default_rng(0)
    count = max(4, round(size * size * DISC_DENSITY_PER_PIXEL))
    discs = []
    for _ in range(count):
        radius = float(rng.uniform(5.0, 8.0))
        low, high = radius, size - radius
        centre = (float(rng.uniform(low, high)), float(rng.uniform(low, high)))
        discs.append(Disc(centre=centre, radius=radius))
    image, _ = make_phantom(discs, (size, size), seed=0)

    def segment() -> Any:
        labels, _ = segment_plane(image, config, pixel_size)
        return labels

    labels = segment()
    intensity = [IntensitySource(name="raw", planes=image[np.newaxis], channels=("c0",))]

    def measure() -> None:
        measure_plane(labels, intensity, pixel_size)

    segment_median, segment_best = _timed(segment, repeats)
    measure_median, measure_best = _timed(measure, repeats)
    return {
        "size_px": size,
        "pixels": size * size,
        "objects": int(labels.max()),
        "repeats": repeats,
        "segment_median_s": round(segment_median, 6),
        "segment_best_s": round(segment_best, 6),
        "measure_median_s": round(measure_median, 6),
        "measure_best_s": round(measure_best, 6),
        "segment_peak_mb": round(_peak_bytes(segment) / 2**20, 2),
        "measure_peak_mb": round(_peak_bytes(measure) / 2**20, 2),
    }


def _libraries() -> dict[str, str]:
    from microflow.provenance.versions import library_versions

    return library_versions(("numpy", "scipy", "scikit-image", "tifffile"))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 512])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None, help="Write the JSON here too.")
    args = parser.parse_args(argv)
    if any(size < MIN_SIZE or size > MAX_SIZE for size in args.sizes):
        parser.error(f"sizes must be between {MIN_SIZE} and {MAX_SIZE} pixels, to bound memory use")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    from microflow import __version__

    report = {
        "microflow": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "libraries": _libraries(),
        "results": [benchmark(size, args.repeats) for size in args.sizes],
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

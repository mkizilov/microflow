# Benchmarks

`benchmarks/bench_pipeline.py` times the classical pipeline (segmentation and per-object measurement)
on synthetic images, and records the peak Python allocation of each step.

## Status

One run is committed: `benchmarks/results/pipeline-macos-arm64-py312.json`. It was produced on one
machine (macOS 27.2, arm64, Python 3.12.11, NumPy 2.5.4, SciPy 1.18.1, scikit-image 0.26.0). It has
three repeats per size, at 256 and 512 pixels. The medians are:

| Size (px) | Objects | Segmentation median (s) | Measurement median (s) | Segmentation peak (MB) | Measurement peak (MB) |
| --- | --- | --- | --- | --- | --- |
| 256 | 44 | 0.001839 | 0.007697 | 3.19 | 0.25 |
| 512 | 166 | 0.007539 | 0.037139 | 9.50 | 0.93 |

This is a smoke measurement. It shows that the script runs and gives the order of magnitude of a plane's
time. It is not a benchmark campaign: three repeats cannot show the variance, and no other machine, system
or batch-level workload has been measured. Compare results only on the same machine and software.

## Running

```sh
uv sync
uv run python benchmarks/bench_pipeline.py --sizes 256 512 1024 --repeats 5 \
    --output benchmarks/out/pipeline-$(date -u +%Y%m%d).json
```

Sizes are between 64 and 2048 pixels per side, which bounds memory use. The output directory
`benchmarks/out/` is git-ignored. Delete the file when you have copied the numbers you need.

## What is measured

| Field | Meaning |
| --- | --- |
| `segment_median_s`, `segment_best_s` | Wall-clock time of `segment_plane` (Otsu, labelling and size filtering), median and best of `repeats` runs. |
| `measure_median_s`, `measure_best_s` | Wall-clock time of `measure_plane` on the segmentation, with one raw intensity source. |
| `segment_peak_mb`, `measure_peak_mb` | Peak Python allocation during one run, from `tracemalloc`. Timing and memory are measured in separate runs, because tracing slows execution. |
| `objects` | Number of objects in the synthetic plane, so the time per object can be derived. |

The image has a constant density of discs per 128 by 128 pixels, so larger sizes contain
proportionally more objects.

## What is not measured

- The batch engine, its process pool, and file I/O. Those depend on the disk and on the number of
  workers, so they need their own runs.
- Statistical tests, ML training and inference. Training depends on the hardware, and a GPU result
  needs a GPU.
- Peak resident memory of the whole process. `tracemalloc` counts Python allocations only, so NumPy
  buffers are included only where NumPy reports them to Python's allocator.

## Reporting

A reported benchmark should give the JSON file, the commit hash, the `libraries` section, and the
hardware (CPU model, core count, memory). Compare runs only on the same machine and software.

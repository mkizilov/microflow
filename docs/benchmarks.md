# Benchmarks

`benchmarks/bench_pipeline.py` times the classical pipeline (segmentation and per-object measurement)
on synthetic images, and records the peak Python allocation of each step.

## Status

**No benchmark numbers have been measured or reported for this version.** The authoring environment
had no scientific Python stack and no network, so the script was not run. The numbers that appear in
any future report must come from a run on the machine, with the software versions, that the report
names.

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

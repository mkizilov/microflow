# Batch processing

`microflow batch CONFIG [--workers N]` runs every image in a dataset through the whole chain:
load, preprocess, segment, measure, and report. It writes per-image outputs and a run record.
See `examples/batch.yaml`.

```sh
uv run microflow batch examples/batch.yaml --workers 2
```

## Configuration

| Key | Meaning |
| --- | --- |
| `name` | Label for the run. |
| `dataset.images` | Images, each with `name`, `path`, and optional `axes`, `channels`, `calibration`. A declared `axes` is checked against the file. |
| `preprocessing` | Steps in the `microflow preprocess` format. They run in order before segmentation. |
| `segmentation` | Segmentation settings, as for `microflow segment`. |
| `segmentation_channel` | Channel name to segment. Default: the first channel. |
| `quality_control` | Thresholds for the per-image QC report. |
| `output_directory` | Where the run is written. |
| `workers` | Images processed in parallel, from 1 to 8. Default 2. |
| `memory_budget_mb` | Estimated working-set limit across images running at once. Default: none. |

Relative paths resolve against the configuration file's directory. Unknown keys are rejected.

## Per-image processing

For each image (every plane of any extra axes, such as time or depth):

1. Load the image, applying any declared axes, channels, and calibration.
2. Run the preprocessing steps.
3. Segment the chosen channel of the preprocessed plane.
4. Measure geometry and intensities with `microflow.quantification`:
   - the raw intensities, for every channel
   - the corrected (preprocessed) intensities, unless a `normalize` step is present
5. Build the QC report from the selected channel.

Outputs for each image go in `items/<name>-<hash>/`:

| File | Contents |
| --- | --- |
| `labels.tif` | int32 label image with the same leading axes as the input. 0 is background. |
| `measurements.csv` | One row per object and plane, with a `plane` column. Columns are as in `docs/measurements.md`. |
| `qc_report.html` | The self-contained QC report for this image. |
| `result.json` | The image's outcome, written last. |

The preprocessed image itself is not written.

## Failures and warnings

- A **failure** means the image produced no outputs. Examples: an unreadable or missing input, a
  missing calibration where a length is needed, an unknown channel name, an exceeded memory
  budget, or an error in a worker. The error and a traceback excerpt are stored in `result.json`
  and `summary.csv`. The rest of the batch continues.
- A **warning** does not stop an image: missing calibration (values reported in pixels), a
  normalize step (corrected intensities not measured), and QC flags such as empty masks or
  excessive coverage.

## Run states and what they mean

The run directory has these files:

| File | When it exists |
| --- | --- |
| `run.lock` | While a run holds the directory. A lock left by a process that has exited is taken over. |
| `run.json` | While the run is in progress (`state: running`), or if it aborted (`state: aborted`, with the error). |
| `batch_manifest.json` | Only after the run finishes. Its `status` is `complete`, `completed_with_failures`, or `cancelled`. |
| `summary.csv` | Only after the run finishes. One row per image. |
| `cancel.requested` | Created by a user to request cancellation. Removed when the next run starts. |
| `items/` | One directory per image. |

Success means `batch_manifest.json` has `status: complete`. The manifest is written last, and only
for a finished run, so an interrupted, aborted, or cancelled run never leaves something that
looks like a complete one. Each image's `result.json` is also written last, after its outputs.

Exit status: `0` for `complete`, `3` for `completed_with_failures`, `130` for `cancelled`, and `1`
for a configuration error, a busy directory, or an aborted run.

## Resume and validated results

Running the same batch again reuses an image only if all of these still hold:

- its previous result has status `ok`
- its input file's SHA-256 is unchanged
- the configuration fingerprint is unchanged (preprocessing, segmentation, channel, QC settings,
  the image's declared axes, channels, and calibration, and the MicroFlow version)
- every recorded output is present with the recorded size and SHA-256

Anything else is reprocessed, and the summary states why (for example "input changed",
"configuration changed", or "output measurements.csv was modified"). The memory budget and
worker count are not part of the fingerprint, because they change scheduling, not results.

Each run first removes partial files and any stale manifest or summary, so no output from an
earlier attempt can pass as current.

## Parallelism, memory and cancellation

- **Backend.** Images run in a bounded `concurrent.futures.ProcessPoolExecutor`. The worker
  processes are separate from the calling process, so a worker crash affects only the images that
  worker was running. With one worker, images run in the calling process. Dask was considered and not added: it is a large dependency that cannot be
  locked in this environment, and a single machine's batch does not need it. A later move to a
  cluster would be a separate task.
- **Bounded work.** At most `workers` images run at once. A worker count above 8 is rejected.
- **Memory budget.** Each image's memory is estimated as elements × 40 bytes, using its header
  shape. Images whose estimate exceeds the budget fail without running. Images are started only
  while the sum of their estimates stays within the budget. This is an estimate of the working
  set, not a measurement of the process's memory, and it is not an operating-system limit.
  `peak_estimated_mb` in the manifest records the largest sum reached.
- **Cancellation.** Cancellation can come from Ctrl-C, from a `cancel.requested` file, or from a
  cancel callable passed to `run_batch`. It stops new work. Images already running finish. The
  rest are recorded as cancelled, and the manifest status is `cancelled`. The completed images
  remain valid, so the next run resumes from them. Further Ctrl-C presses while running images finish
  are ignored. To stop at once, terminate the process (for example with `kill`). That is safe: each
  image's result is published only once it is complete, the manifest is never written for an
  interrupted run, and the next run clears the stale lock and partial files and resumes.
- **Progress.** The CLI prints one line per image as it finishes:
  `[3/10] name: ok (objects 4, 1 flag(s), 2.10s)`, or `(cached)` for reused results, or
  `error: ...` for a failure.

## Limits and unverified points

- Parallelism is per machine only. There is no distributed execution.
- If a worker process dies (for example, from memory exhaustion), its image is recorded as failed
  with "worker process terminated unexpectedly". The pool is replaced and the remaining images
  continue. Which images were running at the time cannot always be determined, so each one in
  flight is recorded as failed and is retried on the next run.
- Process-based parallelism relies on the `spawn` start method on macOS and Windows. The worker
  function is importable at module level, but the parallel path has not been run in the
  environment where this was written. The parallel tests are the first to run once dependencies
  are installed.
- One run takes one batch configuration. A "dataset" here is the set of images listed in that
  configuration, and different settings need separate runs with separate output directories.
- The memory estimate is a heuristic. Very large images can exceed it.
- Corrected intensities are not measured when a normalize step is present, because normalization
  would change intensities.

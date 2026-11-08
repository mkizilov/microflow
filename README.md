# MicroFlow

MicroFlow is a Python toolkit for scientific image processing and quantitative analysis. It covers
preprocessing, segmentation, per-object measurement, quality control, batch processing with
provenance, statistical comparison of biological samples, optional U-Net segmentation, and an
optional HTTP service.

## Project status

- **Executed:** 52 tests of the service path policy, job store, provenance verifier and study report,
  run on the standard library.
- **Not yet executed:** the rest of the test suite (about 680 test functions), Ruff, mypy, the CI
  workflows, the demonstration, the HTTP service, the machine-learning path, the Docker image and every
  benchmark.
- **Blocking:** `uv.lock` has not been generated, so `uv sync --locked`, CI and the Docker build fail
  until it is.

[ROADMAP.md](ROADMAP.md) lists what is implemented, what is verified and what is open, with effort
estimates. Read it before relying on any result.

## Requirements

- Python 3.12 or 3.13. CI is configured for both; the test suite has not yet run on either.
- [uv](https://docs.astral.sh/uv/) for environment and dependency management

## Setup

```sh
uv python install 3.12
uv sync --locked
```

`uv sync` creates the project environment in `.venv/` inside the repository.

## Command-line usage

```sh
uv run microflow --help
uv run microflow version
uv run microflow make-sample-data --out-dir data/synthetic --count 3
```

`make-sample-data` writes deterministic 16-bit TIFF images (`synthetic_NN.tif`) and a
`synthetic_NN.calibration.json` sidecar per image with a 0.5 µm pixel size (`unit: "um"`).
The same seed always produces identical pixels.

## Configuration

Runs are described in YAML (see `examples/synthetic.yaml`). Validate a file before running it:

```sh
uv run microflow validate-config examples/synthetic.yaml
uv run microflow validate-config examples/synthetic.yaml --check-images
```

Unknown keys, invalid axes, duplicate names, missing calibration (when quantification requires it)
and unreadable images are reported with the file and field name.

## Quality-control reports

```sh
uv run microflow qc-report data/synthetic/synthetic_00.tif \
    --labels results/synthetic_00_labels.tif \
    --output results/synthetic_00_qc.html \
    --png-dir results/figures \
    --qc-config examples/qc.yaml \
    --pixel-size 0.5 --unit um
```

The HTML report is self-contained (figures are embedded) and lists flagged results such as empty
masks, excessive coverage, saturation, and objects outside the configured area range. Add
`--fail-on-flags` in CI to exit with status 2 when anything is flagged. See
`docs/quality-control.md`.

## Batch processing

```sh
uv run microflow batch examples/batch.yaml --workers 2
```

Every image in the configuration is preprocessed, segmented, measured and reported. Images run
in a bounded process pool, and a memory budget can limit how many run at once. Running the same
command again resumes: only images whose inputs, settings and outputs have changed are
reprocessed. The run's `batch_manifest.json` appears only when the run finishes, and its status
says whether it completed, had failures, or was cancelled. See `docs/batch.md`.

## Provenance and exported results

```sh
uv run microflow export results/synthetic-batch --output results/synthetic-export
uv run microflow compare-configs previous-batch.yaml examples/batch.yaml
uv run microflow cache-status examples/batch.yaml
```

`export` writes the batch's measurements as CSV and Parquet, with a data dictionary (units and
descriptions) and a provenance record. Each run records its software and algorithm versions,
resolved configuration, input hashes, runtime and timestamps. A run directory and an export
directory refuse to overwrite results of a different analysis unless `--overwrite` is given. See
`docs/provenance.md`.

## Plugins

Preprocessors, segmenters, feature extractors and quality checks can be provided by installed
packages that register entry points. Installing a package is the decision to trust its code, and
MicroFlow runs a plugin only when a configuration names it. See `docs/plugins.md`.

```sh
uv run microflow plugins list
uv run microflow plugins check examples/batch.yaml
```

## Segmentation training (optional PyTorch)

A small U-Net can be trained on image and mask pairs. Splitting is by original sample, so no patch
from a held-out sample reaches training. Install the extra with `uv sync --extra ml`, then:

```sh
uv run microflow ml make-dataset data/synthetic-ml
uv run microflow ml train examples/ml/synthetic.yaml
```

See `docs/machine-learning.md` for the data format, configuration, leakage rules and limits.

A trained checkpoint segments new images with the same outputs as `microflow segment`, and can
run inside batch analyses as the `ml-unet` plugin:

```sh
uv run microflow ml segment checkpoint.pt image.tif --output out/labels.tif \
  --pixel-size 0.5 --unit um
uv run microflow ml compare out/labels.tif classical/labels.tif
```

See `docs/ml-inference.md` for the model checks, device fallback, tiling, quality flags and the
preprocessing a model needs.

## Comparing groups of biological samples

`microflow stats` compares groups on exported measurements. The biological sample is the unit of
inference, so segmented objects and images are never counted as independent replicates. Every image
and sample is assigned explicitly in a design file.

```sh
uv run microflow stats example data/stats-example
uv run microflow stats check examples/stats/design.yaml
uv run microflow stats run examples/stats/design.yaml
```

See `docs/statistics.md` for the design format, the assumptions behind each estimate, the
corrections, and the outputs.

## HTTP analysis service (optional)

`microflow serve` runs an HTTP service for submitting batch analyses, tracking them, and downloading
results and quality-control reports. It needs the `service` extra and a token in the environment. It
listens on loopback only, reads and writes only inside configured directories, and runs no plugins.
The CLI works without it.

```sh
uv sync --extra service
export MICROFLOW_SERVICE_TOKEN="<at least 32 characters>"
uv run microflow serve examples/service/service.yaml
```

See `docs/service.md` for the endpoints, the security model, job states and the limits.

## Development checks

```sh
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
python3 scripts/generate_reference.py --check
```

The CI workflow in `.github/workflows/ci.yml` runs these on Python 3.12 and 3.13, with the optional
extras, the demonstration and the reference check. It has not yet run; see ROADMAP items 1 to 3.

## Dependency locking

Dependencies are declared in `pyproject.toml`. The lock file `uv.lock` is **not yet committed**. It
must be generated once, with network access, using `uv lock`, and then committed. Until then,
`uv sync --locked` fails, and so do the CI jobs and the Docker build that depend on it. After it
exists, regenerate it with `uv lock` whenever `pyproject.toml` changes. See ROADMAP item 1.

## Documentation

- [Roadmap](ROADMAP.md): what is implemented, verified and open, with effort estimates.
- [Installation](docs/installation.md): versions, extras, troubleshooting.
- [User guide](docs/user-guide.md): from images to a statistical report, step by step.
- [Reproducibility](docs/reproducibility.md): what is fixed, what is not, how to rerun.
- [Data provenance](docs/provenance-examples.md): which inputs, settings and software produced a
  result.
- [CLI reference](docs/reference/cli.md) and [HTTP API reference](docs/reference/http-api.md), both
  generated from the code.
- [Audit](docs/audit.md): what was checked for numerical accuracy, units, axes, statistics,
  segmentation and model evaluation, and what remains open.
- [Benchmarks](docs/benchmarks.md) and [Docker](docs/docker.md).

## Demonstration

`examples/demo/run_demo.py` runs the whole chain on synthetic microscopy. It generates calibrated
images with known ground truth, segments and measures them, scores the segmentation against the truth,
exports the measurements, compares two conditions by biological sample, writes one HTML report, and
verifies the run's recorded files.

```sh
uv run python examples/demo/run_demo.py --clean
```

Outputs go to `data/demo/` (git-ignored). Open `data/demo/results/report.html` when it finishes. The
demonstration has not yet been executed; see ROADMAP item 11.

## Layout

```text
src/microflow/   package: core processing, batch, ml, stats, service, provenance, reporting
tests/           pytest suite (deterministic, uses tmp_path only)
examples/        example configurations, the demonstration, and the optional service's settings
benchmarks/      pipeline timing script (no results are committed)
scripts/         generate_reference.py, which writes docs/reference from the code
docs/            user, reference, statistics, service and audit documentation
.github/         CI workflow
Dockerfile       container image for the optional service
compose.yaml     local deployment of that image, published on loopback only
```

See `docs/architecture.md` for the module map and conventions.

# MicroFlow

MicroFlow is a Python toolkit for scientific image processing and quantitative analysis. It covers
preprocessing, segmentation, per-object measurement, quality control, batch processing with
provenance, statistical comparison of biological samples, optional U-Net segmentation, and an
optional HTTP service.

## Project status

Verified on one machine (macOS 27.2, arm64) with the locked environment (`uv sync --locked --all-extras`,
uv 0.8.22):

- Python 3.12.11 and 3.13.7: the full test suite passes, 939 tests passed and 1 skipped. The skip is
  the CUDA test, and this machine has no GPU.
- Ruff lint, Ruff format and strict mypy (163 source files) pass.
- The end-to-end demonstration runs to completion: 16 synthetic images, batch, export, statistics,
  the HTML report, and a re-hash that matches all 64 recorded files.
- The wheel and source distribution build, and the wheel installs from the locked runtime
  requirements. Its CLI works without the `service` extra.
- The ML commands from the CI workflow run on CPU: dataset, splits, training and held-out evaluation.
- A single benchmark run, with three repeats, is committed in `benchmarks/results/`.

Not verified here: the GitHub Actions workflows (the repository has not been published), Linux and
Windows, the Docker image (not built), CUDA training, and accuracy on real microscopy. The demonstration's
Dice scores come from noiseless synthetic phantoms and say nothing about real images.

[ROADMAP.md](ROADMAP.md) lists what is implemented, verified and open, with effort estimates. Read it
before relying on any result.

## Requirements

- Python 3.12 or 3.13. Both pass the test suite locally; the CI workflow has not run yet.
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
extras, the demonstration and the reference check. It has not run on GitHub yet.

## Dependency locking

Dependencies are declared in `pyproject.toml`, and `uv.lock` pins the exact versions of the 63 locked
packages. Use `uv sync --locked` to install exactly that set. After changing `pyproject.toml`,
regenerate the lock with `uv lock` and commit it. `uv lock --check` confirms that the lock matches the
project.

## License

MicroFlow is released under the MIT License. See [LICENSE](LICENSE). Copyright (c) 2026 Mykyta Kizilov.
The example plugin package in `examples/plugins/` carries the same license.

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

Outputs go to `data/demo/` (git-ignored). Open `data/demo/results/report.html` when it finishes. It ran to
completion in the verification environment described above.

## Layout

```text
src/microflow/   package: core processing, batch, ml, stats, service, provenance, reporting
tests/           pytest suite (deterministic, uses tmp_path only)
examples/        example configurations, the demonstration, and the optional service's settings
benchmarks/      pipeline timing script and one committed smoke-run result (benchmarks/results/)
scripts/         reference generation and verification tools
docs/            user, reference, statistics, service and audit documentation
.github/         CI workflow
Dockerfile       container image for the optional service
compose.yaml     local deployment of that image, published on loopback only
```

See `docs/architecture.md` for the module map and conventions.

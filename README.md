# MicroFlow

MicroFlow is a Python toolkit for scientific image processing and quantitative analysis.
This repository currently provides the package scaffold, the command-line entry point, and
synthetic sample data. Analysis algorithms are not implemented yet.

## Requirements

- Python 3.12 or newer
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
uv run microflow compare-configs examples/batch.yaml examples/batch.yaml
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
uv run microflow ml segment checkpoint.pt image.tif --output out/labels.tif --pixel-size 0.5 --unit um
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

See `docs/statistics.md` for the design format, the assumptions behind each estimate, the corrections,
and the outputs.

## Development checks

```sh
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

## Dependency locking

`uv.lock` is the reproducible lock file. Regenerate it with `uv lock` after changing
dependencies in `pyproject.toml`, and commit it. CI installs with `uv sync --locked`, which
fails if the lock file is missing or out of date.

## Layout

```text
src/microflow/   package: image_io, units, sample_data, cli, and stage subpackages
tests/           pytest suite (deterministic, uses tmp_path only)
docs/            architecture notes
.github/         CI workflow
```

See `docs/architecture.md` for the module map and conventions.

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

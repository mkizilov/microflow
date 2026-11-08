# Installation

MicroFlow is a Python package. Its supported Python versions are the ones CI tests: **3.12 and 3.13**.
The project is managed with [uv](https://docs.astral.sh/uv/), which installs everything into `.venv/`
inside the repository and records exact versions in `uv.lock`.

## Requirements

- Python 3.12 or 3.13 (uv can install one for you: `uv python install 3.12`).
- uv, version 0.8.22 or later. CI pins 0.8.22.
- For the optional extras only: PyTorch (`ml`) and FastAPI and uvicorn (`service`). Neither is needed
  for the core analysis.
- A GPU is optional. The `ml` extra installs CPU-only PyTorch on Linux by default.

## Core install

```sh
git clone <repository-url> microflow
cd microflow
uv sync --locked
uv run microflow version
```

`--locked` makes uv use `uv.lock` exactly, so you get the versions that were tested. If `uv.lock` is
out of date with `pyproject.toml`, uv reports it and stops. Do not drop `--locked` for a release.

## Optional extras

| Extra | Install | What it adds |
| --- | --- | --- |
| `ml` | `uv sync --locked --extra ml` | PyTorch, for `microflow ml` training, evaluation and inference. |
| `service` | `uv sync --locked --extra service` | FastAPI and uvicorn, for `microflow serve`. |

Install several at once with `uv sync --locked --extra ml --extra service`.

## Verify the install

```sh
uv run pytest                      # the core suite; ML and service tests skip without their extras
uv run microflow make-sample-data --out-dir data/synthetic
uv run python examples/demo/run_demo.py --clean   # the end-to-end demonstration
```

The demonstration writes to `data/demo/` and takes a few minutes on a laptop. Its final message names
the HTML report to open.

## Docker (optional service)

See [docker.md](docker.md). Docker is only for the HTTP service. The command-line tools need no
container.

## Troubleshooting

- **`uv sync --locked` fails with "lockfile needs to be updated".** The dependencies in
  `pyproject.toml` and `uv.lock` disagree. Run `uv lock` with network access, review the diff, and
  commit the new lock file. Do not edit `uv.lock` by hand.
- **`microflow` is not found.** Run commands through uv (`uv run microflow ...`), or activate the
  environment with `source .venv/bin/activate`.
- **`ml` commands say PyTorch is required.** Install the extra (`uv sync --locked --extra ml`).
- **`serve` says the service extra is missing.** Install it (`uv sync --locked --extra service`).
- **Images with several channels fail for segmentation.** Choose one channel with `--channel`, or the
  batch `segmentation_channel` setting.
- **Areas are refused as "needs pixel calibration".** Give the image a calibration (`--pixel-size` and
  `--unit`, or a calibration in the file). MicroFlow never assumes a unit.

## Uninstall

Delete the repository directory. Everything MicroFlow installs lives in `.venv/` inside it, and
generated outputs live under `data/` and the output directories you chose.

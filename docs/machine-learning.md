# Supervised segmentation with a U-Net

MicroFlow can train a small U-Net to segment objects from image and mask pairs. The pipeline is
optional: it needs the `ml` extra (PyTorch), and everything else in MicroFlow works without it.

## Quick start

```sh
uv sync --extra ml                                  # installs PyTorch (CPU wheels on Linux)
uv run microflow ml make-dataset data/synthetic-ml  # deterministic synthetic images and masks
uv run microflow ml splits examples/ml/synthetic.yaml
uv run microflow ml train examples/ml/synthetic.yaml --device cpu
uv run microflow ml evaluate examples/ml/synthetic.yaml data/ml-runs/synthetic-unet/best.pt
```

`make-dataset`, `splits` and the configuration checks run without PyTorch. `train` and `evaluate`
stop with an install hint if it is missing.

## Data

A **manifest** lists each image with its mask and the original sample it came from:

```yaml
samples:
  - {sample: donor-01, image: images/donor-01_field0.tif, mask: masks/donor-01_field0.tif}
  - {sample: donor-01, image: images/donor-01_field1.tif, mask: masks/donor-01_field1.tif}
  - {sample: donor-02, image: images/donor-02_field0.tif, mask: masks/donor-02_field0.tif}
```

- Images and masks are two-dimensional Y-X planes (`.tif`, `.tiff`, `.npy`). Image and mask shapes
  must match.
- Masks hold only 0 (background) and 1 (object).
- Images must be finite. Intensities are used as stored, with no rescaling at load time.
- Paths are relative to the manifest. Each image may appear once.
- **A sample is the unit of splitting.** Every field of view, plane or tile that comes from one
  specimen must carry that specimen's `sample` name.

## Configuration

Every hyperparameter is set in YAML (`examples/ml/synthetic.yaml` is annotated). Unknown keys are
rejected with their location.

| Section | Keys | Meaning |
| --- | --- | --- |
| top level | `name`, `manifest`, `output_directory`, `seed` | Run label, data, output location, seed for splitting, shuffling, augmentation and initialisation. |
| `split` | `validation`, `test` | Fractions of samples held out. Training gets the rest. Each held-out split gets at least one sample. |
| `model` | `base_channels`, `depth` | Width of the first level (doubled per level) and number of downsamplings. |
| `patches` | `size`, `per_image` | Square training crops, and how many are drawn from each training image per epoch. `size` must be a multiple of `2**depth`. |
| `augment` | `flip`, `rotate90`, `intensity`, `noise_std` | Geometric changes apply to image and mask together. Intensity and noise change the image only. |
| `training` | `epochs`, `batch_size`, `learning_rate`, `weight_decay`, `bce_weight`, `threshold` | Loss is `bce_weight * BCE + (1 - bce_weight) * soft Dice`. `threshold` binarises probabilities for scoring. |
| top level | `device`, `num_workers` | `cpu`, `cuda` or `auto`. Worker count for data loading (0 keeps everything in the main process). |

## Leakage prevention

The pipeline is built so that no information about held-out samples reaches training.

1. **Split by original sample.** Samples are shuffled with the configured seed and assigned whole to
   `train`, `validation` or `test`. Every image of a sample lands in that sample's split, so every
   patch cut from an image stays in one split.
2. **Identical content is refused.** If the same image bytes appear under two different sample
   names, the run stops with a `LeakageError`. Those two samples would have to be split apart and
   would share pixels. If they are one specimen, give them one name.
3. **Training patches come only from training images.** The patch dataset is built from the
   training split alone.
4. **Normalisation is fitted on training pixels only.** The mean and standard deviation are computed
   before any validation or test image is used. A test checks that changing a validation or test
   image does not change the normalisation.
5. **Validation selects checkpoints; test is read once.** After each epoch the validation split
   scores the model, and the best checkpoint is kept by validation Dice. Training never reads the
   test split. `ml evaluate` is the only consumer of the test split.
6. **Evaluation checks that it is evaluating the held-out set.** A checkpoint is refused unless the
   recomputed split, architecture and normalisation match what it recorded, and unless it belongs to
   the training run recorded in the output directory. Changing a seed, a split fraction, a manifest
   or any image after training makes evaluation fail rather than score the wrong samples.
7. **An audit trail.** `split.json` records which samples and which image files (with SHA-256
   hashes) each split holds.

**What this does not cover.** The checks compare content and sample names. They cannot detect that two
differently named files are overlapping crops of one acquisition, or that the same specimen was
imaged twice under different names. Such overlaps must be grouped under one sample by the person who
builds the manifest.

## Training

Each epoch:

- training patches are drawn with a seeded generator and shuffled with a seeded `DataLoader`;
- the network is trained on the patches, and the loss is averaged over the epoch;
- every validation image is predicted whole (padded with zeros up to a multiple of `2**depth`, then
  cropped back), and its loss, Dice and IoU are computed;
- the epoch's mean validation Dice is compared with the best so far. A higher value replaces
  `best.pt`.

`last.pt` is written after the final epoch. Training always runs for the configured number of
epochs. There is no early stopping and no resume; a rerun with the same configuration starts again.

Dice and IoU come from MicroFlow's segmentation scorer, so the edge cases agree with the rest of
the package: two empty masks score 1.0, and one empty mask scores 0.0. Scores are per image, and
the reported mean is the unweighted mean over images.

## Outputs

Written to `output_directory`:

| File | Content |
| --- | --- |
| `run.json` | Training identity, versions and start time. |
| `config.json` | The resolved configuration. |
| `split.json` | The split plan and the images and hashes in each split. |
| `history.csv` | Per-epoch training loss, validation loss, Dice and IoU. |
| `best.pt`, `last.pt` | Checkpoints (see below). |
| `summary.json` | Best epoch, best validation Dice, device and versions. |
| `test_metrics.csv`, `test_summary.json` | Written by `ml evaluate`: per-image test scores and their means. |

A checkpoint holds the weights and the metadata needed to use them safely: the architecture,
normalisation statistics, threshold, split, training identity, configuration, epoch, validation
scores, and the MicroFlow and PyTorch versions. Checkpoints are written atomically, and they are
loaded with `weights_only=True`, so opening one cannot run code.

## Reproducibility

- Splitting, shuffling, augmentation and weight initialisation all derive from `seed`. Each patch's
  augmentation uses its own generator seeded by (seed, epoch, index), so the result does not depend
  on worker count or loading order.
- The training identity hashes the settings and every input file's content, but not the output
  location, worker count, or device. Results in a directory belonging to a different identity are
  not overwritten unless `--overwrite` is given.
- **On one machine with the same PyTorch and thread count**, a repeated run produces the same
  history. A test checks this on CPU to a tolerance of 1e-6.
- Bit-identical results across machines, PyTorch versions, or GPUs are **not** promised. The
  library versions are recorded so that differences can be traced.

## CPU and GPU

- CPU is the default and needs no configuration. CI runs the ML tests on CPU.
- `device: cuda` requires a CUDA device and fails with a clear message otherwise. `device: auto` uses
  CUDA when present and CPU otherwise. The command-line `--device` option overrides the setting.
- On Linux, `pyproject.toml` pins PyTorch to its CPU-only index to keep CI small. To use CUDA
  wheels, remove the `torch` entry under `[tool.uv.sources]` and re-lock.
- The GPU test is marked `gpu` and is skipped when CUDA is unavailable. **It has not been run in
  this environment.**

## Tests

- `tests/test_ml_config.py`, `tests/test_ml_splitting.py`, `tests/test_ml_preparation.py`,
  `tests/test_ml_augment_metrics.py`: configuration, manifest and split rules, leakage checks,
  preparation order, augmentation alignment and metric definitions. They need no PyTorch.
- `tests/test_ml_torch.py` (marker `ml`): network shapes, losses, checkpoint round-trip and safety
  checks, patch determinism, the training-data spy (gradients only from training samples, scores only
  from validation samples), split audit, reproducibility, overwrite protection, evaluation scope and
  refusals, and the optional GPU test. Skipped without PyTorch.
- `tests/test_cli_ml.py`: command-line behaviour, including that `make-dataset` and `splits` run with
  PyTorch blocked, and that `train` gives an install hint without it.
- The `ml` CI job runs `pytest -m ml`, then generates a dataset, shows its splits, trains, and
  evaluates from the command line.

## Limitations

- Two-dimensional, single-channel images only. Multi-channel and volumetric data are not supported.
- All images are held in memory. Large datasets need a lazy loader that is not implemented.
- Single U-Net architecture. No pretrained weights, no deep supervision, no test-time augmentation.
- Thresholds are fixed by configuration. There is no threshold search, class balancing, or
  calibration.
- Checkpoints depend on the PyTorch version that wrote them, and loading requires a compatible
  version.
- Synthetic data is a smoke test, not evidence of accuracy on real images.

## Verification status

**Not run in the environment where this was written**, which had no network, no PyTorch, and no
NumPy. Verified there: static checks (undefined names, import resolution, call signatures, unused
code, import cycles), a check that the torch-free modules reach no PyTorch-importing module through
module-level imports (with a seeded violation caught), and byte-compilation. Everything else,
including all tests, Ruff, mypy, the CI workflow, and the GPU path, must be run on a machine with
the dependencies installed.

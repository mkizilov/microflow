# ML inference: trained models in MicroFlow analyses

A checkpoint written by `microflow ml train` can segment new images. It produces the same outputs as
classical segmentation, so the rest of the pipeline (measurement, quality control, export) works on
either. This page covers how models are validated, how large images are processed, how masks are
postprocessed, what the quality report means, and what the model can and cannot be trusted for.

## Commands

```sh
uv run microflow ml fingerprint model.pt                    # SHA-256 to pin in a configuration
uv run microflow ml inspect model.pt                        # validate and describe a checkpoint
uv run microflow ml segment model.pt image.tif --output out/labels.tif \
    --pixel-size 0.5 --unit um --reference truth.tif        # labels, sidecars and report
uv run microflow segment image.tif --output classical/labels.tif --pixel-size 0.5 --unit um
uv run microflow ml compare out/labels.tif classical/labels.tif --output comparison.json
```

`fingerprint` and `compare` run without PyTorch. `inspect` and `segment` need the `ml` extra.

## Outputs

`ml segment` writes the same files as `microflow segment`, so downstream commands accept either:

| File | Content |
| --- | --- |
| `labels.tif` | int32 label image with the input's axes. 0 is background; objects are 8-connected and numbered 1..N per plane in raster order. |
| `labels.segmentation.json` | The classical sidecar keys: `source`, `axes`, `object_counts`, `thresholds`, `settings`, `scores`. `settings` records the method `ml-unet`, the checkpoint digest, the device used, the tiling, the threshold and the minimum area. |
| `labels.ml.json` | The ML report: checkpoint identity and versions, architecture, normalisation, tiling (including whether the overlap is sufficient), device and fallback, and a quality record for each plane. |

## Loading and version validation

A checkpoint is opened only after its declarations are checked (`ml/model_spec.py`). The checks are
deliberately strict about anything that changes predictions:

- **Format and schema.** The file must be a MicroFlow U-Net checkpoint of the current schema (2).
  Schema 1 checkpoints, written before training recorded its calibrations, are refused.
- **Required declarations.** Architecture, normalisation, threshold, training identity, MicroFlow
  version, training calibrations and epoch must be present and well formed.
- **Architecture limits.** One input channel, depth 1 to 5, base width 1 to 128, and no unknown keys.
- **Normalisation.** Mean and standard deviation must be finite, and the standard deviation positive.
- **Threshold.** Strictly between 0 and 1.
- **MicroFlow version.** The minor version must match (`0.1.x` loads `0.1.y`). A patch difference
  is accepted; a minor or major difference is refused, because the processing around the network
  changes with it.
- **Weights.** They must fit the declared architecture exactly. A mismatch is refused.
- **PyTorch version.** A different minor version only produces a warning, because the same weights
  normally give the same numbers. The warning is printed by `ml inspect` and `ml segment`.

A segmenter plugin configuration must also name the checkpoint's SHA-256, and the run refuses a
different file. The digest is part of the analysis identity, so changing the model changes which
cached results are reused.

## Devices and automatic CPU fallback

- `device: auto` (the default) uses CUDA when it is available, and the CPU otherwise.
- `device: cpu` always uses the CPU.
- `device: cuda` fails with a clear message when no CUDA device is present. It never silently changes device.
- With `fallback_to_cpu` (the default), a recoverable CUDA failure during a run (such as running out
  of memory) moves the rest of that run to the CPU. The affected planes list `device_fallback` in
  their quality flags, and the `.ml.json` report records `fallback_used`. Other errors are not masked.

The accelerated model is a separate copy of the network, so the CPU model is always available. Both
paths run with gradients off and the network in evaluation mode. On one machine, the same input gives
the same output.

## Tiled inference for large images

The network is applied to windows, not to whole planes, so memory is bounded by the tile size.

1. The plane is normalised with the checkpoint's statistics, then padded with zeros at the bottom and
   right to a multiple of `2**depth`. Zeros are the normalised mean.
2. The padded plane is cut into windows of `tile_size`. Neighbouring windows overlap by `overlap`
   pixels. Each pixel is taken from exactly one window, the one in which it is a core pixel, so nothing
   is averaged and each pixel has a single probability.
3. The output is cropped back to the input's size.

**Overlap.** The default overlap is the network's receptive radius, `8 * 2**depth - 6` pixels: an
upper bound on how far an output pixel can depend on the input. For depth 2 that is 26 pixels. When
the overlap covers the receptive field, every core pixel sees the same context as in whole-plane
prediction, so tiled and whole-plane results are equal. A test checks this on a random network with a
tolerance of 1e-5, and checks that a zero overlap visibly differs.

A smaller overlap is allowed, but the plane is flagged `overlap_below_receptive_field` and the seams
between tiles may show. `tile_size` must be a multiple of `2**depth`, and the overlap must be less
than half the tile size.

The receptive radius bound is measured against the network's actual gradient support in the test
suite. It is a bound, not an exact value, so using the default is the safe choice.

## Thresholding

One rule applies everywhere: a pixel is foreground when its probability is greater than or equal to
the threshold. Training validation, inference and the classical-style postprocessing all call the
same function. The default threshold is the one recorded in the checkpoint, and `--threshold`
overrides it. The value used is written to the sidecar and the report.

## Labelled-mask postprocessing

After thresholding, the mask goes through the same steps as classical segmentation: 8-connected
labelling, removal of objects below the minimum area (`--min-area`, in `--area-unit`, which needs a
calibration unless it is 0), optional removal of objects touching the edge (`--exclude-border`), and
raster-order renumbering. Because the rules are shared, an ML label image and a classical one can be
compared directly.

## Quality report

Each plane gets a quality record:

| Field | Meaning |
| --- | --- |
| `foreground_fraction` | Share of pixels labelled as object. |
| `object_count`, `border_object_count` | Objects after postprocessing, and how many touch the edge. |
| `mean_object_probability` | Mean probability over object pixels. `null` when there are none. |
| `uncertain_fraction` | Share of pixels within 0.1 of the threshold. |

Flags are warnings, not failures:

| Flag | Raised when |
| --- | --- |
| `empty_prediction` | No objects were found. |
| `excessive_coverage` | More than 90 % of pixels are foreground. |
| `high_uncertainty` | More than 20 % of pixels are within 0.1 of the threshold. |
| `overlap_below_receptive_field` | The plane is tiled and the overlap is smaller than the receptive radius. |
| `calibration_missing` | The model was trained on calibrated images, and this one has no calibration. |
| `calibration_mismatch` | The pixel size or unit differs from the training images. |
| `device_fallback` | A CUDA failure moved the run to the CPU. |

Flags do not say whether a mask is correct. Only a comparison with reference labels does that.

## Comparing ML and classical masks

`ml compare` takes two label images with the same axes and shape. It reports:

- Dice and IoU over all pixels;
- the number of objects in each image;
- the number of **matched** objects. Pairs are formed in order of decreasing IoU, a pair is accepted
  when its IoU is at least `--match-iou` (default 0.5), and no object is used twice. Ties are broken by
  label number.

Label numbers restart in each plane, so stacks are compared plane by plane. The overall Dice and IoU
are pixel-level and cover the whole stack.

`ml segment --reference` reports Dice and IoU against a reference in the same way as `microflow segment`.

## Required image preprocessing

These are requirements on the input, not things MicroFlow does for you.

- **One 2-D plane per model input.** Multi-channel images need `--channel`. Stacks are processed plane
  by plane. The model has one input channel.
- **Finite values.** Images with NaN or infinite values are refused. Mask or fill them first.
- **Normalisation is automatic and fixed.** The checkpoint's mean and standard deviation (fitted on the
  training images) are applied to every input. Do not normalise each image separately, and do not
  rescale intensities first.
- **Same preprocessing as training.** Background subtraction, flat-field correction, filtering and
  other classical steps are **not** applied automatically. If the model was trained on images that
  were processed in a certain way, the same processing must be applied before `ml segment`, for example
  with `microflow preprocess`. The batch pipeline applies its configured preprocessing before the
  segmenter, so the training steps must match there too.
- **Same pixel size.** The checkpoint records the pixel sizes of its training images. A different
  size, or a missing calibration when training had one, is flagged. MicroFlow does not resample.
- **Same modality and intensity scale.** Nothing checks this. A model trained on one stain, microscope
  or exposure range is not expected to work on another.
- **Dtype.** Integer and floating-point images are converted to float64. Integers are represented
  exactly only up to 2**53, which covers every common microscope bit depth.

## Model limitations

- **Generalisation is unmeasured.** Validation scores on the training data are the only evidence the
  pipeline produces. The CI dataset is synthetic. Its scores say the pipeline runs, not that a model
  will work on real images. Evaluate a model on held-out real data from the same protocol before
  relying on it.
- **Two dimensions only.** There is no 3-D or time-series model, and each plane is segmented on its own.
  Objects that span planes are split at plane boundaries.
- **One class.** The model separates object from background. It does not classify objects.
- **Probabilities are not calibrated.** A probability of 0.8 does not mean an 80 % chance of being
  correct. Thresholds were chosen on the validation split, and a different population may need a
  different threshold.
- **Small receptive field.** A depth-2 model sees about 26 pixels in each direction. Large objects are
  segmented from local appearance and can be split or fragmented.
- **Fixed normalisation.** Normalisation uses training statistics. An input with a very different
  intensity distribution is outside the range the model saw, and the quality report will not necessarily
  say so.
- **Tiles and seams.** Overlap below the receptive radius creates seams. The default avoids them.
- **Memory and time.** Memory is set by the tile size. CPU inference of a large plane can take a long
  time, and no benchmark has been run.
- **Checkpoints are version-bound.** A checkpoint works only with the MicroFlow minor version that wrote
  it. Retrain after upgrading.

## Plugin use in batch analyses

The `ml-unet` segmenter is registered as a plugin, so a batch configuration can use it:

```yaml
segmentation:
  method: plugin
  plugin:
    name: ml-unet
    options:
      checkpoint: models/donor-unet/best.pt
      checkpoint_sha256: <output of microflow ml fingerprint>
      device: auto
      tile_size: 256
  min_object_area: {value: 4, unit: um}
```

The plugin returns labels. The batch pipeline then applies the minimum-area and border rules, and the
usual measurement, quality control and export steps follow. The checkpoint path is used as written, so
relative paths resolve against the working directory.

## Verification status

Written in an environment without PyTorch or NumPy. **Not run:** every test, Ruff, mypy, and the batch
and plugin integration. Verified statically there: name resolution, call signatures, unused code, and
import isolation, meaning that the torch-free modules (including the plugin module, which the batch
pipeline imports) do not import PyTorch at load time. The tests in `tests/test_ml_tiling.py`,
`tests/test_ml_fallback.py`, `tests/test_ml_model_spec.py`, `tests/test_ml_pipeline.py`,
`tests/test_ml_compare.py` and `tests/test_cli_ml_inference.py` (the parts that need no PyTorch) are the
first to run on a machine with NumPy and SciPy. The `tests/test_ml_inference.py` and `tests/test_ml_plugin.py`
tests need PyTorch. The CUDA path has not been run on any machine.

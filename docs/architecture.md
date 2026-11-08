# MicroFlow architecture

MicroFlow is organised as a `src/` package with one module or subpackage per stage of an
image-analysis workflow:

| Module | Responsibility | Status |
| --- | --- | --- |
| `microflow.units` | `Calibration` model: pixel size plus a required length unit | implemented |
| `microflow.image_io` | TIFF read/write, header inspection, and calibration JSON sidecars | implemented |
| `microflow.config` | Pydantic pipeline configuration models and YAML loading | implemented |
| `microflow.loading` | TIFF, OME-TIFF and `.npy` loading with dtype, axes, calibration and channel metadata preserved | implemented |
| `microflow.preprocessing` | Flat-field correction, background, Gaussian and median filters, artifact masking, percentile normalisation; steps in YAML | implemented |
| `microflow.reporting.preview` | Before/after PNG previews (display-only stretch) | implemented |
| `microflow.segmentation` | Otsu, fixed and adaptive thresholds; opening, closing; labelling; distance-transform watershed; small-object and border removal; Dice/IoU | implemented |
| `microflow.quantification` | Per-object geometry and raw/corrected intensity statistics, with physical units when calibrated; CSV table | implemented |
| `microflow.dataset` | Typed per-image metadata checked against the manifest | implemented |
| `microflow.sample_data` | Deterministic synthetic images for demos and tests | implemented |
| `microflow.cli` | Typer entry point (`microflow`) | implemented |
| `microflow.preprocessing` | Denoising, background correction, normalisation | placeholder |
| `microflow.segmentation` | Object detection and masks | placeholder |
| `microflow.quantification` | Calibrated measurements on segmented objects | placeholder |
| `microflow.pipelines` | Composition of the stages above | placeholder |
| `microflow.reporting` | Result tables and reports | placeholder |

## Configuration

A run is described by a YAML file with the sections `name`, `dataset`, `processing`,
`segmentation`, `quantification` and `output`. Each section is validated by a Pydantic model that
forbids unknown keys, so misspelled options are reported rather than ignored. Errors name the
file and the field (for example `dataset.images.0.axes`). See `examples/synthetic.yaml`.

- Each image in `dataset.images` has its own `axes` (default `YX`, letters from `TCZYX`),
  optional `channels`, and optional `calibration`. Images need not share shape, dimensionality or
  channel count; `microflow.dataset.inspect_dataset` checks each image against its declared axes
  and channel names.
- `quantification.require_calibration` (default `true`) makes a missing calibration on any image
  a configuration error.
- Relative paths are resolved against the directory of the configuration file.

`microflow validate-config <file>` checks the schema; add `--check-images` to also read each
image header.

## Image loading

`load_image(path, axes=..., calibration=..., series=..., lazy=...)` returns a `LoadedImage` with
the native array, its axis labels, calibration and channel names.

- TIFF axes come from the file. Passing `axes` asserts they match; a mismatch is an error.
- OME-TIFF calibration (`PhysicalSizeX/Y` with units) and channel names are read from OME-XML.
  Missing units default to micrometres, as the OME specification does. Anisotropic pixels and
  multi-file OME-TIFFs are rejected.
- Plain TIFF calibration comes from the resolution tags, only when the resolution unit is inch or
  centimetre. Other files, including ImageJ files that store spacing in their description, have no
  calibration.
- `.npy` files carry no axis labels, so `axes` is required. Pickled arrays are refused.
- Dtypes are kept. Only integer and floating-point data are accepted. RGB or other sample axes
  are rejected rather than split.
- `lazy=True` memory-maps uncompressed TIFF and `.npy` data. Compressed TIFF is read into memory,
  and `storage` reports which path was used.

## Preprocessing

`microflow preprocess INPUT --output OUT.tif [--steps steps.yaml] [--flatfield flat.tif]
[--pixel-size N --unit U] [--preview preview.png]` runs the stages below and writes float64 TIFF
plus `OUT.provenance.json`. The sidecar lists every stage with its parameters and derived values
(pixel radii, normalisation scale and offset, pixels masked, non-finite pixels found). See
`examples/preprocess_steps.yaml`.

- Stages run in the order listed. Nothing runs unless listed; flat-field correction runs first
  when a reference is given.
- Output is always float64. Integers up to 2**53 convert exactly; larger integers are refused.
- NaN and infinity in the input become invalid pixels, are counted, and stay invalid. Filters
  ignore invalid pixels; normalisation excludes them.
- Lengths are in physical units and need calibration; they convert through the calibration unit.
- Flat-field: divides by `flat / mean(flat)`, so mean intensity is preserved. Reference pixels
  that are zero, negative, or non-finite are refused.
- `normalize` is affine (`output = scale * input + offset`) and recorded. It is opt-in and is
  meant for display and segmentation. Measurements should use the unnormalised image.
- `mask_artifacts` compares current values, so place it before arithmetic steps.
- Median filtering uses a NaN-aware path (per-pixel) when invalid pixels are present, which is
  slower on large images.

## Segmentation

`microflow segment INPUT --output labels.tif [--config seg.yaml] [--reference truth.tif]
[--pixel-size N --unit U]` writes an int32 label image (0 is background) and
`labels.segmentation.json` with the settings, the threshold used on each plane, object counts,
and Dice and IoU when a reference is given. See `examples/segment.yaml`.

- Settings are validated by `SegmentationConfig`. Unknown keys are rejected.
- Each Y-X plane is segmented independently. Labels are numbered 1..N per plane, in raster order
  of each object's first pixel, so they are stable across runs.
- Order: threshold, opening, closing, connected components (8-connected), optional watershed,
  minimum-area removal (an object of exactly the minimum size is kept), optional border removal.
- Lengths and areas need calibration. Adaptive block size is rounded to an odd pixel count of at
  least 3.
- NaN pixels are background and do not affect thresholds.
- Dice = 2|A∩B|/(|A|+|B|); IoU = |A∩B|/|A∪B|, on foreground pixels. Two empty masks score 1.0.
- The watershed uses distance-transform peaks at least `watershed_min_distance` apart as markers.

## Quantification

`microflow measure LABELS --raw RAW --output table.csv` writes one row per object. Geometry
(area, perimeter, centroid, bounding box, eccentricity, equivalent diameter) is reported once per
object; intensity statistics are reported per source (`raw` or `corrected`) and channel. Lengths
are in the calibration unit when the image is calibrated, otherwise in pixels, and column names
state which. Every definition and assumption is in `docs/measurements.md`.

## Conventions

- **Units:** every quantitative value is expressed in the calibration's unit. `Length` and `Area`
  (`microflow.units`) carry explicit units and convert with `.to(unit)`. Calibration sidecars
  (`<image>.calibration.json`) are written by the sample generator; the unit has no default.
- **Determinism:** synthetic data and any randomised step take an explicit seed.
- **Dependencies:** managed by `uv`; `uv.lock` pins every transitive dependency.

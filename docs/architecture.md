# MicroFlow architecture

MicroFlow is organised as a `src/` package with one module or subpackage per stage of an
image-analysis workflow:

| Module | Responsibility | Status |
| --- | --- | --- |
| `microflow.units` | `Calibration` model: pixel size plus a required length unit | implemented |
| `microflow.image_io` | TIFF read/write, header inspection, and calibration JSON sidecars | implemented |
| `microflow.config` | Pydantic pipeline configuration models and YAML loading | implemented |
| `microflow.loading` | TIFF, OME-TIFF and `.npy` loading with dtype, axes, calibration and channel metadata preserved | implemented |
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

## Conventions

- **Units:** every quantitative value is expressed in the calibration's unit. `Length` and `Area`
  (`microflow.units`) carry explicit units and convert with `.to(unit)`. Calibration sidecars
  (`<image>.calibration.json`) are written by the sample generator; the unit has no default.
- **Determinism:** synthetic data and any randomised step take an explicit seed.
- **Dependencies:** managed by `uv`; `uv.lock` pins every transitive dependency.

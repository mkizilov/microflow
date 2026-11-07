# MicroFlow architecture

MicroFlow is organised as a `src/` package with one module or subpackage per stage of an
image-analysis workflow:

| Module | Responsibility | Status |
| --- | --- | --- |
| `microflow.units` | `Calibration` model: pixel size plus a required length unit | implemented |
| `microflow.image_io` | TIFF read/write and calibration JSON sidecars | implemented |
| `microflow.sample_data` | Deterministic synthetic images for demos and tests | implemented |
| `microflow.cli` | Typer entry point (`microflow`) | implemented |
| `microflow.preprocessing` | Denoising, background correction, normalisation | placeholder |
| `microflow.segmentation` | Object detection and masks | placeholder |
| `microflow.quantification` | Calibrated measurements on segmented objects | placeholder |
| `microflow.pipelines` | Composition of the stages above | placeholder |
| `microflow.reporting` | Result tables and reports | placeholder |

## Conventions

- **Units:** every quantitative value is expressed in the calibration's unit. Calibration
  sidecars (`<image>.calibration.json`) are mandatory for calibrated analysis; the unit has no
  default.
- **Determinism:** synthetic data and any randomised step take an explicit seed.
- **Dependencies:** managed by `uv`; `uv.lock` pins every transitive dependency.

# Quality-control reports

`microflow qc-report RAW --labels LABELS --output report.html [--preprocessed PRE]
[--qc-config qc.yaml] [--png-dir figures/] [--fail-on-flags] [--pixel-size N --unit U]`
checks a segmentation and writes one self-contained HTML file. The report contains the flags, a
per-plane summary table, the figures, and the settings used. See `examples/qc.yaml`.

Reporting lives in `microflow.reporting`. It only reads results: images, the label image and
settings. The numerical packages never import it, so the reports can change without affecting
analysis.

## Figures

Each figure is a PNG rendered headless with matplotlib's object-oriented API on the Agg canvas.
`pyplot` is never imported, so no display is needed and batch and CI runs produce the same output.
A test checks that importing the reporting modules does not load `pyplot`.

| Figure | Contents |
| --- | --- |
| Original image | The raw plane, with a display-only stretch between the 0.5th and 99.5th percentiles. |
| Preprocessed image | The same for the preprocessed plane. Present only when `--preprocessed` is given. |
| Segmentation overlay | The raw plane with object boundaries (inner boundary pixels) in red. |
| Object labels | Each object coloured and labelled with its number at its centroid. |
| Intensity distribution | Normalised histograms of intensities inside objects and in the background, for the raw image and, if given, the preprocessed image. Non-finite values are excluded. |
| Object area distribution | Histogram of object areas in the report's area unit, with any configured minimum and maximum drawn as dashed lines. |

The figures show one plane: the plane with the most objects (the first one on a tie). The
summary table covers every plane. The stretch used for display never changes the data.

The HTML embeds the figures as base64 PNG data URIs and uses inline styles only. It has no
external resources and no scripts, so it can be opened offline or attached to a CI run. All
text fields are HTML-escaped.

## Flags

Each flag names its plane and gives the measured value and the limit.

| Code | Raised when |
| --- | --- |
| `empty_mask` | The plane has no segmented objects. |
| `excessive_coverage` | The foreground covers more than `max_foreground_fraction` of the plane. A value equal to the limit is not flagged. |
| `saturation` | More than `max_saturated_fraction` of raw pixels are at or above the saturation level. For integer images the level is the type's maximum (for example 65535 for `uint16`). Float images are not checked unless `saturation_level` is set. |
| `invalid_pixels` | More than `max_invalid_fraction` of pixels in the checked image are NaN or infinite. The checked image is the preprocessed one if given, otherwise the raw one. |
| `object_area_below` | One or more objects are smaller than `min_object_area`. The message lists their labels. |
| `object_area_above` | One or more objects are larger than `max_object_area`. The message lists their labels. |

Area limits convert through the image calibration. Without calibration, configuring them is an
error. Flags are advisory: they describe the data and do not change any result.

`--fail-on-flags` makes the command exit with status 2 when any flag is raised, after the report
is written. This is for CI gates. Status 1 means the inputs could not be read or are inconsistent.

## Summary statistics

For each plane the table reports:

- the object count
- the foreground fraction (foreground pixels over all pixels in the plane)
- the minimum, median and maximum object area, in the report's area unit (`um2`, or `px2` without
  calibration)
- the share of saturated pixels, or `n/a` when no saturation level applies
- the share of invalid (NaN or infinite) pixels

Object areas are pixel counts times the pixel area. This is the same definition as the `area`
column in `microflow measure`.

## Assumptions and limits

- Inputs must share one grid: the label image and any preprocessed image must have the same axes
  and shape as the raw image. Otherwise the command refuses them.
- Figures come from one plane, but flags and summaries cover all planes.
- Histograms use density normalisation, so raw and preprocessed distributions can be compared by
  shape, not by counts.
- The report describes the segmentation as given. It does not re-run or validate it.
- Matplotlib is a required dependency. Figure bytes are expected to be identical for identical
  inputs. A test asserts this, and it passes on macOS arm64 with the locked Matplotlib. Byte equality across
  platforms is not claimed.

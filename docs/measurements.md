# Quantitative measurements

`microflow measure LABELS --raw RAW --output table.csv [--corrected CORRECTED]
[--pixel-size N --unit U] [--require-calibration]` measures every labelled object and writes one
CSV row per object. The measurement code lives in `microflow.quantification`; its typed form is
`MeasurementTable`, whose rows are `ObjectRecord`s holding one `GeometryRecord` and one
`IntensityRecord` per source and channel.

## Units

- With a pixel calibration (the file's own, or `--pixel-size` and `--unit`), lengths are in the
  calibration unit and areas in its square. Column names carry the unit, for example
  `area_um2` or `perimeter_um`.
- Without one, lengths are in pixels (`px`, `px2`) and the table says so. Nothing is silently
  assumed to be micrometres. `--require-calibration` turns this into an error.
- Eccentricity is dimensionless and has no unit suffix.
- Intensities are in the units of the input image. The measurements do not convert them.

## Coordinates

Coordinates are measured from the top-left corner of the image. Pixel `(row, col)` covers
`[row, row+1) x [col, col+1)`, so its centre is at `(row+0.5, col+0.5)`.

## Geometry (one set of columns per object)

| Column | Definition |
| --- | --- |
| `label` | The object's label value in the label image. Background (0) is never measured. |
| `area` | Number of pixels times the pixel area. |
| `perimeter` | Length of the 0.5-level iso-contour of the object mask, computed on the object's bounding box padded by one pixel. The contour runs midway between object and background pixel centres. Holes contribute their own contours. |
| `centroid_row`, `centroid_col` | Mean pixel position, plus half a pixel, in length units. |
| `bbox_min_row`, `bbox_min_col`, `bbox_max_row`, `bbox_max_col` | Bounding box edges, in length units. The max values are exclusive edges, so the width is `bbox_max_col - bbox_min_col`. |
| `eccentricity` | `sqrt(1 - (minor/major)^2)` from the second central moments of the object's pixels. 0 for a disc or square, approaching 1 for a line. |
| `equivalent_diameter` | Diameter of the disc with the same area: `sqrt(4 * area / pi)`. |

Analytic checks in the test suite:

- A square of `s x s` pixels has perimeter `4(s-1) + 2*sqrt(2)` pixels. A single pixel has
  perimeter `2*sqrt(2)`.
- A disc of radius `r` has area close to `pi r^2`, perimeter close to `2 pi r`, and equivalent
  diameter close to `2r`. Discretisation makes these approximate. The tests allow 1-3%.
- An ellipse with semi-axes `a` and `b` has eccentricity `sqrt(1 - (b/a)^2)`.

Perimeter is a discrete estimate. It is consistent across objects and units, but it is not an
exact boundary length for curved shapes.

## Intensity (one set of columns per source, channel and statistic)

Columns are named `<source>_<channel>_<statistic>`:

- `source` is `raw` (the input image) or `corrected` (a separately supplied image on the same
  grid). The two are never combined.
- `channel` is the channel name from the image (OME-TIFF), or `c0`, `c1`, ... when none is given.

| Statistic | Definition |
| --- | --- |
| `valid_pixels` | Number of finite pixels in the object. NaN and infinite pixels are excluded. |
| `mean` | Mean over valid pixels. |
| `median` | Median over valid pixels. |
| `std` | Population standard deviation over valid pixels (divisor n, not n-1). |
| `min`, `max` | Extremes over valid pixels. |
| `integrated_intensity_<unit>2` | Sum over valid pixels times the area of one pixel. Its unit is the intensity unit times the length unit squared. Without calibration it is the plain sum. |

An object with no valid pixels in a channel reports `valid_pixels = 0` and NaN for the other
statistics. Zero is a real intensity value and is included.

## Raw and corrected intensities

- `raw` is measured from the image as acquired.
- `corrected` is an image on the same grid, for example from `microflow preprocess` with
  flat-field correction and background subtraction. Its `.provenance.json` sidecar is checked.
  If it lists a `normalize` stage the command refuses to measure it, because normalisation
  rescales intensities and would change every intensity result. If there is no sidecar, the
  command warns that the image cannot be verified.
- Geometry is measured once, from the labels, and is not repeated for each source.

## Assumptions and limits

- The label image must be one Y-X plane with an integer dtype, and 0 is background. Objects are
  measured independently of each other; overlapping objects cannot occur in a label image.
- The raw and corrected images must have the same axes and shape as each other. Measurement
  supports axes `YX` or one channel axis `C`. Multi-plane (T or Z) images must be measured one
  plane at a time.
- Measurements use each object's own pixels only. No neighbourhood or background is subtracted
  inside `measure`; apply background correction beforehand, through `preprocess`.
- The `quantification.measures` setting in a full MicroFlow configuration is validated but not yet
  used by `measure`, which always writes every column above.

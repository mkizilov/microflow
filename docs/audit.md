# Scientific audit

This audit reviews MicroFlow's numerical accuracy, physical units, axis handling, statistical
assumptions, segmentation quality and model evaluation. It records what was checked, how, what was
changed, and what remains open. A reviewer should be able to repeat each check.

**Important:** the authoring environment for this release had no scientific Python stack and no network.
Several checks below were therefore done by reading the code, by hand calculation, or on the standard
library only. Each entry says which kind of check was made. Items marked *open* need a run with the
full environment before this audit can be called complete.

## Summary of changes made in this audit

| Area | Finding | Change |
| --- | --- | --- |
| Units | A zero minimum area (meaning "no filter") was refused on an uncalibrated image, forcing users to invent a calibration. | Zero converts to zero pixels in any unit, so it needs no calibration. A positive quantity still needs one. Tests added. |
| ML tiling | The receptive-field derivation in the docstring did not match the value it justified. | Derivation rewritten: deepest-path radii 9.5, 24.5 and 54.5 pixels for depths 1 to 3. The bound `8 * 2**depth - 6` is kept and is checked against those values in a test. |
| Thresholding | The classical path is strict (`>`), the ML path inclusive (`>=`). Not documented. | Documented in `docs/architecture.md`. |
| Radii | Morphological radii and filter windows round to whole pixels. Not documented. | Documented, with the size of the error, under "Units and calibration" below. |
| Real-batch test | An HTTP test relied on uncalibrated images with default settings, which the unit rule refuses. | Test images calibrated. |
| Service | A worker count above one would fork a process pool from the threaded service. | Single worker per batch, enforced in code. |
| Store and report | `X \| None` annotations fail on Python 3.9, which hid the code from the only interpreter available. | `from __future__ import annotations` added. No behaviour change on 3.12. |

## Units and calibration

**Checked by reading:** `Length.to` and `Area.to` multiply by the ratio of nanometres per unit, and
`Area` squares it. `to_pixels` divides by the pixel size and `area_to_pixels` by its square. Both are
correct. The measurement scale applies linearly to lengths and quadratically to areas
(`quantification/features.py`). Intensity integration multiplies the sum of pixel values by the
squared pixel size, which is the correct integral when intensities are per pixel. QC area limits
convert through the same calibration as the measurements.

**Checked by hand:** `2 µm` at `0.5 µm/px` is `4 px`, and `4 µm²` is `16 px²`.

**Open:** no test has run that compares a measured physical quantity against an analytic value on
calibrated images. The shape tests in `tests/test_quantification.py` assert such values, but they
have not been executed here.

**Unit policy, by design:** no quantity is assumed to be in pixels or micrometres. A positive
physical quantity on an uncalibrated image is refused. This is pinned by tests and is kept.

**Whole-pixel rounding.** Morphological radii (`open_mask`, `close_mask`) and the watershed minimum
distance are converted to pixels and then rounded to the nearest whole pixel, and filter windows
(`median`, background) use `2 * round(r) + 1` pixels. The error in the radius is at most half a pixel.
For a physical radius `R` at pixel size `p`, the pixel radius is `R / p`, so the rounding error in
physical units is at most `p / 2`, and its relative size is at most `p / (2R)`. At 0.5 µm per pixel:
a 2 µm radius (4 px) is exact; a 1.2 µm radius (2.4 px) rounds to 2 px, an error of 0.4 px (0.2 µm,
about 17 % of the radius). The bound for that radius is 0.25 µm and 21 %. The error is largest for
small structures, where it can dominate, so an opening or closing radius of one or two pixels should be
checked against the expected object scale. The minimum-area filter is not rounded:
it compares the exact pixel area with the object's pixel count.

## Axis handling

**Checked by reading:** `segment_image`, `segment_stack` and `compare_label_stacks` move the Y and X
axes to the end, process each plane, and move them back, so arbitrary leading axes (T, C, Z) are
handled. The `write_image` axis record was fixed in an earlier turn so that multi-channel label stacks
keep their axes. The statistics data reader uses names, never positions.

**Open:** a round trip of a non-trivial axis order (for example `CZYX` with a transposed input) through
the batch engine has not been executed here.

## Numerical accuracy

**Perimeter.** The 0.5-level iso-contour of a square of `s` by `s` pixels has length
`4(s-1) + 2√2`. The documented formula matches this, and the expected values were checked by
arithmetic: 2.828 for one pixel, 6.828 for a 2 by 2 square, 10.828 for 3 by 3.

**Hedges' g.** For groups `[1, 2]` and `[3, 4]`, g equals `-(8/7)√2 = -1.6162`, which the test
asserts. The bias correction factor is `1 - 3/(4(n₁+n₂) - 9)`, checked against the formula.

**Benjamini-Hochberg and Holm.** The worked examples in `tests/test_stats_corrections.py` were
computed by hand from the definitions and match the implementation's conventions (monotone smoothing
from the largest p-value downwards for BH, running maximum for Holm).

**Permutation p-value.** The exact two-sided value for complete separation of three against three is
2/20. The Monte Carlo estimator uses (k+1)/(B+1), so it cannot report zero.

**Open:** these expectations are checked against hand values, not against an independent statistics
package run on the same data. The test suite runs them against SciPy where a SciPy reference exists.

## Segmentation quality

**Checked by reading:** the classical pipeline applies threshold, opening, closing, 8-connected
labelling, optional watershed, minimum-area removal and border removal, then renumbers labels in raster
order. Adaptive thresholding negates the offset when calling scikit-image, so a positive configured
offset raises the threshold, which matches its documented meaning.

**Scoring:** Dice and IoU count any non-zero label as foreground, so label images and binary masks
score the same way. Two empty masks score 1.0, and a mask that is empty against a non-empty one scores
0.0. Both conventions are documented and tested.

**Demonstration scoring:** the demonstration scores each image's labels against its ground truth
(`examples/demo/run_demo.py`). Synthetic truth is noiseless, so these scores are optimistic. They
measure the pipeline on a known target, not performance on microscopy.

**Open:** no segmentation accuracy has been measured on real data, and no run of the demonstration has
been executed here.

## Statistical assumptions

The statistical assumptions are stated in `docs/statistics.md`. The audit found that the design
enforces the central rule (one value per biological sample) by construction: the inference functions
accept sample values only. Object counts are reported, never used as sample sizes. A test checks that
adding objects to one sample does not change its sample count.

**Open:** the rule protects against pseudoreplication within the design. It cannot detect a design that
groups samples wrongly. That remains the responsibility of the person who writes the design file.

## Model evaluation

**Checked by reading:** the score path reuses the segmentation scorer, so ML and classical masks are
evaluated with the same definitions. The U-Net receptive-field bound was re-derived by hand (see
above). Tiled and whole-plane prediction are compared in tests, with the overlap set to the bound.

**Checked by reading:** evaluation refuses a checkpoint whose training identity, architecture,
normalisation or split does not match the current data, so a model cannot be scored on a dataset it
was not prepared for.

**Open:** the ML tests skip without PyTorch and have not run here. No accuracy claim is made for any
model.

## Reproducibility

Random number generators are seeded explicitly everywhere that randomness is used, including the
synthetic data, the bootstrap, the permutation test and the segmentation of phantoms. Each bootstrap
and permutation draws from its own seeded generator, so results do not depend on the order in which
measures are processed.

**Verified on the standard library:** the provenance verifier re-hashes every recorded input and
output, and the tests confirm that an edited input is reported as modified, that a deleted output is
reported as missing, and that the published SHA-256 of `abc` is reproduced.

**Open:** cross-platform bit-identity is not promised. Results are reproducible on one machine with
fixed library versions.

## What remains open

- Every test that depends on NumPy, SciPy, scikit-image, pydantic, typer, matplotlib, pyarrow, PyTorch,
  FastAPI or pytest has not been run in the authoring environment. Run `uv run pytest` before relying on
  this audit.
- No accuracy claim has been measured on real microscopy.
- No benchmark has been run; see `docs/benchmarks.md`.
- The Docker image has not been built and the compose file has not been run.

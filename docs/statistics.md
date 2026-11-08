# Statistical comparison of groups of biological samples

`microflow stats` compares groups of biological samples on the measurements that `microflow export`
writes. It reports descriptive statistics, bootstrap confidence intervals, effect sizes, tests with
corrected p-values, and publication figures. Its central rule is that **the biological sample is the
unit of inference**. Objects and images are never treated as independent replicates.

## The rule, and why it matters

A segmented image can contain hundreds of objects, and a study can have many images per specimen. If
objects were counted as replicates, a comparison could look highly significant from a handful of
animals. That error is pseudoreplication. This module avoids it structurally:

1. Each measure is first summarised **within each biological sample**: the mean (or median) of its
   valid objects, across all of that sample's images.
2. Every descriptive statistic, interval, effect size and test takes those per-sample values as input.
   Object values are reported, but only as counts and as descriptive summaries and faint background
   points in plots.
3. Every image belongs to exactly one sample, and every sample to exactly one group. These assignments
   are declared explicitly in the design file. Nothing is inferred from file names.

Results therefore carry two counts that must not be confused: **samples** (the replicates behind every
test) and **objects** (the observations summarised within them). The output tables show both.

## Study design

A design file names the measurements, the groups, the samples in each group, the images of each
sample, the measures, and the comparisons. Example: `examples/stats/design.yaml`.

```yaml
measurements: measurements.csv      # microflow export, one row per object
data_dictionary: data_dictionary.csv
measures: [area_um2, eccentricity]
per_sample: mean                    # or median
groups:
  control:
    samples:
      - {sample: control-01, items: [control-01_field0, control-01_field1]}
  treated:
    samples:
      - {sample: treated-01, items: [treated-01_field0, treated-01_field1]}
comparisons: [[control, treated]]
```

Validation refuses:

- a sample in two groups, or listed twice;
- an image in two samples;
- a comparison with an undeclared group, a group compared with itself, or a repeated pair;
- a measurement table image that belongs to no sample, which would silently change the analysis;
- a design image with no measurements;
- a row with the wrong number of fields, a non-numeric value, or an infinite value.

Missing values (`NA`, `NaN`, empty) are excluded from their sample's summary, and the sample is excluded
from inference if it has no valid objects. Both are reported.

**Technical versus biological replicates.** Fields of view, tiles and repeated imaging of one specimen
are technical replicates. Declare them as several images of one sample. Separate specimens, donors,
cultures or animals are biological replicates and need separate sample names. The design cannot tell
these apart for you; the person who writes it must.

## Statistical assumptions

These are the assumptions behind the reported numbers. Check them before drawing conclusions.

- **Independence of samples.** Samples are assumed independent. Two samples from one animal, or
  samples processed in one batch that share a systematic effect, are not independent, and the tests
  overstate precision. The design must group them correctly.
- **Sample value is a fair summary.** The mean (or median) of a sample's objects is taken as the
  sample's value. Its meaning depends on the object population, so per-sample counts and
  distributions are reported alongside it.
- **Enough samples.** Each compared group needs at least `min_samples_per_group` samples (default and
  minimum 2). Tests and intervals from two or three samples have little power and unreliable
  coverage. Groups below three samples are flagged `low_replication`. Groups below the minimum are not
  tested, and the comparison reports `NA`.
- **Welch's t-test** does not assume equal variances. It assumes approximately normal sample values,
  which is weak for very small samples.
- **Mann-Whitney** makes no normality assumption, but with few samples its smallest attainable p-value
  is limited, and it tests a shift in distribution rather than a difference in means.
- **Permutation test** of the difference in means assumes exchangeability of samples under the null
  hypothesis. It is exact when the number of ways to split the samples is at most `max_exact`, and
  otherwise a Monte Carlo estimate with the (k + 1) / (B + 1) rule, which never reports zero.
- **Bootstrap intervals** resample samples with replacement within each group. The percentile
  interval is approximate and tends to be too narrow at very small n. With two samples per group the
  intervals are wide and rough, and should be read as indicative.
- **Hedges' g** is the standardised mean difference, pooled across the two groups, with the
  small-sample correction. It assumes comparable spread in the two groups. Where a resample has zero
  pooled spread, that resample gives no finite g; the share that does is reported as stability. Below
  90 % stability the row is flagged `unstable_bootstrap`.
- **Rank-biserial correlation** is reported with the Mann-Whitney test: +1 when every sample in the
  first group exceeds every sample in the second, -1 for the reverse, and 0 for no separation.
- **Multiple comparisons.** Tests are corrected within each test family (Welch, Mann-Whitney,
  permutation) across all measures and comparisons in one run. The default is Benjamini-Hochberg,
  which controls the false discovery rate. Holm controls the family-wise error rate and is more
  conservative. `none` reports raw p-values. Choose the correction when you write the design, and
  state it with the results. Missing p-values are excluded from the family.
- **Measurements are used as exported.** Units are the exporter's: areas in µm² when the image was
  calibrated, otherwise the file's own unit. Intensities are in the input image's units. The module
  does not normalise or transform values, so a transform such as log is the user's decision.
- **Preprocessing is outside this module.** Background correction, flat-field correction and
  normalisation change intensities. Their settings should be the same across the groups being
  compared, and recorded in the export's provenance.

What the module does **not** assume: a particular distribution of objects, or that objects are
independent. Object-level scatter is never used in a test.

## Outputs

`microflow stats run design.yaml` writes to `output_directory`:

| File | Content |
| --- | --- |
| `descriptive.csv` | One row per measure and group: sample count, image and object counts, sample mean, SD, SEM, bootstrap interval for the mean, sample median and quartiles, object mean, SD and median, and flags. |
| `comparisons.csv` | One row per measure and comparison: sample counts, object counts, means, difference with its bootstrap interval, Hedges' g with its interval and stability, rank-biserial correlation, raw and adjusted p-values for each test, the permutation method used, and flags. |
| `sample_values.csv` | One row per sample and measure: the value used for inference, with its image and object counts, and whether it was included. |
| `summary.json` | Identity, design settings, input hashes, versions, counts, test families, warnings and figure list. Written last, so its presence marks a finished run. |
| `plots/` | For each measure, a PNG (300 dpi) and SVG of the sample values with the group mean and bootstrap interval, with object values as faint background points. And `effect_sizes.png/.svg`, the Hedges' g forest plot. |

**Formatting.** CSV files use LF line endings and shortest round-trip floats, so identical results give
identical bytes. Missing numbers are written as `NA`, never as blank or zero. Flags are separated by `;`.

**Identity and overwriting.** Results carry an analysis identity computed from the design settings, the
SHA-256 of the measurement table and data dictionary, and the software and library versions. Results
from a different identity are not overwritten unless `--overwrite` is given. Output location and
file locations are excluded from the identity, so moving the data does not change it. Changing the
figure setting does change it, because it changes which files exist.

**Atomicity.** The design, the table and every assignment are validated before the output directory is
touched. Invalid input therefore never leaves a half-written results directory.

**Random numbers.** Each bootstrap and permutation draws from its own generator, seeded from the
design seed and the names of the measure and comparison. Results do not depend on the order in which
measures are listed, and a repeated run with the same design gives identical numbers.

## Commands

```sh
uv run microflow stats example data/stats-example   # synthetic two-condition data
uv run microflow stats check examples/stats/design.yaml   # validate; no test
uv run microflow stats run examples/stats/design.yaml     # analyse and write results
uv run microflow stats run examples/stats/design.yaml --no-plots --overwrite
```

`check` lists each group's sample, image and object counts and runs no test. `run` prints each
comparison with its sample sizes, Hedges' g and its interval, and the adjusted Welch p-value, then any
warnings.

## Worked example

`microflow stats example` generates two conditions, four donors per condition, two fields of view per
donor, and 35 to 45 objects per field. The only designed effect is on area: treated objects have a
1.4-fold median area. Eccentricity and raw intensity have no designed condition effect. The example
design in `examples/stats/design.yaml` analyses it.

Expected behaviour, which the test suite asserts: the area comparison has four samples per group, a
negative Hedges' g (control minus treated), and a Welch p-value below 0.05. Eccentricity and raw
intensity are reported with their intervals, but their p-values are **not** asserted. At four donors per
group, a null measure can cross 0.05 by chance, and the example is not a calibration study. These
expectations come from the generator's parameters. They pass on macOS arm64 under Python 3.12 and 3.13
with the locked dependencies.

## Limitations

- **Small designs.** Two or three samples per group give wide, rough intervals and low power. A null
  result does not mean no effect.
- **Sample-level summaries lose within-sample information.** A sample with a heavy tail and a sample
  with a uniform spread can have the same mean. Distribution plots and the per-sample counts should be
  read alongside the test.
- **No mixed models.** Nested or crossed designs (for example, several conditions in one experiment
  with batch effects) need a mixed model, which this module does not fit. Declare the grouping that
  matches the question, or analyse the batches separately.
- **Two groups per comparison.** Comparisons are pairwise. Omnibus tests across more than two groups
  are not provided, and pairwise tests over many groups should use the correction that matches the
  question.
- **No covariates or adjustment.** Confounders such as imaging date or donor age are not modelled.
- **Plots are for inspection and drafting.** They are sized for print, but a journal may need its own
  styling. The SVG is written with fixed identifiers so repeated runs match.
- **Floating-point reproducibility.** Results are reproducible on one platform with the same library
  versions. They are not promised to be bit-identical across platforms or SciPy versions.

## Verification status

Executed on macOS arm64, under Python 3.12.11 and 3.13.7. Every statistics test passes: the estimators,
corrections, design validation, data reading, analysis, output, plots, CLI, the example, and the report.
A hand-checked value is now also asserted in code. The Hedges' g for two two-value groups is
-(8/7)√2 = -1.616244. An earlier, rounded constant in a test was wrong and has been corrected.

The demonstration (`examples/demo/run_demo.py`, seed 20260101) gave, for area, Hedges' g of -2.160
(95 % interval -5.605 to -1.613) for control against treated, with four biological samples per group and
an adjusted Welch p-value of 0.086. That is the output of this run on this machine, reported as a result,
not as a validation.

Not done: the statistics have not been compared with an independent package, nested designs are not
modelled, and the figures have not been reviewed by eye. See `ROADMAP.md` items on those topics,
12 and 19.

# User guide

This guide walks through a complete analysis: from a folder of microscopy images to segmented objects,
per-object measurements, a statistical comparison between conditions, and one HTML report. Each step
names the command and the file it produces. For every option, see [the CLI reference](reference/cli.md).

## 1. Describe your images

Every image needs a **calibration**: its pixel size and unit. MicroFlow never assumes that a pixel is a
micrometre. Give the calibration in the batch configuration, or in the image's metadata when the file
carries one (OME-TIFF physical sizes, or TIFF resolution tags in inches or centimetres).

```yaml
dataset:
  images:
    - name: donor1_field0
      path: raw/donor1_field0.tif
      calibration: {pixel_size: 0.5, unit: um}
```

Axes default to `YX`. Declare others with `axes: CYX`, `axes: ZYX` and so on. MicroFlow reads the
axes stored in the file and refuses a declaration that does not match it.

Check a pipeline configuration before running it:

```sh
uv run microflow validate-config examples/synthetic.yaml --check-images
```

`validate-config` accepts pipeline configurations only. It does not accept batch configurations yet
(ROADMAP item 15). A batch configuration is checked when `microflow batch` starts: a configuration
error stops the command with status 1 before any image is processed.

## 2. Segment and measure a batch

```yaml
# batch.yaml
name: my-batch
dataset: {...}                       # see step 1
segmentation:
  method: otsu                       # or threshold (needs a value), adaptive, or plugin
  min_object_area: {value: 4.0, unit: um}
quality_control:
  min_object_area: {value: 4.0, unit: um}
  max_object_area: {value: 200.0, unit: um}
output_directory: results/batch
workers: 2
```

```sh
uv run microflow batch batch.yaml
```

The batch writes, for each image, a label image (`labels.tif`, where 0 is background), a table of
per-object measurements, and a quality-control report (`qc_report.html`). The run writes
`batch_manifest.json` last, so its presence means the run finished. Running the same command again
reuses valid results, so an interrupted run resumes.

Exit status: `0` complete, `3` finished with some failed images, `130` cancelled, `1` a configuration
or directory error.

## 3. Export measurements

```sh
uv run microflow export results/batch --output results/export --format csv --format parquet
```

The export writes `measurements.csv` (one row per object, with the image in the `item` column),
`data_dictionary.csv` (the type, unit and meaning of each column), and `provenance.json`. Area columns
carry the unit in their name: `area_um2`. Intensity columns name their source and channel:
`raw_c0_mean`.

## 4. Compare groups by biological sample

Write a design file that says which images belong to which **biological sample** and which sample to
which group. Two fields of view from one donor are two images of one sample, not two replicates.

```yaml
name: treated-vs-control
measurements: results/export/measurements.csv
data_dictionary: results/export/data_dictionary.csv
output_directory: results/stats
measures: [area_um2, eccentricity]
groups:
  control:
    samples:
      - {sample: donor-01, items: [donor1_field0, donor1_field1]}
  treated:
    samples:
      - {sample: donor-02, items: [donor2_field0, donor2_field1]}
comparisons: [[control, treated]]
```

```sh
uv run microflow stats check design.yaml     # sizes of each group, no test run
uv run microflow stats run design.yaml       # tables, figures, summary
```

Read the output with [the statistics guide](statistics.md) beside you. It states the assumptions each
number depends on.

## 5. Write one report

```sh
uv run microflow stats report results/stats --output results/report.html --batch-dir results/batch
```

The report is a single HTML file. It holds the descriptive and comparison tables, the figures
embedded inline, the warnings, the method notes, and a link from each image to its quality-control
report, so you can open the whole analysis from one file.

## 6. Check what you have

```sh
uv run microflow verify-run results/batch
```

This re-hashes every input and output the run recorded, and reports any that are missing or changed.
It checks that the files are the ones the run produced. It does not recompute the analysis. See
[data provenance](provenance-examples.md).

## Where to go next

- [Installation](installation.md) for versions and the optional extras.
- [Reproducibility](reproducibility.md) for how to rerun an analysis and get the same answer.
- [Audit](audit.md) for what has been checked, and what has not.
- [Statistics](statistics.md), [machine learning](machine-learning.md) and
  [inference](ml-inference.md), and [the service](service.md) for the optional pieces.

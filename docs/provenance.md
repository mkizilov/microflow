# Provenance, reproducibility and exported results

Every batch run records what produced its results. Results can be exported to CSV and Parquet
with defined column names, units and types. Results of different analyses or data sets are not
silently overwritten, and the cache can be checked against a configuration before anything runs.

## What a run records

Each finished run's `batch_manifest.json` contains a `provenance` section with:

| Field | Content |
| --- | --- |
| `schema_version` | Output schema version (currently 1). The manifest itself has `schema_version` 2. |
| `microflow_version` | Software version. |
| `analysis_id` | Identity of the method (see below). |
| `result_set_id` | Identity of the results: the analysis plus the input hashes. |
| `started_utc`, `finished_utc` | Execution timestamps, UTC, ISO 8601, to the second. |
| `resolved_config` | The full configuration as validated, with paths resolved. |
| `algorithm_versions` | Version of each algorithm family (loading, preprocessing, segmentation, quantification, quality control, batch). Bumped by hand when an algorithm's output changes. |
| `library_versions` | Installed versions of numpy, scipy, scikit-image, tifffile, matplotlib, pyarrow, pydantic, pyyaml and typer. `not installed` if absent. |
| `runtime` | Python version and implementation, platform, machine. The host name and user are deliberately not recorded. |
| `inputs` | For each image that was read: name, path, SHA-256 and size in bytes. |
| `warnings` | Failed and cancelled images, per-image warnings (for example QC flags), and the run status when it is not `complete`. |
| `outputs` | Files written by an export (see below). Empty for the batch manifest, whose per-image outputs are listed under `items`. |

Each image's `result.json` records its input hash and configuration fingerprint, and its output
files with their hashes and sizes.

## Identities

- **`analysis_id`** is a digest of the schema version, the MicroFlow version, the algorithm
  versions, the computational library versions (numpy, scipy, scikit-image, tifffile), and the
  analysis settings: preprocessing, segmentation, segmentation channel and quality control.
  Timestamps, runtime details, input files, dataset membership, worker count, memory budget and
  output directory are **not** part of it.
- **`result_set_id`** is a digest of the `analysis_id` and the (name, SHA-256) pairs of the inputs.

Digests are SHA-256 of canonical JSON (sorted keys, no whitespace), so equal content always gives
the same digest. A test pins one digest value to catch accidental changes to that encoding.

## Overwrite protection

Results are never replaced silently.

**Batch output directories.** The run writes `analysis.json` with its `analysis_id`.

- Same identity: the run resumes. Images whose inputs, settings and outputs still match are
  reused, and the rest are reprocessed.
- Different identity: the run is refused with a message naming both identities. Pass `--overwrite`
  to replace the managed files of the earlier analysis.
- Existing batch files with no recorded identity are refused in the same way.

**Export directories.** The export writes `provenance.json` with its `result_set_id`.

- Same result set: the export is refreshed without `--overwrite`.
- Different result set, or existing exported files with no recorded identity: refused. Pass
  `--overwrite` to replace them.
- An export that was interrupted (its exported files exist but `provenance.json` does not) is
  refused until overwritten. The provenance file is written last, so its absence means the
  export is incomplete.

In both cases the overwrite removes only the managed files. Unrelated files in the same
directory are left alone.

## Exported results

`microflow export RUN_DIR --output DIR [--format csv] [--format parquet] [--allow-partial]
[--overwrite]` writes:

| File | Content |
| --- | --- |
| `measurements.csv` | One row per object: `item`, `plane`, then the measurement columns. LF line endings; floats in shortest round-trip form, so identical data gives identical bytes. |
| `measurements.parquet` | The same table with typed columns. Each column's `unit` and `description` are stored as field metadata. The file metadata holds the provenance record (without its `outputs`) under `microflow.provenance`, and the output schema version. |
| `data_dictionary.csv` | For each column: name, dtype, unit and description. |
| `provenance.json` | The export's provenance: source analysis, result set, inputs, warnings, and the SHA-256 of each file written. |

Only images with status `ok` are exported. Exporting a run that did not complete requires
`--allow-partial`; the export then records the excluded images in its warnings. Each exported
image's measurement file is checked against its recorded hash first. A modified file stops the
export, and the message says to re-run the batch.

### Column names and units

Every column name states its unit. The header is fixed for a given set of measurements:

- Geometry: `label` (dimensionless), `area_<u>2`, `perimeter_<u>`, `centroid_row_<u>`,
  `centroid_col_<u>`, `bbox_min_row_<u>`, `bbox_min_col_<u>`, `bbox_max_row_<u>`,
  `bbox_max_col_<u>`, `eccentricity` (dimensionless), `equivalent_diameter_<u>`.
- Intensity, per source and channel: `raw_<channel>_*` or `corrected_<channel>_*` with
  `valid_pixels` (count), `mean`, `median`, `std`, `min`, `max` (intensity), and
  `integrated_intensity_<u>2` (intensity times area).

`<u>` is the calibration unit (`nm`, `um`, `mm` or `m`), or `px` when the image had no
calibration. Intensity values are in the input image's own units; they are not converted, and the
dictionary says so. The exported header is locked by a test, so any change to it is deliberate.

## Comparing configurations and checking the cache

- `microflow compare-configs OLD NEW` lists every setting that differs, with the old and new
  values. Changes to the analysis settings are marked `[changes analysis]`. It also says whether
  the two configurations have the same analysis identity.
- `microflow cache-status CONFIG` reports, without changing anything, whether the output
  directory belongs to this analysis, and for each image whether its cached result is reusable
  or why it would be rerun (`input changed`, `configuration changed`, `output ... was modified`,
  and so on).

## Determinism

Within one machine and one set of library versions, the same configuration and inputs give
identical results. Tests check this: two fresh runs give identical measurement bytes and labels,
an interrupted and resumed run gives the same outputs as an uninterrupted one, and repeated
exports give identical CSV and dictionary bytes.

## Limits and unverified points

- **Parquet byte-identity is not promised.** The data and metadata are checked for equality. The
  file bytes depend on the pyarrow version, so they may differ between versions.
- **Cross-platform bit-identity is not promised.** Floating-point results can differ between
  platforms and BLAS builds. The library versions are recorded so such differences can be traced.
- **Identity is conservative.** Upgrading a computational library or MicroFlow changes the
  `analysis_id`, so existing batch directories are refused until overwritten. That is deliberate:
  the numbers may have changed.
- **Timestamps are excluded from identity** but recorded in provenance, so two runs of the same
  analysis have different `started_utc` and `finished_utc`.
- **pyarrow is a declared dependency that has not been installed or exercised here.** The Parquet
  tests will be the first to exercise it. See the project notes.
- **Intensity units are not physical units.** Mean and integrated intensities are reported in the
  input's units, and the dictionary says so.

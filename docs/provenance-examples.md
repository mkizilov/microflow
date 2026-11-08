# Data provenance examples

Provenance answers four questions about a result: which images it came from, which settings produced
it, which software did the work, and whether the files are still the ones it recorded. This page shows
each answer with the file that holds it. The examples are from the demonstration in `examples/demo/`.

## 1. Which input images were used?

`batch_manifest.json` lists every input, with its path, size and SHA-256 hash:

```json
"provenance": {
  "inputs": [
    {
      "name": "control-01_field0",
      "path": "/.../data/demo/raw/control-01_field0.tif",
      "sha256": "9f2c...",
      "size_bytes": 33192
    }
  ]
}
```

The hash is of the file's bytes, so a changed file has a different hash even when its name and size
are unchanged.

## 2. Which settings produced the result?

The same record holds `resolved_config`: the batch configuration after defaults are filled in. Two runs
can be compared with:

```sh
uv run microflow compare-configs results-a/batch.yaml results-b/batch.yaml
```

The output lists each setting that differs, and marks those that change the analysis. A setting that
changes only where output is written, or how many workers run, does not change the analysis identity.

## 3. Which software and libraries did the work?

`provenance.algorithm_versions` names the MicroFlow version and the algorithms it used, and
`provenance.library_versions` gives NumPy, SciPy, scikit-image, tifffile, pyarrow and the others. The
`runtime` section gives the platform. Host names and user names are not recorded.

## 4. Are the files still the ones that were recorded?

```sh
uv run microflow verify-run results/batch
```

Output for an untouched run:

```text
run status: complete; 34 recorded file(s) checked
all recorded files match
```

Output after one input was edited and one label image was deleted:

```text
run status: complete; 34 recorded file(s) checked
modified: input control-01_field0 (/.../raw/control-01_field0.tif)
missing: output control-02_field0/labels (/.../items/control-02_field0-.../labels.tif)
```

Exit status is `1` in both failing cases, so the check can run in a script. The verifier reads the run
manifest, each item's `result.json`, and the files it names. It needs no scientific library.

## 5. Which inputs produced each exported row?

Each row of `measurements.csv` has an `item` column, which names the image. Join it to the input in
the manifest by name:

```python
import csv, json

manifest = json.load(open("results/batch/batch_manifest.json"))
hashes = {item["name"]: item["sha256"] for item in manifest["provenance"]["inputs"]}
with open("results/export/measurements.csv", newline="") as handle:
    for row in csv.DictReader(handle):
        source_hash = hashes[row["item"]]  # the image this object was measured in
```

## 6. What did a statistical result depend on?

The statistics summary records the hashes of the measurement table and the data dictionary it read,
along with the design settings and the analysis identity:

```json
"inputs": {
  "measurements_sha256": "5b1e...",
  "data_dictionary_sha256": "e21a..."
}
```

If either file changes, the identity changes, and rerunning without `--overwrite` is refused. The
report repeats the design identity in its method notes, so a printed report can be matched to its
analysis.

## Limits

Provenance records what was used and what was produced. It does not verify the science. A correct hash
of the wrong image, or a correct configuration with a wrong threshold, is still wrong. See
[the audit](audit.md) for what was checked, and [reproducibility](reproducibility.md) for how to rerun.

# Reproducibility

MicroFlow is built so that the same inputs and settings give the same results, and so that a result
can later be checked against the files it came from. This page says what that guarantee covers, what
it does not, and how to rerun an analysis.

## What is fixed

- **Versions.** `uv.lock` pins every dependency. Use `uv sync --locked`. Releases are tested on
  Python 3.12 and 3.13 in CI.
- **Randomness.** Every random step takes an explicit seed: the synthetic data, the bootstrap, the
  permutation test, and the phantoms used in tests and the demonstration. Each bootstrap and
  permutation draws from its own generator, derived from the seed and the names of the measure and
  comparison. Results do not depend on the order of measures.
- **Analysis identity.** A batch run records an identity computed from its settings and the software and
  library versions. A statistics run records an identity computed from its design, the hash of its
  measurement table, and the versions. A different identity is never silently overwritten.
- **Output bytes.** CSV files use LF line endings and shortest round-trip floats, so repeated runs with
  the same inputs write the same bytes. Figures are written with fixed SVG identifiers and no creation
  date.

## What is not fixed

- **Bit-identical floating point across machines.** NumPy and SciPy may use different instruction sets
  or thread counts. Results are reproducible on one machine with fixed library versions. They are not
  promised bit-identical across platforms.
- **Parquet bytes.** Parquet files contain the same data and metadata, but their bytes are not promised
  to match across pyarrow versions.
- **Model training on GPU.** Training on CUDA is not bit-reproducible. CPU training with a fixed seed and
  thread count is.
- **Wall-clock timings.** These depend on the machine. See [benchmarks](benchmarks.md).

## How to rerun an analysis

1. Use the same source. Record the commit: `git rev-parse HEAD`.
2. Use the same environment: `uv sync --locked`, and note `microflow version`.
3. Use the same inputs. Check them against the recorded hashes:

   ```sh
   uv run microflow verify-run results/batch
   ```

4. Use the same configuration file. Compare it with the one recorded in the run's provenance:

   ```sh
   uv run microflow compare-configs old-batch.yaml new-batch.yaml
   ```

5. Rerun with the same command. The output directory's identity guard refuses to mix results from a
   different analysis unless you pass `--overwrite`.

For the demonstration, the whole chain can be rerun from one command, and the synthetic images are
regenerated from the seed in `examples/demo/run_demo.py`:

```sh
uv run python examples/demo/run_demo.py --clean --seed 20260101
```

## What to record with a published result

- The MicroFlow version and commit.
- The `uv.lock` file's hash, or the lock file itself.
- The configuration and design files.
- The input hashes from `batch_manifest.json` (`provenance.inputs`), or `verify-run` output.
- The seed for any random step, which the configuration files already contain.
- Hardware, for any timing claim.

## Limits of the verifier

`microflow verify-run` shows that the files on disk are the ones the run recorded. It does not show that
the recorded analysis was correct, that the settings were the right ones, or that the images are the
right images for the question. Those need the audit, the statistics assumptions and the scientific
judgement of the person running the analysis.

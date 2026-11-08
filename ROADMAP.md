# MicroFlow roadmap

This roadmap records what MicroFlow implements, what has been verified, and what remains open.
It was written at the final review, after the feature work, against commit `992b02b` plus the
review commit that adds this file. Read it before relying on any part of the project.

**The short version:** the feature set is implemented, but almost none of the software has been
executed. Of about 735 test functions, 52 have run, on the standard library only, through a scratch
runner. The lock file, the CI workflows, the Docker image, the end-to-end demonstration and every
benchmark have never run. Most of the work below is verification, not new features.

## How to read this document

- **Implemented:** the code or documentation exists in the repository.
- **Verified:** it was executed, and the evidence is recorded in this document or in the repository.
- **Partial:** part of it exists, or it exists only as configuration.
- **Unverified:** it exists but has not been executed or measured.

Priorities: **P0** blocks reproducible use or is a correctness or security risk. **P1** is an
unexecuted integration or hardware check. **P2** is scientific validation and feature work. **P3** is
cleanup and deferred decisions.

## Status summary

| Area | Implemented | Verified here | Not yet verified |
| --- | --- | --- | --- |
| Configuration, units, image loading | yes | static checks only | runtime behaviour, ImageJ calibration (not read) |
| Preprocessing and classical segmentation | yes | static checks; hand checks of units and perimeter | runtime tests; accuracy on real data |
| Measurements | yes | static checks; hand checks | runtime tests; `measures` setting unused |
| Quality-control reports and figures | yes | static checks | figures and report never viewed |
| Batch engine | yes | static checks | parallel path, cancellation and resume at runtime |
| Provenance, verification and export | yes | `verify-run` tests: 6 of 6 on the standard library | Parquet (pyarrow) never run |
| Plugins | yes | static checks | execution, isolation, example installation test (skipped) |
| Machine learning (training, evaluation, inference) | yes | static checks; receptive-field bound re-derived by hand | every runtime test; CUDA path |
| Statistics | yes | estimators hand-checked; report tests 6 of 6 | runtime tests; independent reference; nested designs |
| HTTP service | yes | path policy 32 of 32, job store 8 of 8 (standard library); static isolation check | HTTP layer, lifespan, routing, Docker |
| Docker and compose | written | none | build, run, health check |
| CI workflows | written | YAML structure read by hand | every job; `uv.lock` is a prerequisite |
| End-to-end demonstration | written | item names, paths and flags cross-checked | execution, timing, report review |
| Benchmarks | script written | argument checks run | no timing measured; no results committed |
| Documentation | yes | links, command references, example paths, TOML parsed | human review; several claims were corrected in this review |

## Evidence behind this status

Everything below was run in the authoring environment, which has Apple's Python 3.9.6, no network, no
`uv`, and none of the scientific or web packages.

- **Standard-library tests, executed:** 52 of 52 pass. `test_service_paths` (32), `test_service_store`
  (8), `test_provenance_verify` (6) and `test_stats_report` (6). They ran through a scratch
  pytest-subset runner, not pytest, and on Python 3.9 rather than 3.12.
- **Static checks over 170 Python files:** no undefined names, unused imports, call-arity mismatches,
  missing module attributes or unused locals. No torch import is reachable from the torch-free
  modules, and no web-framework import is reachable from the CLI. No line exceeds 100 characters.
- **Import cycles:** five package-level cycles. An earlier review reported only one; the earlier checker
  stopped at its first cycle. See item 4.
- **Documentation and configuration:** 24 relative links resolve in 20 documents. All 29 documented
  `microflow` command forms exist in the generated reference. The reference drift check passes.
  `pyproject.toml` parses (with pip's vendored TOML parser), and both entry-point targets exist. 17
  example YAML files have no tab indentation, and their relative input paths resolve. Inputs under
  `data/` and `results/` are exempt because documented commands generate them.
- **Not parsed:** YAML files and the CI workflow. No YAML parser is available here.
- **Hand checks:** Benjamini-Hochberg and Holm on a four-value family, Hedges' g for two two-value
  groups, the exact permutation count for complete separation of three against three, the
  perimeter formula for square objects, and the unit conversions.
- **Test functions never executed:** about 683 of 735 (static count of `def test_`, before
  parametrisation). These include every test that uses NumPy, SciPy, scikit-image, tifffile,
  pydantic or typer.
- **Live processes:** the sandbox blocks `ps`. Nothing was started in the background during this
  review, so there was nothing to stop.

## Requirements check

| Original requirement | Implemented | Verified here | Open items |
| --- | --- | --- | --- |
| 1. Scaffold, uv, CI, AGENTS, sample data | yes | CI never ran; `uv.lock` absent | 1, 3, 16 |
| 2. Configuration and dataset model | yes | static checks; config tests unexecuted | 2, 15 |
| 3. Image loading (TIFF, OME-TIFF, NumPy) | yes; ImageJ calibration not read | static checks | 2, 23 |
| 4. Preprocessing and provenance of steps | yes; `ProcessingConfig` settings unused | static checks | 5, 11 |
| 5. Classical segmentation | yes; adaptive options silently ignored | static checks; accuracy unmeasured | 6, 18 |
| 6. Per-object measurements with units | yes; `measures` unused | hand checks of perimeter and units | 2, 5 |
| 7. Quality-control reports and figures | yes | figures never viewed | 12 |
| 8. Batch engine with resume and cancellation | yes; process pool unrun | static checks | 7, 14, 17 |
| 9. Provenance, export, configuration comparison | yes; Parquet unverified | verifier 6 of 6 | 2, 21 |
| 10. Plugin architecture | yes; in-process, no timeout | static checks; example installation test skipped | 2, 7 |
| 11. U-Net training with leakage guards | yes | static checks | 9, 20 |
| 12. ML inference with tiling and fallback | yes | receptive-field bound re-derived by hand | 9, 20 |
| 13. Statistics with explicit samples | yes | hand checks; report tests 6 of 6 | 11, 12, 19 |
| 14. Authenticated HTTP service | yes | path policy and store tests executed | 8, 10, 13 |
| 15. Packaging, Docker, benchmarks, CI matrix | docs yes; Docker and benchmarks written | documentation checks; reference drift check | 13, 16, 17, 21 |
| Scientific audit | yes | by reading and hand arithmetic | 18, 19 |
| End-to-end demonstration | written | cross-checked only | 11 |
| Test coverage reported | no | no coverage was measured | 2 |

## Corrections made in this review

- **README.** The opening said analysis algorithms were not implemented. It now describes the state
  above. The lock-file section said `uv.lock` existed, which was false. The CI and demonstration
  sections implied they had run. All three are corrected.
- **Architecture.** Four implemented stages were still listed as placeholders. Those rows are removed.
  The `pipelines` placeholder is kept, with an accurate description (item 22).
- **User guide.** It said `validate-config` checks batch configurations. It does not. That command
  accepts only pipeline configurations (`MicroFlowConfig` forbids batch keys). Item 15 tracks the gap.
- **Import cycles.** Earlier statements that the batch cycle was the only one were wrong. There are five
  (item 4).
- **Verification sections.** The service and statistics sections described their tests as unrun. The
  service path-policy and store tests and the statistics report tests have now run, and the sections
  say so.
- **AGENTS.md.** Authoring-workflow wording and references to managed configuration are removed.
  The repository boundaries and verification rules are kept.

## 1. Generate and commit `uv.lock`; prove locked installs

**Priority:** P0. **Status:** unverified blocker. `uv.lock` does not exist, so `uv sync --locked` fails
for every user, in every CI job that uses it, and in the Docker build.

**Scope.** Run `uv lock` with network access on Python 3.12. Commit `uv.lock`. Confirm that the PyTorch
CPU index applies only on Linux, as `[tool.uv.sources]` specifies. Run `uv sync --locked` and
`uv sync --locked --extra ml --extra service`. Add `uv lock --check` to CI.

**Rationale.** The project claims reproducibility, and reproducibility needs a recorded set of
versions. Without a lock no tested version is recorded, and the README and installation documents
currently describe a lock file that is not there.

**Dependencies.** Network access to the package indexes. Nothing else.

**Acceptance criteria.**
- `uv.lock` is committed, and `uv sync --locked` succeeds on Linux and macOS with Python 3.12 and 3.13.
- `uv lock --check` passes in CI.
- The resolved versions of NumPy, SciPy, scikit-image, tifffile, pydantic, pyarrow and PyTorch are
  recorded in `docs/reproducibility.md`.

**Effort:** 2–4 hours.

## 2. Execute the full test suite on Python 3.12 and triage every failure

**Priority:** P0. **Status:** implemented, unverified. About 683 of 735 test functions have never
executed.

**Scope.** Run `uv run pytest` on 3.12 without extras, on Linux and macOS. Fix code, or fix a test
whose expectation was wrong, and record the reason in the commit message. Measure coverage with
`pytest-cov`, which is not yet a dependency. Check that every skip has the documented reason. Confirm
that the example-plugin installation test runs and does not skip. It currently skips without `uv`.

**Rationale.** Most of the code has never run. The expected failure modes are API mismatches with the
pinned libraries, wrong numeric tolerances, path and platform differences, and the assumptions the
documents flag as unverified: tifffile behaviour for compressed data and for truncated files, and the
OME metadata keys.

**Dependencies.** Item 1.

**Acceptance criteria.**
- `uv run pytest` passes with no failures on 3.12 on Linux and macOS.
- No test is deleted to make the suite pass. Every changed expectation has a commit message that says
  why the expectation was wrong.
- The skip count and coverage are recorded in `docs/audit.md`.
- The example-plugin installation test runs in CI.

**Effort:** 12–24 hours. The range depends on how many assumptions prove wrong. If failures reveal
design faults rather than small mismatches, double the upper bound.

## 3. Lint, format and type-check strictly

**Priority:** P0. **Status:** implemented, never run (Ruff, Ruff format, mypy).

**Scope.** Run `uv run ruff check .`, `uv run ruff format --check .` and `uv run mypy` (strict, over
`src` and `tests`). Make them pass in CI on 3.12 and 3.13.

**Rationale.** Expected findings: formatting differences across many files, `Any` returns from
tifffile and torch, an untyped `_InlineExecutor.submit` that currently has a `type: ignore`, and
untyped route decorators in the service. The torch and FastAPI overrides in `pyproject.toml` are
deliberate and should stay as narrow as they are.

**Dependencies.** Item 1, for pinned tool versions.

**Acceptance criteria.**
- All three commands pass in CI.
- No suppression is added beyond the documented torch and FastAPI overrides. Each remaining
  `type: ignore` has a stated reason.
- The formatting change is one separate commit.

**Effort:** 6–12 hours.

## 4. Break the package import cycles and prove that imports work

**Priority:** P0. **Status:** implemented, with five package-level import cycles that have never been
executed under Python's import system:
- `microflow.batch` ↔ `batch.engine`, `batch.worker` and `batch.export`
- `microflow.segmentation` ↔ `segmentation.pipeline`
- `microflow.preprocessing` ↔ `preprocessing.pipeline`

The batch engine and export import `from microflow.batch import store` while the package is still
initialising. Python can resolve that particular form, but the other back-edges depend on the order
in which names are defined.

**Rationale.** A wrong order raises `ImportError: cannot import name ... from partially initialized
module`. That would break `import microflow.batch`, and with it the CLI batch command, the service and
the demonstration. Nothing has ever imported those modules at runtime.

**Scope.** Move shared names into leaf modules, or remove the package-level re-exports that create the
back-edges. Add a test that imports every `microflow` module in a fresh interpreter.

**Dependencies.** Item 2 for the test harness.

**Acceptance criteria.**
- The import test passes on 3.12.
- The compact cycle check reports no cycles.
- The existing tests still pass with no behaviour change.

**Effort:** 4–8 hours.

## 5. Wire up or remove the dead configuration settings

**Priority:** P0, correctness. **Status:** partial. The settings are validated and recorded, but
nothing reads them:
- `ProcessingConfig.smoothing_sigma` and `ProcessingConfig.subtract_background` (`config.py`)
- `QuantificationConfig.measures` (no reader in `quantification` or `batch`)

**Rationale.** A user who sets one of these gets no effect and no warning. In a scientific tool, that
is a silent correctness failure. Its provenance record also claims a setting that had no effect.

**Scope.** For each setting, either connect it to the behaviour it names (the `subtract_background`
and `gaussian` preprocessing steps, and the column selection for `measures`), or remove it from the
schema with a migration note. Prefer connecting them.

**Dependencies.** Item 2.

**Acceptance criteria.** For each setting, a test shows that changing it changes the output, or the
setting is removed and `validate-config` rejects it with a message that names the field. The recorded
`resolved_config` matches the behaviour.

**Effort:** 4–8 hours.

## 6. Refuse explicitly set options that the chosen method ignores

**Priority:** P0, correctness. **Status:** partial. The plugin method refuses built-in options, and
the check is tested. Classical methods silently ignore `adaptive_*` and `watershed_*` settings that do
not apply to them. `adaptive_offset` is read only on the adaptive branch.

**Scope.** Extend the same validator to the adaptive and watershed fields. Check `model_fields_set`, so
that defaults never trigger the error.

**Rationale.** It is the same silent-ignore failure as item 5, in the segmentation settings.

**Dependencies.** Item 2.

**Acceptance criteria.** A configuration that sets `adaptive_offset` with `method: otsu` fails, and the
message names the field. Defaults still validate. Tests cover every method.

**Effort:** 2–3 hours.

## 7. Isolate plugin execution: timeouts, memory limits and a subprocess option

**Priority:** P0, security. **Status:** implemented as in-process execution with a documented trust
model. There is no timeout, no memory cap and no process isolation. The service refuses plugins, but
the batch command runs them.

**Scope.** Run each plugin call in a worker process with a wall-clock timeout and an address-space
limit, or add a subprocess mode and gate it. A hung or memory-hungry plugin must fail only the item it
was processing. Document the behaviour on platforms where the limit cannot be enforced.

**Rationale.** A plugin runs with the process's full privileges, and it can hang a batch or exhaust
memory for every image.

**Dependencies.** Item 2. Item 14 (the parallel path), because the isolation touches the executor.

**Acceptance criteria.**
- A test plugin that sleeps forever and one that allocates beyond the limit each fail only their own
  item with an explicit error. The rest of the batch completes.
- `docs/plugins.md` matches the behaviour, including on unsupported platforms.

**Effort:** 12–20 hours.

## 8. Harden the service for non-loopback deployment

**Priority:** P0, security. **Status:** implemented for loopback use. Missing: a request-size limit,
throttling of failed authentication, a token that rotates without a restart, an audit log, and a
bound on validation work. Validating a statistics design reads the whole measurement table. TLS is
left to a proxy and is not tested.

**Scope.** Add a body-size limit that returns 413. Throttle clients after repeated failed tokens. Log
accepted and refused requests without the token. Reload the token on `SIGHUP` or from a file. Bound
the validation work. Add a tested reverse-proxy example.

**Rationale.** `--allow-remote` is supported. A remote client could now send an unbounded body and try
tokens without limit.

**Dependencies.** Item 10.

**Acceptance criteria.** Tests for 413 on oversized bodies and 429 after repeated failures. A test that
the token never appears in the log. The proxy example is run once, or a manual checklist is recorded.

**Effort:** 8–14 hours.

## 9. Execute the machine-learning path on CPU and check checkpoints

**Priority:** P1. **Status:** implemented, never executed. The tests skip without PyTorch. The `ml` CI
job has never run and depends on item 1.

**Scope.** Run `uv sync --locked --extra ml` and `uv run pytest -m ml`. Train, evaluate and infer on
the synthetic dataset. Confirm the checkpoint round trip with `weights_only=True`, the `TorchVersion`
regression test, the tiled-against-whole-plane equality at a tolerance of 1e-5, the refusal to
evaluate a mismatched checkpoint, and the leakage spy tests.

**Rationale.** The checkpoint loader, the tiling and the leakage guards are the most intricate
numerical and security code in the project, and they have only been read.

**Dependencies.** Items 1 and 2.

**Acceptance criteria.** `pytest -m ml` passes on 3.12 with CPU PyTorch, and the `ml` CI job is green.
The synthetic run's metrics are recorded in `docs/machine-learning.md`, labelled as a smoke test and
not as an accuracy claim.

**Effort:** 6–10 hours.

## 10. Execute the HTTP service test suite

**Priority:** P1. **Status:** implemented, never executed at the HTTP layer. The path policy and job
store ran on the standard library (see the evidence section). The HTTP tests need FastAPI.

**Scope.** Run `uv sync --locked --extra service` and `uv run pytest -m service` on 3.12 and 3.13.
Check Starlette's handling of encoded slashes (`%2F`) against the traversal tests. Those tests
currently accept two status codes for one case. Replace that allowance with the observed behaviour and
document it.

**Rationale.** Route wiring, dependency injection, the lifespan handler and the exception handlers
have never run.

**Dependencies.** Item 1.

**Acceptance criteria.** `pytest -m service` passes on 3.12 and 3.13. The two-code allowance is replaced
by the single observed status code, with a note.

**Effort:** 4–6 hours.

## 11. Run the end-to-end demonstration and fix integration failures

**Priority:** P1. **Status:** implemented, never executed. The item names, file paths and CLI flags
were cross-checked by reading.

**Scope.** Run `uv run python examples/demo/run_demo.py --clean` locally and in CI, and fix whatever
fails. Record the run time and the outputs. Test the stop-on-failure path by breaking one input in a
scratch copy.

**Rationale.** The demonstration is the only test of the whole chain in sequence: batch, then export,
then statistics, then the report, then verification. No part of that chain has run together.

**Dependencies.** Items 1 and 2.

**Acceptance criteria.** The demonstration completes on Linux and macOS. `verify-run` reports every file
as matching. The segmentation scores against the ground truth are recorded. The report opens. The
broken-input test stops the demonstration at the failed step.

**Effort:** 4–8 hours.

## 12. Review every figure and the HTML report visually

**Priority:** P1. **Status:** implemented, never viewed. This covers the six QC figures per image, the
per-measure and effect-size statistics plots, the PNG preview, and the HTML report.

**Scope.** Review each figure and the report at print size. Check label overlap, axis units, colour
choices for colour-vision deficiency, legend legibility, the empty-result case, and the case with
many objects.

**Rationale.** Plots are the main communication output. Overlapping labels and missing units are common
failures, and no static check catches them.

**Dependencies.** Item 11.

**Acceptance criteria.** The reviewed images and the findings are recorded in `docs/quality-control.md`
and `docs/statistics.md`. The fixes are committed. Structural tests check the format and the caption
text, and every axis label carries its unit.

**Effort:** 4–8 hours.

## 13. Build and run the Docker image, and add a health check

**Priority:** P1. **Status:** written, not built. The `Dockerfile`, `compose.yaml` and `.dockerignore`
exist. A Docker host outside this environment is required. Item 1 is a prerequisite.

**Scope.** Build from a clean checkout. Start the compose file. Confirm `GET /v1/health` through the
published loopback port. Submit a job with images in `/data/input` and check its output in
`/data/output`. Confirm that the read-only root filesystem allows only the documented writes. Add a
`HEALTHCHECK`. Run the service tests against the container.

**Rationale.** The deployment depends on container paths, write locations and the non-root uid. None of
these has been tested.

**Dependencies.** Items 1 and 10, and a Docker host.

**Acceptance criteria.** The build succeeds. The health check and a complete job succeed. A write outside
`/data/output`, `/data/state` or `/tmp` fails. The results are recorded in `docs/docker.md`.

**Effort:** 3–5 hours.

## 14. Test the parallel batch path on Linux and macOS

**Priority:** P1. **Status:** implemented, never run with more than one worker. The service avoids the
path on purpose, with one worker per batch.

**Scope.** Run the batch tests with `workers=2`. Test a worker crash, in which the images in flight
fail and the others complete. Kill a run part-way through and resume it. Check memory-budget admission.
Check `spawn` on macOS and `fork` on Linux.

**Rationale.** Multi-process code with resume and cancellation is the riskiest part of the batch
engine, and it has never run. The memory budget is an estimate, not an operating-system limit.

**Dependencies.** Items 2 and 4.

**Acceptance criteria.** Every batch test passes with one and with two workers on both platforms. The
crash and resume tests exist and pass. For the same inputs, the CSV output is byte-identical with one
worker and with two.

**Effort:** 4–8 hours.

## 15. Validate batch configurations on the command line, and correct the user guide

**Priority:** P1. **Status:** partial. `validate-config` accepts only pipeline configurations. The
service's `/v1/validate` checks batch files. The command line has no batch check, other than starting
the run.

**Scope.** Add `microflow validate-batch CONFIG [--check-images]`, using the same checks as
`load_batch_config` and the service's `prepare_batch`. Correct the user guide to show the real
commands.

**Rationale.** Users need a check before a run that costs nothing and processes no images.

**Dependencies.** Item 2.

**Acceptance criteria.** Exit status 0 for a valid file, and 1 for an invalid one, with the file and the
field name in the message. Tests cover each class of error.

**Effort:** 2–3 hours.

## 16. Validate documentation and example files in CI

**Priority:** P1. **Status:** partial. The checks described in the evidence section run by hand, from a
scratch script. They are not in CI, and YAML is not parsed.

**Scope.** Add a `docs` CI job that runs the link check, the command check and the reference drift
check. Parse each example with its project loader: `validate-config` for pipeline examples,
`load_batch_config`, `load_design`, `load_steps` and `load_training_config` for the others.

**Rationale.** The documents have already drifted from the code in this project. The user guide claimed
a command did something it does not, and the README overview was years out of date.

**Dependencies.** Items 1, 2 and 15.

**Acceptance criteria.** The job fails on a broken link, a stale reference or an invalid example. Every
example loads with its loader.

**Effort:** 3–5 hours.

## 17. Commit benchmark results and add a batch-throughput benchmark

**Priority:** P1. **Status:** partial. `benchmarks/bench_pipeline.py` exists, and its argument checks
ran. No timing has been measured, no results are committed, and there is no batch-level benchmark.

**Scope.** Run the pipeline benchmark at 256, 512 and 1024 pixels, with five repeats and bounded memory,
on a documented machine. Add a batch benchmark with 16 images at one and at two workers, measuring wall
time and peak memory. Commit the JSON files with the hardware and library versions in
`docs/benchmarks.md`.

**Rationale.** No performance claim is currently supported. A benchmark number is valid only for the
machine and the commit that produced it.

**Dependencies.** Items 1, 2 and 14.

**Acceptance criteria.** The JSON is committed with the hardware, the library versions and the commit
hash. Repeat-to-repeat variation is reported. No number appears in the documents without its source
file.

**Effort:** 8–14 hours.

## 18. Validate segmentation accuracy on real annotated data

**Priority:** P2. **Status:** unverified. Every score so far comes from noiseless synthetic phantoms,
and those scores are optimistic.

**Scope.** Obtain an annotated microscopy dataset whose licence permits this use, or use the lab's own
data. Run classical methods (Otsu, adaptive, watershed) and the ML plugin under identical contracts.
Report Dice and IoU with bootstrap intervals, per image and per biological sample. Document the failure
modes: touching objects, low contrast and debris.

**Rationale.** Segmentation quality is the scientific core of the project, and nothing has been
measured on real data.

**Dependencies.** Items 2, 9 and 11, and access to suitable data. Data access is outside the project's
control.

**Acceptance criteria.** An evaluation report is committed. It covers dataset provenance, the split
rules, scores with confidence intervals, and examples of failures. Thresholds are chosen on a validation
split only.

**Effort:** 16–30 hours. If no annotated dataset exists, add 40 or more hours for annotation.

## 19. Validate the statistics against an independent implementation, and add nested designs

**Priority:** P2. **Status:** implemented. The estimators were checked by hand on a few cases. They have
not been compared with an independent implementation. Nested and crossed designs are not modelled.

**Scope.** Compare the Welch, Mann-Whitney, permutation, bootstrap and Hedges' g values on five fixtures
against SciPy and statsmodels, or R, with stated tolerances. Add mixed-effects or nested-design support,
or refuse nested designs with a clear message. Add omnibus tests for more than two groups.

**Rationale.** The statistics documentation makes claims that have been checked only by hand. A nested
design would currently be analysed as if its samples were independent.

**Dependencies.** Item 2.

**Acceptance criteria.** The fixtures agree within tolerance in CI. Nested designs are either modelled
with a test or refused with a message. `docs/statistics.md` matches.

**Effort:** 12–20 hours.

## 20. Extend the ML module, and add a GPU checklist

**Priority:** P2. **Status:** partial. The module handles two-dimensional, single-channel images, held in
memory. It has no early stopping and no resume. The CUDA path has never run.

**Scope.** Add multichannel input. Add optional three-dimensional patches. Add early stopping on
validation Dice. Add resume from `last.pt`. Verify CUDA with a manual checklist for an operator with
the hardware, because the project's boundaries forbid this environment from touching attached hardware.
Add a CUDA CI job only if a runner exists.

**Rationale.** These are the main stated limits in `docs/machine-learning.md`.

**Dependencies.** Items 1 and 9.

**Acceptance criteria.** Each feature has CPU tests. A resumed run produces the same weights as an
uninterrupted run with the same seed, on CPU. The GPU checklist is run by a person and its results are
recorded.

**Effort:** 16–24 hours.

## 21. Release engineering: build, install from the wheel, golden outputs, changelog

**Priority:** P2. **Status:** partial. The packaging metadata exists. The project has not been built,
no wheel has been installed, there is no changelog, and there is no set of golden outputs across
platforms.

**Scope.** Run `uv build`. Install the wheel into a clean environment. Confirm the `microflow` script
and the `microflow.segmenters` entry point. Write a changelog and tag `v0.1.0`. Commit golden outputs:
the export column contract, a provenance digest, and the CSV from one small batch. Check them in CI.

**Rationale.** Users install a wheel, not the repository. Golden outputs make cross-platform drift
visible.

**Dependencies.** Items 1 and 2.

**Acceptance criteria.** The wheel installs in a fresh environment on Linux and macOS. `microflow version`
and `microflow plugins list` work. The golden checks pass.

**Effort:** 4–6 hours.

## 22. Decide the fate of the placeholder `microflow.pipelines` package

**Priority:** P3. **Status:** placeholder. The package is empty, and its docstring says "Not yet
implemented". It was reserved for a programmatic composition API. Today the batch engine performs the
composition.

**Scope.** Either implement a thin Python function that delegates to the same code as the CLI, with
tests, or delete the package and its architecture row.

**Rationale.** An empty public package in a scientific library misleads its users.

**Dependencies.** None.

**Acceptance criteria.** The package has a tested public function, or it is gone. The architecture row
matches.

**Effort:** 1–2 hours to remove, or 8–12 hours to implement. The totals below assume removal.

## 23. Known small gaps and code hygiene

**Priority:** P3. **Status:** known and small.
- ImageJ calibration metadata is not read, so ImageJ TIFFs load without their pixel size.
- `preprocessing/pipeline.py` has a duplicate unit-conversion helper, `_pixels`, which duplicates
  `microflow.units.to_pixels` and can drift from it.
- `_InlineExecutor.submit` is untyped and carries a `type: ignore`.

**Scope.** Read ImageJ calibration, or document it as unsupported. Replace the duplicate helper with
`to_pixels`. Type the inline executor.

**Rationale.** Duplicated conversion logic can drift, and an unread calibration silently changes
measured units.

**Dependencies.** Item 3.

**Acceptance criteria.** One unit-conversion path. ImageJ calibration is read and tested, or its absence
is documented. The existing tests pass.

**Effort:** 2–4 hours.

## Effort summary

| Priority | Items | Low (hours) | High (hours) |
| --- | --- | --- | --- |
| P0 | 1–8 | 50 | 93 |
| P1 | 9–17 | 38 | 67 |
| P2 | 18–21 | 48 | 80 |
| P3 | 22–23 | 3 | 6 |
| **Total** | 1–23 | **139** | **246** |

**Assumptions behind the total:**
- One engineer who knows the Python scientific stack and this codebase.
- Network access, a Linux machine and a macOS machine, and a Docker host for item 13.
- Item 2 assumes most of the roughly 683 unexecuted tests fail for small reasons. If the failures are
  design faults, its upper bound roughly doubles.
- Item 18 depends on access to data. The total excludes annotation effort, which could add 40 hours or
  more.
- Items 19 and 20 assume the scope stated above. The GPU checklist is run by a person, not included in
  the hours.
- Estimates exclude CI queue time, review cycles and translation. They have not been checked against a
  measured rate for this codebase.
- Item 22 uses the removal estimate. Implementing it adds about 10 hours at the top of the range.

At 35 to 40 hours a week, the total is about 4 to 7 engineer-weeks.

## Sequencing

The critical path is **1 → 2**. Items 3, 4, 9, 10, 11, 14 and 15 depend on item 2. Items 5, 6 and 7 can
start in parallel once the test harness from item 2 exists. Item 8 needs item 10. Items 12 and 17 wait
for the executed demonstration and the parallel path. Items 18 to 21 come last, because they depend on
an executed, verified base.

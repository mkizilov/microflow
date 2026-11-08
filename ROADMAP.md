# MicroFlow roadmap

This roadmap states what MicroFlow implements, what has been executed and passed, and what remains
open. It was refreshed after the publication verification pass, which ran the real build, type, lint
and test tools for the first time with the locked dependencies installed.

## How to read this document

- **Implemented:** the code or documentation exists in the repository.
- **Verified:** it was executed here, and the result is recorded below or in the repository.
- **Partial:** part of it exists, or it ran only in part.
- **Unverified:** it exists but has not been executed here, or only by a stand-in.

Priorities: **P0** blocks publication or is a correctness or security risk. **P1** is an unexecuted
integration or platform check. **P2** is scientific validation or a feature. **P3** is cleanup.

## Verification record (macOS 27.2, arm64; uv 0.8.22; locked environment)

| Check | Result |
| --- | --- |
| `uv lock --check` and `uv sync --locked --all-extras` | Pass. `uv.lock` pins 63 packages. |
| Python 3.12.11: full test suite | 939 passed, 1 skipped (the CUDA test; no GPU here), 1 warning. About 80 s. |
| Python 3.13.7: full test suite | 939 passed, 1 skipped, 1 warning. About 83 s. |
| `ruff check .` (both Python versions) | Pass. |
| `ruff format --check .` | Pass, 193 files. |
| `mypy` in strict mode, 163 source files (both Python versions) | Pass. |
| `scripts/generate_reference.py --check` | Pass. |
| `scripts/generate_license_inventory.py --check` | Pass. 61 packages; no GPL or AGPL; MPL-2.0 in `certifi` and `pathspec`. |
| End-to-end demonstration (`examples/demo/run_demo.py --clean`) | Complete. 16 images, 169 object rows, statistics, HTML report, and `verify-run` matches all 64 recorded files. |
| ML commands from the CI workflow, on CPU | Pass: dataset, splits, training (3 epochs), held-out evaluation. Smoke test only. |
| Wheel and sdist build; metadata | Pass. `License-Expression: MIT`, `License-File: LICENSE`, and the LICENSE file is installed. |
| Wheel installed from locked runtime requirements, without the `service` extra | Pass. `microflow version` and `--help` work. `serve` refuses with an instruction to install the extra. |
| Pipeline benchmark, 256 and 512 px, 3 repeats | Ran. Result committed as a smoke measurement in `benchmarks/results/`. |
| History scan (all 19 commits) | No key-like strings. Largest blob 32 KB. Two findings are reported below. |

## What is implemented, verified, partial or unverified

| Area | Implemented | Verified here | Partial or unverified |
| --- | --- | --- | --- |
| Configuration, units, image loading | Yes | Full suite; module import test for every module | ImageJ calibration not read |
| Preprocessing and classical segmentation | Yes | Full suite; hand checks | Accuracy on real data not measured; single outlier pixels can dominate Otsu |
| Measurements | Yes | Full suite; analytic checks | Perimeter biased high by about 5 % for large discs |
| Quality control and figures | Yes | Full suite; figure bytes stable within this platform | Figures not reviewed by eye |
| Batch engine (parallel `spawn` on macOS) | Yes | Full suite, including two `workers=2` tests | Linux `fork` not run |
| Provenance, verification, export | Yes | Full suite; `verify-run` in the demonstration | Parquet written through pyarrow, tested here |
| Plugins | Yes | Full suite, including the example installed through `uv` | No timeout or memory limit; in-process |
| Machine learning | Yes | Full suite on CPU | CUDA path unverified; accuracy not measured |
| Statistics | Yes | Full suite; demonstration output | Not compared with an independent package; nested designs not modelled |
| HTTP service | Yes | Full suite, including HTTP API tests | Non-loopback use not tested; no body-size limit or throttling |
| Docker and compose | Written | Nothing | Not built or run: no container runtime was used |
| GitHub Actions workflows | Written | Their commands were run locally on macOS | Not run on GitHub; Linux not tested |
| Documentation | Yes | Links, command reference and example paths checked | Human review pending |
| Licensing | MIT LICENSE, PEP 639 metadata, inventory | Build metadata, inventory check, wheel contents | Redistribution duties for MPL-2.0 files are open (step 17) |

## Defects found and fixed in this pass

1. **The package could not be imported (critical).** `units.py` used `Area` in an annotation before
   defining it, so `import microflow.units` failed with `NameError`. That also broke the CLI. The
   earlier checks, including the static checker, missed it. Fixed by moving the class, and a test now
   imports every module in the package.
2. **ML evaluation did not detect changed held-out data.** It compared two identities that were both
   written at training time. It now also compares with the identity computed from the current images,
   masks and settings.
3. **Test expectations that were wrong, not code.** The OME unit test mixed units. The integrated
   intensity column includes its source and channel. A statistics helper skipped design path
   resolution. A hand-computed Hedges value was rounded wrongly. A numpy boolean was compared with `is`.
   A test image was constant, which normalisation correctly refuses. The CLI measure test lacked axes.
   A progress test read an attribute of the wrong type. A plugin fixture used a single saturated pixel
   that removed all objects, so the plugin never ran.
4. **Typing.** Strict mypy reported 72 issues. Code was corrected where it was wrong, and the relaxed
   checks for SciPy, pyarrow and scikit-image are declared in `pyproject.toml`.
5. **Lint.** Ruff reported 28 issues. All are fixed.

## Still open before publication

- **Commit history contains a personal local path.** Commit `fe84d30` put the absolute path of the
  owner's machine, which includes their user name, into `AGENTS.md`. Commit `939f590` removed it from the
  current file, but the history still holds it. This was not rewritten here, because the history is the
  owner's to decide on. See step 1.
- **All 19 commits use a placeholder identity.** Author and committer are
  `OpenCode Project Agent <opencode-agent@localhost>`. It is not a personal address, but GitHub cannot
  link it to an account. See step 1.

## Future steps

Each step gives scope, rationale, dependencies, acceptance criteria and an effort estimate in hours.

### 1. Decide the publication identity and whether to rewrite the history

- **Scope:** choose the author identity for public history, and decide whether to remove the local path
  from commit `fe84d30`. Any rewrite is a separate, coordinated action.
- **Rationale:** both findings are visible to anyone who clones the repository.
- **Dependencies:** the owner's decision. Requires that no other clone is left behind to diverge.
- **Acceptance criteria:** `git log --all -p` finds no personal path; the author identity is one the
  owner chooses; the decision is recorded here.
- **Effort:** 1–6 hours (1 hour to decide; 2–5 hours for a coordinated rewrite and verification).

### 2. Run the GitHub Actions workflows on Linux and fix platform differences

- **Scope:** push the repository, run `ci.yml` on Linux runners, and fix whatever differs: the CPU PyTorch
  index, the `fork` start method, and OpenMP behaviour. Decide whether the license inventory check belongs
  in CI. Its output depends on which platform-specific packages are installed, so it can differ on Linux.
- **Rationale:** every CI job has been reproduced only on macOS.
- **Dependencies:** step 1; a GitHub repository.
- **Acceptance criteria:** the `test`, `reference`, `ml`, `service` and `demo` jobs pass on both Python
  versions, and the results are recorded in `docs/installation.md`.
- **Effort:** 6–10 hours.

### 3. Build and run the Docker image and the compose file

- **Scope:** on a machine with a Docker host, build the `Dockerfile` from a clean checkout, start
  `compose.yaml`, check `GET /v1/health`, submit a job, and confirm the read-only root filesystem allows
  only the documented writes. Add a `HEALTHCHECK`.
- **Rationale:** the image is written but has never been built, and the container paths are untested.
- **Dependencies:** a Docker host outside this environment; step 1 for the committed history.
- **Acceptance criteria:** the build succeeds, the health check passes, a job completes, and
  `docs/docker.md` records the results.
- **Effort:** 4–6 hours.

### 4. Pin the build backend and developer tools

- **Scope:** pin `hatchling` (currently `>=1.27`, which PEP 639 requires) and the tools in the dev group
  to exact versions, and record the pins in `docs/installation.md`.
- **Rationale:** the build backend is not locked, so a build can change without a commit.
- **Dependencies:** none.
- **Acceptance criteria:** `uv build` gives identical metadata on two runs with the same lock.
- **Effort:** 1–2 hours.

### 5. Measure coverage in CI with a recorded baseline

- **Scope:** add `pytest-cov` to the dev group, report coverage in CI, and set a floor at the measured value.
- **Rationale:** the suite is large, but no coverage number has ever been recorded.
- **Dependencies:** step 2.
- **Acceptance criteria:** the coverage percentage is recorded here, and CI fails below the floor.
- **Effort:** 2–3 hours.

### 6. Isolate plugin execution

- **Scope:** run each plugin call with a wall-clock timeout and a memory limit, or in a subprocess mode
  that is opt-in. A failing plugin must fail only its item.
- **Rationale:** a plugin currently runs inside the batch process with full privileges and no limit.
- **Dependencies:** none; touches the batch executor.
- **Acceptance criteria:** tests show a sleeping plugin and a memory-hungry plugin each fail only their own
  item, and `docs/plugins.md` describes the behaviour on each platform.
- **Effort:** 12–20 hours.

### 7. Harden the service for non-loopback deployment

- **Scope:** add a request-size limit (413), throttling after repeated failed tokens (429), a log of
  accepted and refused requests that never records the token, and token reloading without a restart.
- **Rationale:** `--allow-remote` is supported, but these protections are missing.
- **Dependencies:** step 3 for a realistic deployment test.
- **Acceptance criteria:** tests for each protection pass, and a reverse-proxy example is run once.
- **Effort:** 8–14 hours.

### 8. Replace the relaxed typing with real stubs

- **Scope:** add maintained type stubs for SciPy and pyarrow where they exist, type the scikit-image
  calls that the code depends on, and remove the module-level relaxations in `pyproject.toml`.
- **Rationale:** the relaxations let untyped values through in the numerical modules.
- **Dependencies:** step 4 for pinned stub versions.
- **Acceptance criteria:** `mypy` passes with the relaxations removed from the listed modules.
- **Effort:** 6–10 hours.

### 9. Make global thresholding robust to outlier pixels

- **Scope:** mask saturated and invalid pixels before thresholding, and offer a robust threshold option.
  Add tests on images with single hot pixels.
- **Rationale:** one pixel at 60000 moved Otsu's threshold from 106 to 3025 and removed real objects.
- **Dependencies:** none.
- **Acceptance criteria:** the single-hot-pixel image keeps its objects with the new option, and the
  default behaviour is documented if it changes.
- **Effort:** 6–10 hours.

### 10. Correct or document the perimeter bias

- **Scope:** evaluate a bias-corrected perimeter estimator (for example Crofton or Kulpa) on discs and
  squares across radii, then decide between replacing the estimator and publishing a correction factor.
- **Rationale:** the iso-contour perimeter is about 5 % high for large discs and 7 % at radius 20 pixels.
- **Dependencies:** step 9 is independent; the measurement definitions must be revised with the change.
- **Acceptance criteria:** the chosen estimator's error at several radii is recorded in
  `docs/measurements.md`, and the tests pin it.
- **Effort:** 6–10 hours.

### 11. Run a benchmark campaign

- **Scope:** add a batch-level benchmark, run the pipeline benchmark at several sizes and repeats on
  macOS and Linux, and report the variance.
- **Rationale:** the committed result has three repeats on one machine, so it shows no variance.
- **Dependencies:** step 2 for the Linux runs.
- **Acceptance criteria:** results are committed with hardware, software versions and repeat counts, and
  the variance is reported.
- **Effort:** 8–14 hours.

### 12. Validate segmentation on real annotated images

- **Scope:** obtain a microscopy dataset whose licence permits this use, or use the laboratory's own
  data. Compare classical methods and the ML plugin under the same contracts, and report Dice and IoU with
  intervals, per image and per sample.
- **Rationale:** every score so far comes from noiseless synthetic phantoms.
- **Dependencies:** data access, which is outside the project's control; step 9 may be needed first.
- **Acceptance criteria:** an evaluation report is committed with its split rules and failure examples, and
  thresholds are chosen on a validation split only.
- **Effort:** 16–30 hours, plus 40 hours or more if the images must be annotated.

### 13. Cross-check the statistics and add nested designs

- **Scope:** compare the Welch, Mann-Whitney, permutation, bootstrap and Hedges' g values with an
  independent implementation, using stated tolerances. Add mixed-effects or nested-design support, or refuse
  nested designs with a clear message.
- **Rationale:** nested designs would currently be analysed as independent samples.
- **Dependencies:** step 8 is not required; the cross-check needs an independent reference tool.
- **Acceptance criteria:** the cross-check passes within tolerance in CI, and nested designs are either
  modelled or refused.
- **Effort:** 12–20 hours.

### 14. Extend the ML module and run the CUDA checklist

- **Scope:** add multichannel and three-dimensional input, early stopping, and resume from `last.pt`.
  Run the CUDA checklist on a machine with a GPU, and record the results.
- **Rationale:** these are the stated limits in `docs/machine-learning.md`. The CUDA path has never run.
- **Dependencies:** a GPU machine for the checklist, which is a manual step.
- **Acceptance criteria:** each feature has CPU tests, a resumed run matches an uninterrupted run with the
  same seed on CPU, and the GPU checklist is recorded.
- **Effort:** 16–24 hours.

### 15. Review every figure and the HTML report by eye

- **Scope:** inspect each QC figure, statistics plot and the report at print size. Check label overlap,
  axis units, colour choices for colour-vision deficiency, and the empty-result case.
- **Rationale:** plots are the main output, and no automated check catches overlapping labels.
- **Dependencies:** step 2 provides the demonstration output on Linux as well.
- **Acceptance criteria:** the findings and fixes are committed, and structural tests cover each axis label.
- **Effort:** 4–8 hours.

### 16. Validate batch configurations from the command line

- **Scope:** add `microflow validate-batch` with the same checks as the service's validation, and correct
  the user guide to show it.
- **Rationale:** `validate-config` rejects batch files, and the user guide previously said it did not.
- **Dependencies:** none.
- **Acceptance criteria:** exit status 0 for a valid file and 1 for an invalid one, with the field named.
- **Effort:** 2–3 hours.

### 17. Clear the redistribution duties for the container image

- **Scope:** confirm which packages reach the image, and whether `certifi` or `pathspec` are among them.
  Where MPL-2.0 files are included, provide their source as the license requires, and record a notice policy.
- **Rationale:** `THIRD_PARTY_LICENSES.md` lists two MPL-2.0 packages. Their duties apply to binaries.
- **Dependencies:** step 3 (the image's real contents).
- **Acceptance criteria:** the image's package list is checked against the inventory, and the notice policy is
  written in `docs/docker.md`.
- **Effort:** 2–4 hours.

### 18. Release engineering

- **Scope:** tag `v0.1.0`, write a changelog, and build the release wheel and sdist in CI. Optionally sign
  the artifacts.
- **Rationale:** users install a built artifact, not the repository.
- **Dependencies:** steps 1, 2 and 4.
- **Acceptance criteria:** the tag, changelog and CI-built artifacts exist, and the installed CLI passes the
  smoke test from the wheel.
- **Effort:** 3–6 hours.

### 19. Decide the placeholder `microflow.pipelines` package

- **Scope:** implement a small, tested composition function that calls the same code as the CLI, or delete
  the package and its architecture row.
- **Rationale:** an empty public package misleads users.
- **Dependencies:** none.
- **Acceptance criteria:** the package has a tested public function, or it is removed and the docs match.
- **Effort:** 1–8 hours.

### 20. Read ImageJ calibration and remove the duplicate unit helper

- **Scope:** read ImageJ pixel calibration from its TIFF metadata, or document it as unsupported. Replace the
  duplicate `_pixels` helper in `preprocessing/pipeline.py` with `microflow.units.to_pixels`.
- **Rationale:** an unread calibration silently changes measured units, and duplicated conversion can drift.
- **Dependencies:** none.
- **Acceptance criteria:** one unit-conversion path, and ImageJ calibration is read and tested or documented.
- **Effort:** 2–4 hours.

### 21. Tidy the sandbox-specific environment messages

- **Scope:** the OpenMP warning about `/tmp` appears because the sandbox blocks writes there. Document the
  warning, and check whether setting `OMP_NUM_THREADS` or a project-local temporary path removes it.
- **Rationale:** the warning is noise for contributors and could be mistaken for a fault.
- **Dependencies:** none.
- **Acceptance criteria:** the warning is either gone or explained in `docs/installation.md`.
- **Effort:** 1–2 hours.

## Effort summary

| Priority | Steps | Low (hours) | High (hours) |
| --- | --- | --- | --- |
| P0 | 1, 6, 7, 9 | 27 | 50 |
| P1 | 2, 3, 4, 5, 8, 10, 11, 15, 16, 17, 18 | 44 | 76 |
| P2 | 12, 13, 14 | 44 | 74 |
| P3 | 19, 20, 21 | 4 | 14 |
| **Total** | 1–21 | **119** | **214** |

Assumptions behind the total:

- One engineer who knows Python, the scientific stack and this code.
- Access to a GitHub repository, a Linux runner, a Docker host and a GPU machine. Steps 3 and 14 need
  hardware this environment cannot provide.
- Step 12 assumes a suitable dataset is available. If annotation is needed, add 40 hours or more.
- Estimates exclude CI queue time, review cycles and the owner's decisions. They have not been measured
  against this codebase's real rate.

At 35 to 40 hours a week, the total is about 3 to 6 engineer-weeks.

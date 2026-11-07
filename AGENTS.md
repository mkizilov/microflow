# MicroFlow: project boundaries and working agreement

- Work only in `/Users/mk-rune/test_projects/04-microflow`. This is one of fourteen independent repositories.
- Do not read, modify, clean up, or run commands in sibling repositories or the user's configuration.
- Work autonomously: choose sensible defaults and document assumptions; do not ask questions or delegate.
- Never install anything globally or use sudo, Homebrew, launchd, host credentials, or host Docker sockets.
- Dependencies, toolchains, caches, browser binaries, SDKs, temporary files and test data stay in this repository. Preserve the supplied project-local environment. Python packages go in `.venv`.
- `.agent-local/` is ignored, project-local build/runtime state. `.opencode/` is managed orchestration configuration; do not change or remove it or bypass the sandbox.
- Use existing system tools read-only, or install pinned tools under this repository. Do not change shell profiles, user settings, Keychain, or global Git settings.
- Bind test servers to loopback; use ports 21300–21399. Limit build/test workers to two unless a required benchmark explicitly needs more.
- Docker-dependent integration tests must remain reproducible but be reported as unverified here if they require access outside this repository. Do not bypass this boundary.
- Do not touch attached hardware, enroll services, flash firmware, install agents, or alter OS settings. Supply manual checklists for such steps.
- Implement only the current numbered prompt, preserving prior behavior. Run appropriate checks, fix failures, and report genuine limitations without fabricating test or benchmark results.
- Before finishing every numbered turn: stop its servers/watchers, remove scratch outputs, update relevant documentation, and make a local conventional-commit Git commit. Do not push.
- End each turn with a short summary of changes, checks actually run, remaining limitations, and commit hash. The orchestrator will provide the next prompt only after this turn ends.

Extend this file with the project's language-specific architecture and verification commands while retaining these boundaries.

## Language and tooling

- Python 3.12+, managed with `uv`. Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`.
- Source lives in `src/microflow/`; tests in `tests/`; docs in `docs/`.
- Verification commands (run from the repository root):
  - `uv sync --locked`
  - `uv run ruff check .` and `uv run ruff format --check .`
  - `uv run mypy`
  - `uv run pytest`
  - `uv run microflow make-sample-data --out-dir data/synthetic`

## Engineering rules for this project

- **Deterministic tests:** every random step takes an explicit seed; tests must not depend on wall-clock time, network, or ordering of unseeded sources.
- **Scientific units:** every quantitative value carries a unit. Calibration requires an explicit length unit (`microflow.units.Calibration`); never silently assume pixels or micrometres.
- **Isolated dependencies:** install only through `uv` into `.venv/` inside this repository. Do not install packages globally or into the system Python.
- **Clean temporary files:** tests write only under pytest's `tmp_path`; generated data goes to `data/synthetic/` (git-ignored) and is removed when no longer needed.
- **Scope:** analysis algorithms (preprocessing, segmentation, quantification) are added only by their own numbered task. The current subpackages are placeholders.

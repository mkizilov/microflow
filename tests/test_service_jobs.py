from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from microflow.batch import BatchError, load_batch_config
from microflow.config import ConfigError
from microflow.service import jobs as jobs_module
from microflow.service.errors import ServiceError
from microflow.service.jobs import INTERRUPTED_MESSAGE, JobManager, RunControl, make_batch_runner
from microflow.service.settings import SERVICE_WORKERS, Layout, prepare_layout
from microflow.service.store import JobRow, JobStore
from microflow.service.validation import PreparedBatch, prepare_batch
from service_fixtures import (
    FakeRunner,
    Tree,
    batch_mapping,
    make_tree,
    settings_for,
    wait_for,
    write_batch_config,
)


@dataclass
class Harness:
    tree: Tree
    layout: Layout
    store: JobStore
    manager: JobManager
    runner: FakeRunner

    def submit(self, name: str = "batch.yaml", output: str = "run-1", **mapping: Any) -> JobRow:
        config = batch_mapping(self.tree, output=output, **mapping)
        write_batch_config(self.tree.inputs, config, name)
        return self.submit_existing(name)

    def submit_existing(self, name: str) -> JobRow:
        path = self.tree.inputs / name
        prepared = prepare_batch(str(path), self.layout)
        return self.manager.submit(
            kind="batch",
            prepared=prepared,
            recheck=lambda: prepare_batch(str(path), self.layout),
        )

    def status(self, job: JobRow) -> str:
        return self.manager.get(job.id).status

    def close(self) -> None:
        self.manager.shutdown()
        self.store.close()


def _harness(tmp_path: Path, runner: FakeRunner, **overrides: Any) -> Harness:
    tree = make_tree(tmp_path)
    settings = settings_for(tree, **overrides)
    layout = prepare_layout(settings)
    store = JobStore(layout.state_path)
    manager = JobManager(settings, store, runner)
    return Harness(tree=tree, layout=layout, store=store, manager=manager, runner=runner)


def test_a_job_runs_to_success_and_records_its_counts(tmp_path: Path) -> None:
    h = _harness(tmp_path, FakeRunner())
    try:
        job = h.submit()
        wait_for(lambda: h.status(job) == "succeeded")
        final = h.manager.get(job.id)
        assert final.exit_code == 0
        assert final.summary == {
            "counts": {"ok": 1, "failed": 0, "cancelled": 0, "resumed": 0},
            "items": 1,
            "workers": 1,
        }
        assert final.started_utc is not None and final.finished_utc is not None
        assert final.error_code is None
    finally:
        h.close()


def test_a_run_with_failed_images_is_reported_with_exit_code_3(tmp_path: Path) -> None:
    h = _harness(tmp_path, FakeRunner(status="completed_with_failures"))
    try:
        job = h.submit()
        wait_for(lambda: h.status(job) == "completed_with_failures")
        assert h.manager.get(job.id).exit_code == 3
    finally:
        h.close()


def test_an_engine_refusal_fails_the_job_with_its_code_and_message(tmp_path: Path) -> None:
    h = _harness(tmp_path, FakeRunner(error=BatchError("the output directory is busy")))
    try:
        job = h.submit()
        wait_for(lambda: h.status(job) == "failed")
        final = h.manager.get(job.id)
        assert final.exit_code == 1
        assert final.error_code == "batch_refused"
        assert final.error_message == "the output directory is busy"
    finally:
        h.close()


def test_an_unexpected_exception_is_reported_generically_and_logged_not_leaked(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    h = _harness(tmp_path, FakeRunner(error=RuntimeError("secret detail at /root/key")))
    try:
        job = h.submit()
        wait_for(lambda: h.status(job) == "failed")
        final = h.manager.get(job.id)
        assert final.error_code == "internal"
        assert final.error_message == jobs_module.INTERNAL_MESSAGE
        assert "secret detail" not in (final.error_message or "")
        # The full error is kept in the server log, which is where operators look for it.
        assert "failed unexpectedly" in caplog.text
        assert "secret detail" in caplog.text
    finally:
        h.close()


def test_a_running_job_can_be_cancelled_and_ends_with_exit_code_130(tmp_path: Path) -> None:
    h = _harness(tmp_path, FakeRunner(hold=True))
    try:
        job = h.submit()
        wait_for(lambda: h.status(job) == "running")
        assert h.manager.cancel(job.id).cancel_requested
        wait_for(lambda: h.status(job) == "cancelled")
        final = h.manager.get(job.id)
        assert final.exit_code == 130
        assert final.error_code is None
    finally:
        h.runner.release()
        h.close()


def test_a_queued_job_is_cancelled_without_ever_starting(tmp_path: Path) -> None:
    h = _harness(tmp_path, FakeRunner(hold=True), max_concurrent_jobs=1, max_queued_jobs=2)
    try:
        first = h.submit("first.yaml", "run-1")
        wait_for(lambda: h.status(first) == "running")
        second = h.submit("second.yaml", "run-2")
        assert h.status(second) == "queued"
        cancelled = h.manager.cancel(second.id)
        assert cancelled.status == "cancelled"
        assert cancelled.started_utc is None
        assert cancelled.exit_code == 130
        h.runner.release()
        wait_for(lambda: h.status(first) == "succeeded")
        h.manager.shutdown()
        assert h.runner.calls == 1
    finally:
        h.runner.release()
        h.close()


def test_a_finished_job_cannot_be_cancelled(tmp_path: Path) -> None:
    h = _harness(tmp_path, FakeRunner())
    try:
        job = h.submit()
        wait_for(lambda: h.status(job) == "succeeded")
        with pytest.raises(ServiceError) as refused:
            h.manager.cancel(job.id)
        assert refused.value.code == "job_finished"
        assert "already succeeded" in refused.value.message
    finally:
        h.close()


def test_an_unknown_job_is_not_found(tmp_path: Path) -> None:
    h = _harness(tmp_path, FakeRunner())
    try:
        with pytest.raises(ServiceError) as missing:
            h.manager.get("does-not-exist")
        assert missing.value.code == "not_found"
        with pytest.raises(ServiceError) as unknown:
            h.manager.cancel("does-not-exist")
        assert unknown.value.code == "not_found"
    finally:
        h.close()


def test_capacity_refuses_work_beyond_what_can_run_and_wait(tmp_path: Path) -> None:
    h = _harness(tmp_path, FakeRunner(hold=True), max_concurrent_jobs=1, max_queued_jobs=1)
    try:
        first = h.submit("a.yaml", "run-1")
        wait_for(lambda: h.status(first) == "running")
        second = h.submit("b.yaml", "run-2")
        with pytest.raises(ServiceError) as refused:
            h.submit("c.yaml", "run-3")
        assert refused.value.code == "queue_full"
        assert "limit is 2" in refused.value.message
        h.runner.release()
        wait_for(lambda: h.status(first) == "succeeded" and h.status(second) == "succeeded")
    finally:
        h.runner.release()
        h.close()


def test_no_more_jobs_run_at_once_than_the_concurrency_limit(tmp_path: Path) -> None:
    h = _harness(tmp_path, FakeRunner(hold=True), max_concurrent_jobs=2, max_queued_jobs=2)
    try:
        jobs = [h.submit(f"job{index}.yaml", f"run-{index}") for index in range(4)]
        wait_for(lambda: h.runner.calls == 2)
        statuses = [h.status(job) for job in jobs]
        assert statuses.count("running") == 2
        assert statuses.count("queued") == 2
        assert h.runner.peak == 2
        h.runner.release()
        wait_for(lambda: all(h.status(job) == "succeeded" for job in jobs))
        assert h.runner.peak == 2
        assert h.runner.calls == 4
    finally:
        h.runner.release()
        h.close()


def test_progress_is_recorded_while_a_job_runs(tmp_path: Path) -> None:
    h = _harness(tmp_path, FakeRunner(hold=True))
    try:
        job = h.submit()
        wait_for(lambda: h.status(job) == "running")
        wait_for(lambda: h.manager.get(job.id).progress is not None)
        progress = h.store.get(job.id)
        assert progress is not None
        assert (progress.progress_completed, progress.progress_total) == (1, 2)
    finally:
        h.runner.release()
        h.close()


def test_shutdown_cancels_running_jobs_and_marks_queued_ones_interrupted(tmp_path: Path) -> None:
    h = _harness(tmp_path, FakeRunner(hold=True), max_concurrent_jobs=1, max_queued_jobs=1)
    try:
        first = h.submit("a.yaml", "run-1")
        wait_for(lambda: h.status(first) == "running")
        second = h.submit("b.yaml", "run-2")
        h.manager.shutdown()
        assert h.status(first) == "cancelled"
        queued = h.manager.get(second.id)
        assert queued.status == "interrupted"
        assert queued.error_code == "interrupted"
        assert queued.error_message == INTERRUPTED_MESSAGE
        with pytest.raises(ServiceError) as closed:
            h.submit("c.yaml", "run-3")
        assert closed.value.code == "unavailable"
    finally:
        h.runner.release()
        h.store.close()


def test_finished_jobs_survive_a_restart(tmp_path: Path) -> None:
    h = _harness(tmp_path, FakeRunner())
    job = h.submit()
    wait_for(lambda: h.status(job) == "succeeded")
    h.close()
    store = JobStore(h.layout.state_path)
    manager = JobManager(settings_for(h.tree), store, FakeRunner())
    try:
        assert manager.interrupted == 0
        again = manager.get(job.id)
        assert again.status == "succeeded"
        assert again.summary is not None and again.summary["items"] == 1
    finally:
        manager.shutdown()
        store.close()


def test_a_job_left_running_by_a_crash_is_marked_interrupted_on_restart(tmp_path: Path) -> None:
    tree = make_tree(tmp_path)
    layout = prepare_layout(settings_for(tree))
    store = JobStore(layout.state_path)
    store.insert(
        JobRow(
            id="crashed",
            kind="batch",
            status="queued",
            config_name="batch.yaml",
            config_path=str(tree.inputs / "batch.yaml"),
            output_name="output/run-1",
            output_directory=str(tree.outputs / "run-1"),
            submitted_utc="2026-01-01T00:00:00+00:00",
        )
    )
    assert store.claim("crashed", "2026-01-01T00:00:01+00:00")
    store.close()
    reopened = JobStore(layout.state_path)
    manager = JobManager(settings_for(tree), reopened, FakeRunner())
    try:
        assert manager.interrupted == 1
        job = manager.get("crashed")
        assert job.status == "interrupted"
        assert job.error_code == "interrupted"
    finally:
        manager.shutdown()
        reopened.close()


def test_a_job_whose_output_directory_changed_after_submission_fails_rather_than_running(
    tmp_path: Path,
) -> None:
    h = _harness(tmp_path, FakeRunner())
    try:
        write_batch_config(h.tree.inputs, batch_mapping(h.tree, output="run-1"), "b.yaml")
        prepared = prepare_batch(str(h.tree.inputs / "b.yaml"), h.layout)
        write_batch_config(h.tree.inputs, batch_mapping(h.tree, output="run-2"), "b.yaml")
        job = h.manager.submit(
            kind="batch",
            prepared=prepared,
            recheck=lambda: prepare_batch(str(h.tree.inputs / "b.yaml"), h.layout),
        )
        wait_for(lambda: h.status(job) == "failed")
        assert h.manager.get(job.id).error_code == "invalid_config"
        assert h.runner.calls == 0
    finally:
        h.close()


def test_a_configuration_that_became_invalid_fails_without_running(tmp_path: Path) -> None:
    h = _harness(tmp_path, FakeRunner())
    try:
        write_batch_config(h.tree.inputs, batch_mapping(h.tree), "b.yaml")
        prepared = prepare_batch(str(h.tree.inputs / "b.yaml"), h.layout)

        def broken() -> PreparedBatch:
            raise ConfigError("b.yaml: dataset: field required")

        job = h.manager.submit(kind="batch", prepared=prepared, recheck=broken)
        wait_for(lambda: h.status(job) == "failed")
        assert h.manager.get(job.id).error_code == "invalid_config"
        assert h.runner.calls == 0
    finally:
        h.close()


def test_the_batch_runner_caps_each_run_at_the_service_worker_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}

    def fake_run_batch(config: object, *, progress: object, cancel: object, workers: int) -> object:
        seen["workers"] = workers
        return SimpleNamespace(status="complete", counts={"ok": 0}, items=(), workers=workers)

    monkeypatch.setattr(jobs_module, "run_batch", fake_run_batch)
    tree = make_tree(tmp_path)
    write_batch_config(tree.inputs, batch_mapping(tree, workers=4), "b.yaml")
    config = load_batch_config(tree.inputs / "b.yaml")
    control = RunControl(cancelled=lambda: False, progress=lambda done, total: None)
    outcome = make_batch_runner()(config, control)
    assert seen["workers"] == SERVICE_WORKERS == 1
    assert outcome.status == "complete"

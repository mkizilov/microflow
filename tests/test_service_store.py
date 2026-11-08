import threading
from pathlib import Path

from microflow.service.store import JobRow, JobStore

NOW = "2026-01-01T00:00:00+00:00"


def _job(job_id: str, status: str = "queued") -> JobRow:
    return JobRow(
        id=job_id,
        kind="batch",
        status=status,
        config_name="batch.yaml",
        config_path="/inputs/batch.yaml",
        output_name="output/run",
        output_directory="/outputs/run",
        submitted_utc=NOW,
    )


def test_a_job_round_trips_through_the_database(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    store.insert(_job("one"))
    row = store.get("one")
    assert row is not None
    assert row.status == "queued"
    assert row.config_name == "batch.yaml"
    assert row.summary is None
    assert row.cancel_requested is False
    assert store.get("absent") is None
    store.close()


def test_records_survive_closing_and_reopening_the_database(tmp_path: Path) -> None:
    path = tmp_path / "jobs.sqlite"
    store = JobStore(path)
    store.insert(_job("kept"))
    assert store.claim("kept", NOW)
    assert store.finish(
        "kept",
        status="succeeded",
        exit_code=0,
        error_code=None,
        error_message=None,
        summary={"counts": {"ok": 2}},
        finished_utc=NOW,
    )
    store.close()
    reopened = JobStore(path)
    row = reopened.get("kept")
    assert row is not None
    assert row.status == "succeeded"
    assert row.summary == {"counts": {"ok": 2}}
    reopened.close()


def test_only_one_thread_can_claim_a_queued_job(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    store.insert(_job("contested"))
    results: list[bool] = []
    lock = threading.Lock()

    def attempt() -> None:
        won = store.claim("contested", NOW)
        with lock:
            results.append(won)

    threads = [threading.Thread(target=attempt) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results.count(True) == 1
    assert results.count(False) == 15
    store.close()


def test_each_transition_applies_only_from_the_state_it_names(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    store.insert(_job("queued-job"))
    assert not store.finish(
        "queued-job",
        status="succeeded",
        exit_code=0,
        error_code=None,
        error_message=None,
        summary=None,
        finished_utc=NOW,
    )
    assert not store.request_cancel("queued-job")
    assert not store.update_progress("queued-job", 1, 2)
    assert store.cancel_queued("queued-job", NOW)
    assert not store.cancel_queued("queued-job", NOW)
    assert not store.claim("queued-job", NOW)
    cancelled = store.get("queued-job")
    assert cancelled is not None
    assert cancelled.status == "cancelled"
    assert cancelled.exit_code == 130
    store.close()


def test_a_running_job_accepts_cancellation_and_progress(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    store.insert(_job("running-job"))
    assert store.claim("running-job", NOW)
    assert store.update_progress("running-job", 1, 4)
    assert store.request_cancel("running-job")
    row = store.get("running-job")
    assert row is not None
    assert row.progress_completed == 1 and row.progress_total == 4
    assert row.cancel_requested
    store.close()


def test_listing_is_newest_first_filtered_and_limited(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    store.insert(_job("first"))
    store.insert(_job("second"))
    store.insert(_job("third"))
    assert store.claim("first", NOW)
    everything = store.list_jobs(statuses=None, limit=10)
    assert [row.id for row in everything] == ["third", "second", "first"]
    assert [row.id for row in store.list_jobs(statuses=("running",), limit=10)] == ["first"]
    assert [row.id for row in store.list_jobs(statuses=None, limit=2)] == ["third", "second"]
    store.close()


def test_the_active_count_covers_queued_and_running_jobs_only(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    store.insert(_job("a"))
    store.insert(_job("b"))
    store.insert(_job("c"))
    assert store.claim("a", NOW)
    assert store.cancel_queued("b", NOW)
    assert store.count_active() == 2
    store.close()


def test_marking_interrupted_touches_only_unfinished_jobs(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    store.insert(_job("done"))
    store.insert(_job("waiting"))
    store.insert(_job("working"))
    assert store.claim("done", NOW)
    assert store.finish(
        "done",
        status="succeeded",
        exit_code=0,
        error_code=None,
        error_message=None,
        summary=None,
        finished_utc=NOW,
    )
    assert store.claim("working", NOW)
    assert store.mark_interrupted(NOW, "stopped") == 2
    done = store.get("done")
    assert done is not None and done.status == "succeeded"
    interrupted = store.get("working")
    assert interrupted is not None
    assert interrupted.status == "interrupted"
    assert interrupted.error_message == "stopped"
    store.close()

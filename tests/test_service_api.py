"""HTTP API tests, with the real lifespan and routes.

They need the optional service extra (FastAPI and httpx) and run in the CI ``service`` job. They are
skipped elsewhere.
"""

import os
import shutil
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytestmark = pytest.mark.service

from fastapi.testclient import TestClient  # noqa: E402

from microflow.config import ConfigError  # noqa: E402
from microflow.image_io import write_image  # noqa: E402
from microflow.sample_data import Disc, make_phantom  # noqa: E402
from microflow.service.app import create_app  # noqa: E402
from microflow.service.errors import ServiceError  # noqa: E402
from microflow.service.jobs import Runner  # noqa: E402
from service_fixtures import (  # noqa: E402
    TOKEN,
    FakeRunner,
    Tree,
    batch_mapping,
    make_tree,
    settings_for,
    write_batch_config,
)

AUTH = {"Authorization": f"Bearer {TOKEN}"}
TERMINAL = ("succeeded", "completed_with_failures", "cancelled", "failed", "interrupted")


@contextmanager
def running(tree: Tree, runner: Runner | None = None, **overrides: Any) -> Iterator[TestClient]:
    app = create_app(settings_for(tree, **overrides), token=TOKEN, runner=runner)
    with TestClient(app) as client:
        yield client


def _validate(client: TestClient, kind: str, path: str) -> Any:
    return client.post("/v1/validate", json={"kind": kind, "path": path}, headers=AUTH)


def _submit(
    client: TestClient, tree: Tree, name: str = "batch.yaml", output: str = "run-1"
) -> dict[str, Any]:
    write_batch_config(tree.inputs, batch_mapping(tree, output=output), name)
    response = client.post(
        "/v1/jobs", json={"kind": "batch", "path": str(tree.inputs / name)}, headers=AUTH
    )
    assert response.status_code == 202, response.text
    body: dict[str, Any] = response.json()
    return body


def _job(client: TestClient, job_id: str) -> dict[str, Any]:
    body: dict[str, Any] = client.get(f"/v1/jobs/{job_id}", headers=AUTH).json()
    return body


def _await(
    client: TestClient, job_id: str, statuses: tuple[str, ...], timeout: float = 30.0
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while True:
        body = _job(client, job_id)
        if body["status"] in statuses:
            return body
        if time.monotonic() > deadline:
            raise AssertionError(f"job stayed {body['status']!r}")
        time.sleep(0.02)


def test_health_needs_no_token(tmp_path: Path) -> None:
    with running(make_tree(tmp_path)) as client:
        response = client.get("/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


PROTECTED: list[tuple[str, str, dict[str, Any] | None]] = [
    ("POST", "/v1/validate", {"kind": "batch", "path": "/x"}),
    ("POST", "/v1/jobs", {"kind": "batch", "path": "/x"}),
    ("GET", "/v1/jobs", None),
    ("GET", "/v1/jobs/abc", None),
    ("POST", "/v1/jobs/abc/cancel", None),
    ("GET", "/v1/analyses", None),
    ("GET", "/v1/analyses/abc/files", None),
    ("GET", "/v1/analyses/abc/files/x.csv", None),
    ("GET", "/v1/analyses/abc/qc", None),
]


@pytest.mark.parametrize(("method", "path", "body"), PROTECTED)
def test_every_protected_route_refuses_requests_without_the_token(
    tmp_path: Path, method: str, path: str, body: dict[str, Any] | None
) -> None:
    with running(make_tree(tmp_path)) as client:
        response = client.request(method, path, json=body)
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize(
    "header",
    ["Bearer " + "w" * 40, "Basic dXNlcjpwYXNz", f"Token {TOKEN}", "Bearer"],
)
def test_wrong_or_malformed_credentials_are_refused(tmp_path: Path, header: str) -> None:
    with running(make_tree(tmp_path)) as client:
        response = client.get("/v1/jobs", headers={"Authorization": header})
    assert response.status_code == 401


def test_the_scheme_is_case_insensitive_and_the_token_must_match_exactly(tmp_path: Path) -> None:
    with running(make_tree(tmp_path)) as client:
        accepted = client.get("/v1/jobs", headers={"Authorization": f"bearer {TOKEN}"})
        near_miss = client.get("/v1/jobs", headers={"Authorization": f"Bearer {TOKEN[:-1]}x"})
    assert accepted.status_code == 200
    assert near_miss.status_code == 401


def test_interactive_docs_and_the_schema_are_not_published(tmp_path: Path) -> None:
    with running(make_tree(tmp_path)) as client:
        assert client.get("/docs").status_code == 404
        assert client.get("/openapi.json").status_code == 404


def test_validation_accepts_a_good_batch_and_describes_it(tmp_path: Path) -> None:
    tree = make_tree(tmp_path)
    write_batch_config(tree.inputs, batch_mapping(tree), "batch.yaml")
    with running(tree) as client:
        response = _validate(client, "batch", str(tree.inputs / "batch.yaml"))
    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is True
    assert body["summary"] == {
        "name": "service-test",
        "images": 2,
        "output": "output/run-1",
        "workers": 1,
    }
    assert body["warnings"] == []


def test_validation_refuses_configurations_outside_the_input_roots(tmp_path: Path) -> None:
    tree = make_tree(tmp_path)
    elsewhere = tree.root / "elsewhere"
    elsewhere.mkdir()
    write_batch_config(elsewhere, batch_mapping(tree), "batch.yaml")
    with running(tree) as client:
        response = _validate(client, "batch", str(elsewhere / "batch.yaml"))
    assert response.status_code == 403
    error = response.json()["error"]
    assert error["code"] == "path_not_allowed"
    assert str(tree.root) not in error["message"]


def test_validation_reports_invalid_configurations_with_their_details(tmp_path: Path) -> None:
    tree = make_tree(tmp_path)
    write_batch_config(tree.inputs, batch_mapping(tree, bogus=1), "batch.yaml")
    with running(tree) as client:
        response = _validate(client, "batch", str(tree.inputs / "batch.yaml"))
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "invalid_config"
    assert any("bogus" in detail for detail in error["details"])


def test_relative_paths_and_malformed_bodies_get_clear_errors(tmp_path: Path) -> None:
    tree = make_tree(tmp_path)
    with running(tree) as client:
        relative = _validate(client, "batch", "batch.yaml")
        unknown_kind = _validate(client, "other", "/x")
        extra_field = client.post(
            "/v1/validate", json={"kind": "batch", "path": "/x", "shell": "rm"}, headers=AUTH
        )
    assert relative.status_code == 400
    assert relative.json()["error"]["code"] == "bad_request"
    assert unknown_kind.status_code == 422
    assert unknown_kind.json()["error"]["code"] == "invalid_request"
    assert extra_field.status_code == 422
    assert any("shell" in detail for detail in extra_field.json()["error"]["details"])


def test_a_statistics_design_can_be_validated_over_the_api(tmp_path: Path) -> None:
    from stats_fixtures import design_mapping, write_design, write_dictionary, write_measurements

    tree = make_tree(tmp_path)
    write_measurements(tree.inputs)
    write_dictionary(tree.inputs)
    mapping = design_mapping(tree.inputs, output_directory=str(tree.outputs / "s"))
    path = write_design(tree.inputs, mapping)
    with running(tree) as client:
        response = _validate(client, "stats", str(path))
    assert response.status_code == 200
    assert response.json()["summary"]["groups"] == {"control": 2, "treated": 2}


def test_a_submitted_job_runs_to_success_and_is_listed(tmp_path: Path) -> None:
    tree = make_tree(tmp_path)
    with running(tree, FakeRunner(write={"summary.csv": "a,b\n1,2\n"})) as client:
        job = _submit(client, tree)
        assert job["status"] in ("queued", "running", "succeeded")
        final = _await(client, job["id"], TERMINAL)
        assert final["status"] == "succeeded"
        assert final["exit_code"] == 0
        assert final["summary"]["items"] == 1
        listed = client.get("/v1/jobs", headers=AUTH).json()["jobs"]
        assert [item["id"] for item in listed] == [job["id"]]
        filtered = client.get("/v1/jobs?status=succeeded", headers=AUTH).json()["jobs"]
        assert len(filtered) == 1
        assert client.get("/v1/jobs?status=running", headers=AUTH).json()["jobs"] == []


def test_invalid_job_requests_are_refused_and_create_nothing(tmp_path: Path) -> None:
    tree = make_tree(tmp_path)
    elsewhere = tree.root / "elsewhere"
    elsewhere.mkdir()
    write_batch_config(elsewhere, batch_mapping(tree), "batch.yaml")
    with running(tree) as client:
        refused = client.post(
            "/v1/jobs", json={"kind": "batch", "path": str(elsewhere / "batch.yaml")}, headers=AUTH
        )
        bad_filter = client.get("/v1/jobs?status=made-up", headers=AUTH)
        bad_limit = client.get("/v1/jobs?limit=500", headers=AUTH)
        unknown = client.get("/v1/jobs/does-not-exist", headers=AUTH)
        assert client.get("/v1/jobs", headers=AUTH).json()["jobs"] == []
    assert refused.status_code == 403
    assert bad_filter.status_code == 400
    assert bad_limit.status_code == 422
    assert unknown.status_code == 404
    assert unknown.json()["error"]["code"] == "not_found"


def test_a_full_queue_refuses_new_jobs_with_429(tmp_path: Path) -> None:
    tree = make_tree(tmp_path)
    runner = FakeRunner(hold=True)
    write_batch_config(tree.inputs, batch_mapping(tree, output="run-2"), "b.yaml")
    with running(tree, runner, max_concurrent_jobs=1, max_queued_jobs=0) as client:
        first = _submit(client, tree, "a.yaml", "run-1")
        _await(client, first["id"], ("running",))
        response = client.post(
            "/v1/jobs", json={"kind": "batch", "path": str(tree.inputs / "b.yaml")}, headers=AUTH
        )
        assert response.status_code == 429
        assert response.json()["error"]["code"] == "queue_full"
        runner.release()
        _await(client, first["id"], TERMINAL)


def test_a_running_job_is_cancelled_and_cannot_be_cancelled_twice(tmp_path: Path) -> None:
    tree = make_tree(tmp_path)
    runner = FakeRunner(hold=True)
    with running(tree, runner) as client:
        job = _submit(client, tree)
        _await(client, job["id"], ("running",))
        accepted = client.post(f"/v1/jobs/{job['id']}/cancel", headers=AUTH)
        assert accepted.status_code == 202
        assert accepted.json()["cancel_requested"] is True
        final = _await(client, job["id"], TERMINAL)
        assert final["status"] == "cancelled"
        assert final["exit_code"] == 130
        assert final["error"] is None
        again = client.post(f"/v1/jobs/{job['id']}/cancel", headers=AUTH)
        assert again.status_code == 409
        assert again.json()["error"]["code"] == "job_finished"


def test_a_failed_job_reports_its_code_and_message(tmp_path: Path) -> None:
    tree = make_tree(tmp_path)
    runner = FakeRunner(error=ServiceError("invalid_config", "the settings are not usable"))
    with running(tree, runner) as client:
        job = _submit(client, tree)
        final = _await(client, job["id"], TERMINAL)
    assert final["status"] == "failed"
    assert final["exit_code"] == 1
    assert final["error"] == {"code": "invalid_config", "message": "the settings are not usable"}


def test_a_run_with_failed_images_is_reported_with_exit_code_3(tmp_path: Path) -> None:
    tree = make_tree(tmp_path)
    with running(tree, FakeRunner(status="completed_with_failures")) as client:
        final = _await(client, _submit(client, tree)["id"], TERMINAL)
    assert final["status"] == "completed_with_failures"
    assert final["exit_code"] == 3


RESULTS = {
    "summary.csv": "a,b\n1,2\n",
    "nested/notes.json": "{}",
    "items/x/labels.tif": "not really a tiff",
    "secret.txt": "not a result kind",
    ".hidden.csv": "hidden",
}


def _completed_run(tree: Tree, client: TestClient) -> str:
    job = _submit(client, tree)
    _await(client, job["id"], TERMINAL)
    return str(job["id"])


def test_completed_analyses_are_listed_and_their_result_files_listed(tmp_path: Path) -> None:
    tree = make_tree(tmp_path)
    with running(tree, FakeRunner(write=RESULTS)) as client:
        job_id = _completed_run(tree, client)
        analyses = client.get("/v1/analyses", headers=AUTH).json()["analyses"]
        files = client.get(f"/v1/analyses/{job_id}/files", headers=AUTH).json()
    assert [item["id"] for item in analyses] == [job_id]
    assert [item["path"] for item in files["files"]] == [
        "items/x/labels.tif",
        "nested/notes.json",
        "summary.csv",
    ]
    assert files["truncated"] is False


def test_a_result_file_downloads_as_a_locked_down_attachment(tmp_path: Path) -> None:
    tree = make_tree(tmp_path)
    with running(tree, FakeRunner(write=RESULTS)) as client:
        job_id = _completed_run(tree, client)
        response = client.get(f"/v1/analyses/{job_id}/files/summary.csv", headers=AUTH)
    assert response.status_code == 200
    assert response.content == b"a,b\n1,2\n"
    assert response.headers["content-disposition"].startswith("attachment")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "default-src 'none'" in response.headers["content-security-policy"]
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize(
    ("path", "statuses"),
    [
        ("..%2Fsecret.txt", (403,)),
        ("..%2F..%2Fetc%2Fpasswd", (403,)),
        # A double slash may be refused by routing before the handler sees it.
        ("%2Fetc%2Fpasswd", (400, 404)),
        ("secret.txt", (403,)),
        (".hidden.csv", (403,)),
        ("nested", (403,)),
        ("missing.csv", (404,)),
    ],
)
def test_downloads_cannot_reach_outside_the_run_or_hidden_or_other_files(
    tmp_path: Path, path: str, statuses: tuple[int, ...]
) -> None:
    tree = make_tree(tmp_path)
    (tree.root / "secret.txt").write_text("outside secret", encoding="utf-8")
    with running(tree, FakeRunner(write=RESULTS)) as client:
        job_id = _completed_run(tree, client)
        response = client.get(f"/v1/analyses/{job_id}/files/{path}", headers=AUTH)
    assert response.status_code in statuses
    assert b"outside secret" not in response.content
    assert b"not a result kind" not in response.content


def test_a_symbolic_link_inside_a_run_is_neither_listed_nor_served(tmp_path: Path) -> None:
    tree = make_tree(tmp_path)
    secret = tree.root / "outside.csv"
    secret.write_text("outside", encoding="utf-8")
    with running(tree, FakeRunner(write=RESULTS)) as client:
        job_id = _completed_run(tree, client)
        run_dir = tree.outputs / "run-1"
        os.symlink(secret, run_dir / "link.csv")
        files = client.get(f"/v1/analyses/{job_id}/files", headers=AUTH).json()
        served = client.get(f"/v1/analyses/{job_id}/files/link.csv", headers=AUTH)
    assert "link.csv" not in [item["path"] for item in files["files"]]
    assert served.status_code == 403


def test_results_are_not_served_while_a_job_is_still_running(tmp_path: Path) -> None:
    tree = make_tree(tmp_path)
    runner = FakeRunner(hold=True, write=RESULTS)
    with running(tree, runner) as client:
        job = _submit(client, tree)
        _await(client, job["id"], ("running",))
        files = client.get(f"/v1/analyses/{job['id']}/files", headers=AUTH)
        runner.release()
        _await(client, job["id"], TERMINAL)
    assert files.status_code == 409
    assert files.json()["error"]["code"] == "not_completed"


def test_failed_jobs_are_not_listed_as_analyses(tmp_path: Path) -> None:
    tree = make_tree(tmp_path)
    with running(tree, FakeRunner(error=ServiceError("invalid_config", "no"))) as client:
        job_id = _completed_run(tree, client)
        analyses = client.get("/v1/analyses", headers=AUTH).json()["analyses"]
        files = client.get(f"/v1/analyses/{job_id}/files", headers=AUTH)
    assert analyses == []
    assert files.status_code == 409


def test_results_that_have_been_removed_are_reported_as_missing(tmp_path: Path) -> None:
    tree = make_tree(tmp_path)
    with running(tree, FakeRunner(write=RESULTS)) as client:
        job_id = _completed_run(tree, client)
        shutil.rmtree(tree.outputs / "run-1")
        response = client.get(f"/v1/analyses/{job_id}/files", headers=AUTH)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_a_run_without_a_manifest_has_no_quality_control_report(tmp_path: Path) -> None:
    tree = make_tree(tmp_path)
    with running(tree, FakeRunner(write=RESULTS)) as client:
        job_id = _completed_run(tree, client)
        response = client.get(f"/v1/analyses/{job_id}/qc", headers=AUTH)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "not_completed"


def test_jobs_and_their_results_survive_a_restart_of_the_service(tmp_path: Path) -> None:
    tree = make_tree(tmp_path)
    with running(tree, FakeRunner(write=RESULTS)) as client:
        job_id = _completed_run(tree, client)
    with running(tree, FakeRunner()) as client:
        restored = _job(client, job_id)
        files = client.get(f"/v1/analyses/{job_id}/files", headers=AUTH).json()["files"]
    assert restored["status"] == "succeeded"
    assert len(files) == 3


def test_an_application_refuses_a_short_token_and_missing_directories(tmp_path: Path) -> None:
    tree = make_tree(tmp_path)
    with pytest.raises(ConfigError, match="at least 32 characters"):
        create_app(settings_for(tree), token="short")
    with pytest.raises(ConfigError, match="does not exist"):
        create_app(settings_for(tree, input_roots=(tmp_path / "absent",)), token=TOKEN)


def test_a_real_batch_runs_through_the_api_and_reports_quality_control(tmp_path: Path) -> None:
    tree = make_tree(tmp_path)
    mapping = batch_mapping(tree, output="real-run")
    images = [Path(item["path"]) for item in mapping["dataset"]["images"]]
    for index, path in enumerate(images):
        discs = [Disc((20.0, 20.0), 8.0), Disc((42.0, 44.0), 6.0)]
        image, _ = make_phantom(discs, (64, 64), seed=index)
        write_image(path, np.round(image).astype(np.uint16))
    config = write_batch_config(tree.inputs, mapping, "real.yaml")

    with running(tree) as client:
        response = client.post(
            "/v1/jobs", json={"kind": "batch", "path": str(config)}, headers=AUTH
        )
        assert response.status_code == 202, response.text
        job_id = response.json()["id"]
        final = _await(client, job_id, TERMINAL, timeout=120.0)
        assert final["status"] == "succeeded", final
        assert final["summary"]["counts"]["ok"] == 2

        qc = client.get(f"/v1/analyses/{job_id}/qc", headers=AUTH).json()
        assert [item["status"] for item in qc["items"]] == ["ok", "ok"]
        assert all(item["objects"] >= 1 for item in qc["items"])
        report_path = qc["items"][0]["qc_report"]
        assert report_path is not None and report_path.endswith("qc_report.html")

        page = client.get(f"/v1/analyses/{job_id}/files/{report_path}", headers=AUTH)
        assert page.status_code == 200
        assert b"<html" in page.content.lower()

        files = client.get(f"/v1/analyses/{job_id}/files", headers=AUTH).json()["files"]
        listed = {item["path"] for item in files}
        assert {"batch_manifest.json", "summary.csv"} <= listed
        assert any(path.endswith("labels.tif") for path in listed)

        labels = client.get(
            f"/v1/analyses/{job_id}/files/" + next(p for p in listed if p.endswith("labels.tif")),
            headers=AUTH,
        )
        assert labels.content[:4] in (b"II*\x00", b"MM\x00*")

        resubmitted = client.post(
            "/v1/jobs", json={"kind": "batch", "path": str(config)}, headers=AUTH
        )
        again = _await(client, resubmitted.json()["id"], TERMINAL, timeout=120.0)
    assert again["status"] == "succeeded"
    assert again["summary"]["counts"]["resumed"] == 2

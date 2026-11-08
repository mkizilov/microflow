# The analysis service

`microflow serve` runs an HTTP service that submits batch analyses, reports their status, and serves
their results and quality-control reports. It is optional. The command-line interface works fully
without it, and without the extra installed.

## Quick start

```sh
uv sync --extra service
mkdir -p data/service-input data/service-output
export MICROFLOW_SERVICE_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
uv run microflow serve examples/service/service.yaml
```

Then, from another shell with the same token in `$MICROFLOW_SERVICE_TOKEN`:

```sh
curl -s -H "Authorization: Bearer $MICROFLOW_SERVICE_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"kind": "batch", "path": "/abs/path/to/data/service-input/batch.yaml"}' \
  http://127.0.0.1:21300/v1/jobs
```

Submission returns `202` with a job id. Poll `GET /v1/jobs/{id}` until `status` is final.

## Security model

The service is designed for one user on one machine, behind loopback. These are its controls.

**Authentication.** Every route except `GET /v1/health` requires `Authorization: Bearer <token>`. The
token comes only from the `MICROFLOW_SERVICE_TOKEN` environment variable, must be at least 32
characters, and is compared in constant time. The service refuses to start without a valid token.
The token is never written to the settings file, the logs or any response.

**Local restriction.** The service listens on `127.0.0.1` by default. A non-loopback address needs
`--allow-remote`, and that flag is there to make the choice deliberate. Put a TLS-terminating proxy
in front before using it. The service has no TLS of its own.

**Directory restriction.** The settings file names the only directories the service may read
(`input_roots`) and write (`output_roots`). Every client path must be absolute, and is resolved
(following symbolic links and `..`) before it is compared with those directories:

- an input must exist inside an input root. A link that points outside is refused like any other
  outside path;
- an output directory must lie strictly inside an output root, and must not be an existing file;
- the batch configuration's image paths and output directory are checked, so a configuration cannot
  reach beyond the roots;
- the state database must not sit inside an output root, so it cannot be downloaded.

**Downloads.** Only files from a *completed* analysis are served, by relative name. The service
refuses hidden names, names containing `..`, symbolic links anywhere on the path, and any file whose
kind is not a result kind (`.csv`, `.json`, `.tif`, `.tiff`, `.png`, `.svg`, `.html`, `.parquet`).
Listings show only those kinds. Downloads are attachments, with `X-Content-Type-Options: nosniff`, a
`default-src 'none'` content security policy and `Cache-Control: no-store`, because quality-control
reports are HTML.

**No uploads.** The API accepts paths, not file contents. Nothing is written by a client except
through a batch configuration whose paths pass the checks above.

**No plugins.** A configuration that uses a plugin (a segmenter, a feature, a quality check, or a
plugin preprocessing step) is refused with `plugins_disabled`. Plugins run in the service's process
with full privileges, and a client should not be able to name one.

**Quiet by default.** The OpenAPI schema and the interactive docs are disabled. Error responses carry
a stable code and a message. Absolute server paths are replaced by the labels clients may see, and
unexpected exceptions return a generic message while the full error goes to the service log.

## Endpoints

All routes are under `/v1`. Errors have one shape: `{"error": {"code", "message", "details"}}`.

| Method and path | Purpose | Success |
| --- | --- | --- |
| `GET /v1/health` | Liveness. No token. | `200 {"status": "ok"}` |
| `POST /v1/validate` | Check a `batch` or `stats` configuration without running it. Body: `{"kind", "path"}`. | `200` with a summary and warnings |
| `POST /v1/jobs` | Submit a batch analysis. Body: `{"kind": "batch", "path"}`. | `202` with the job |
| `GET /v1/jobs?status=&limit=` | List jobs, newest first. `limit` is 1 to 200. | `200 {"jobs": [...]}` |
| `GET /v1/jobs/{id}` | Job status, progress, exit code, error and summary. | `200` |
| `POST /v1/jobs/{id}/cancel` | Cancel a queued job, or ask a running one to stop. | `202` |
| `GET /v1/analyses?limit=` | Completed analyses (succeeded or completed with failures). | `200 {"analyses": [...]}` |
| `GET /v1/analyses/{id}/files` | Result files of a completed analysis, with sizes. | `200 {"files": [...], "truncated"}` |
| `GET /v1/analyses/{id}/files/{path}` | Download one result file. | `200` attachment |
| `GET /v1/analyses/{id}/qc` | Per-image quality control: status, objects, flags, warnings and the report path. | `200` |

A job view contains `id`, `kind`, `status`, `config` (the file's name), `output` (the output
directory as `root-name/relative-path`), the timestamps, `exit_code`, `cancel_requested`, `progress`
(`completed` and `total` images), `error` (`code` and `message`) and `summary`. Absolute server paths
are not returned.

## Job states and exit codes

| Status | Meaning | Exit code |
| --- | --- | --- |
| `queued` | Accepted, waiting for a worker. | |
| `running` | Being processed. | |
| `succeeded` | Every image succeeded. | `0` |
| `completed_with_failures` | The run finished, and some images failed. Their results are still served. | `3` |
| `cancelled` | Cancelled by a client, or stopped by a shutdown while running. | `130` |
| `failed` | The run could not start or was refused. `error.code` says why (see below). | `1` |
| `interrupted` | The service stopped before a queued job started, or while a job was running and it did not end cleanly. Submit the configuration again to reuse its cached results. | |

The exit codes match `microflow batch`.

Error codes on a `failed` job: `invalid_config` (the configuration changed or became invalid after
submission), `not_found` (an image disappeared), `path_not_allowed`, `batch_refused` (the output
directory is busy, or holds results of a different analysis), `io_error`, and `internal` (an unexpected
error; the detail is in the service log).

A cancelled job stops after the image in progress. Images already finished keep their results, and
the manifest records the run as `cancelled`, as the batch command does.

**Cancellation rules.** A queued job is cancelled immediately and never runs. A running job is asked to
stop, and `cancel_requested` is set until it ends. Cancelling a job that has already finished is a
`409 job_finished`. If a job finishes before the cancellation takes effect, its real outcome is kept.

## Concurrency and limits

- At most `max_concurrent_jobs` batches run at once (default 1, at most 4).
- At most `max_queued_jobs` more wait (default 8). A submission beyond that gets `429 queue_full`.
- Each batch runs with one worker, inline in its job thread. A larger worker count in a configuration
  is reduced to one, and the reduction is reported as a warning. The batch engine's process pool is
  never used, so the service never forks from its own multithreaded process. Parallelism comes from
  running several jobs at once.
- Two jobs cannot run over the same output directory at the same time. The second gets `failed` with
  `batch_refused`, as it would from the command line. Once the first has finished, the directory can
  be used again, and an identical configuration resumes from its cached results.
- Listings are bounded: `limit` is at most 200, and a file listing shows at most 5000 files and says
  when it was cut short.

## Persistence and restarts

Job records are kept in `jobs.sqlite` in `state_directory`. Each state change is a single guarded
statement, so two threads cannot start the same job. On start, any job left `queued` or `running` by
an earlier process is marked `interrupted`. Nothing is resumed automatically. Finished jobs and their
results survive a restart and remain available.

On shutdown the service stops accepting jobs, cancels the running ones and waits for them to stop
after their current image. Jobs that never started are marked `interrupted`.

## Configuration

`examples/service/service.yaml` is annotated. The settings are:

| Setting | Default | Meaning |
| --- | --- | --- |
| `state_directory` | required | Where `jobs.sqlite` lives. Must not be inside an output root. |
| `input_roots` | required | Directories the service may read. Absolute, or relative to the settings file. |
| `output_roots` | required | Directories the service may write to. |
| `max_concurrent_jobs` | 1 | Batches running at once (1 to 4). |
| `max_queued_jobs` | 8 | Batches waiting (0 to 256). |
| `host` | `127.0.0.1` | Address to listen on. |
| `port` | 21300 | Port. |
| `allow_remote` | false | Accept a non-loopback address. |

The token is not a setting. It comes from `MICROFLOW_SERVICE_TOKEN`. Rotating it means restarting the
service.

## Limitations

- **Single machine, single user.** Nothing is multi-tenant. Every authenticated client sees every job.
- **No TLS, no rate limiting, no request size limit.** Bind to loopback, or put a proxy in front that
  provides TLS and limits. A very large request body is read in full before it is rejected.
- **Validation reads the whole measurement table** for a statistics design, so a very large file costs
  memory and time on a validation request.
- **Paths are absolute.** A client needs to know the server's paths. The service does not translate
  them.
- **Errors from a batch can name a file inside an input root.** Load errors include the image's path,
  which the client has already named. Paths outside the roots are never echoed.
- **Cancellation is cooperative.** The batch checks for it between images, so a long single image is
  finished before the job stops.
- **Plugins are not supported over the API.** Use the command line for them.
- **No analysis other than batch** runs as a job. Statistics designs can be validated, but are run from
  the command line.
- **The job store is one SQLite file** written by one process. Do not run two services on one state
  directory.
- **Downloads are not range-limited or quota-limited.** A client that can read a result can read all
  of it.

## Verification status

**Not run in the environment where this was written**, which had no network and none of the web or
scientific packages. Verified there: static checks over the whole tree (undefined names, imports,
call signatures, unused code, cycles), a check that FastAPI and uvicorn are imported only by the two
service modules that need them and never by the CLI's import chain (confirmed on a seeded violation),
the path policy and job state rules by reading, and compilation.

The tests are written but not run. `tests/test_cli_service_optional.py`, `tests/test_service_paths.py`,
`tests/test_service_store.py`, `tests/test_service_jobs.py`, `tests/test_service_validation.py` and
`tests/test_service_settings.py` need only the core dependencies. `tests/test_service_api.py` needs the
`service` extra and runs in the CI `service` job, including a real batch on synthetic images. The
HTTP-level assertions about Starlette's routing of encoded slashes are the least certain part.

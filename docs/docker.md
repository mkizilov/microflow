# Docker deployment for the analysis service

The `Dockerfile` and `compose.yaml` package the optional HTTP service (`microflow serve`). They are
for running the service, not the command-line tools.

## Status

**Not built and not run.** The authoring environment had no container runtime and no network, so the
image was not built, and the compose file was not started. The files follow the project's own service
settings and security model (see [service](service.md)), and they are reviewable configuration. Before
relying on them, build and start the image on a machine with Docker, and run the service tests against
it.

## Build and run

```sh
export MICROFLOW_SERVICE_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
mkdir -p data/service-input data/service-output data/service-state
docker compose up --build
```

The service listens on `127.0.0.1:21300` on the host. It is not reachable from other machines.

Submit jobs with absolute paths **inside the container**. Inputs are mounted read-only at `/data/input`,
so a job configuration placed in `data/service-input/` on the host is seen at `/data/input/` in the
container, and its image paths must use the container paths:

```yaml
dataset:
  images:
    - {name: sample, path: /data/input/raw/sample.tif}
output_directory: /data/output/run-1
```

## What the image does

- Uses Python 3.12 slim and installs the locked dependencies with `uv sync --locked --extra service`.
  It needs `uv.lock` to be committed.
- Runs as an unprivileged user (uid 10001), not root.
- Keeps the root filesystem read-only, with `/tmp` as a writable tmpfs.
- Mounts inputs read-only, and writes results to `/data/output` and the job database to `/data/state`.
- Drops all Linux capabilities and sets `no-new-privileges`.
- Reads the token from the environment at run time. The token is never written into the image.
- Publishes the port on the host's loopback interface only.

The container listens on all of its own interfaces (`0.0.0.0`) and passes `--allow-remote`. That is
safe only because the host publishes the port on loopback. Do not change the published address to
`0.0.0.0` without a TLS-terminating proxy in front. The service itself does not provide TLS.

## What is not verified

- The image builds from a clean checkout with `uv.lock` present.
- The service starts inside the container and answers `GET /v1/health`.
- The read-only root filesystem allows every write the service makes (`/data/state`, `/data/output`
  and `/tmp` are the only writable paths; this needs a run to confirm).
- The `service` tests pass against the container.

## Limits

- `uv.lock` must be committed. Without it, the build fails by design.
- Matplotlib's configuration directory is set to `/tmp/matplotlib`, which the read-only root allows.
- The image does not include the ML extra. Training and inference need a separate image with PyTorch.
- The API refuses plugins at submission, so the image's plugin entry points are never used for jobs.
  The package installs them with the rest of MicroFlow; they are inert in this image.
- Resource limits (memory, CPU) are not set in `compose.yaml`. Add them for your host.

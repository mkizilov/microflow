# Container image for the optional analysis service (microflow serve). Not built or run in the
# authoring environment; see docs/docker.md for what was and was not verified.
#
# Build requires uv.lock (run `uv lock` once with network access and commit it).
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/microflow/.venv \
    MPLCONFIGDIR=/tmp/matplotlib \
    PATH=/opt/microflow/.venv/bin:$PATH

# uv is pinned to the version CI uses.
COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /usr/local/bin/uv

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin microflow \
    && mkdir -p /data/input /data/output /data/state /etc/microflow \
    && chown microflow:microflow /data/output /data/state

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --locked --extra service --no-dev

COPY examples/service/service-docker.yaml /etc/microflow/service.yaml

USER microflow
VOLUME ["/data/output", "/data/state"]
EXPOSE 21300

# The token is supplied at run time as MICROFLOW_SERVICE_TOKEN; it is never baked into the image.
# --allow-remote lets the container listen on all of its interfaces. Publish the port on the host's
# loopback interface only (see compose.yaml) so that nothing outside the host can connect.
ENTRYPOINT ["microflow", "serve", "/etc/microflow/service.yaml", "--host", "0.0.0.0", "--allow-remote"]

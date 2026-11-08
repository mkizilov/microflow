"""Errors the service reports to clients, and the HTTP status each code maps to. Torch-free and
FastAPI-free, so the same errors come out of the store, the path policy and the job executor."""

from collections.abc import Sequence

STATUS_BY_CODE: dict[str, int] = {
    "bad_request": 400,
    "unauthorized": 401,
    "path_not_allowed": 403,
    "not_found": 404,
    "job_finished": 409,
    "not_completed": 409,
    "invalid_request": 422,
    "invalid_config": 422,
    "plugins_disabled": 422,
    "queue_full": 429,
    "unavailable": 503,
    "internal": 500,
}


class ServiceError(Exception):
    """A request cannot be served. ``code`` is stable for clients; ``message`` is safe to show."""

    def __init__(self, code: str, message: str, *, details: Sequence[str] = ()) -> None:
        if code not in STATUS_BY_CODE:
            raise ValueError(f"unknown service error code {code!r}")
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = tuple(details)

    @property
    def status(self) -> int:
        return STATUS_BY_CODE[self.code]

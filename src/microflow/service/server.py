"""Run the service under uvicorn. Needs the ``service`` extra.

Imported only by ``microflow serve``.
"""

from microflow.service.settings import ServiceSettings


def serve(settings: ServiceSettings, *, token: str) -> None:
    import uvicorn

    from microflow.service.app import create_app

    app = create_app(settings, token=token)
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level="info",
        server_header=False,
    )

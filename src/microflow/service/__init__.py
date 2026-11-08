"""An optional HTTP service for submitting, tracking and downloading batch analyses.

Only :mod:`microflow.service.app` and :mod:`microflow.service.server` import FastAPI or uvicorn, and
they are loaded only by ``microflow serve``. The other modules (settings, path policy, store, jobs,
validation) work without the extra, and the command-line interface never needs it. See
docs/service.md.
"""

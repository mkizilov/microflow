"""Reporting: previews, quality-control checks, headless figures and self-contained HTML.

Reporting only reads results. Nothing in the numerical packages imports it. Modules are imported
directly (for example ``microflow.reporting.html``) so that light imports such as the preview
writer do not load matplotlib.
"""

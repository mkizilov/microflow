"""Provenance: what produced a result, from which inputs, with which settings, and when.

Submodules: ``hashing`` (canonical digests), ``versions`` (software, algorithm and library
versions), ``record`` (the provenance record and identities), ``guard`` (protection against
overwriting unrelated results), ``compare`` (configuration differences) and ``table`` (typed
CSV and Parquet writers). Import them directly; this package imports nothing heavy at load time.
"""

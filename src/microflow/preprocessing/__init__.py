"""Preprocessing: flat-field correction, background, filtering, masking and normalisation."""

from microflow.preprocessing.operations import PreprocessingError
from microflow.preprocessing.pipeline import PreprocessResult, ProcessingRecord, preprocess

__all__ = ["PreprocessResult", "PreprocessingError", "ProcessingRecord", "preprocess"]

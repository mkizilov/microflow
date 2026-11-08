"""Classical segmentation: thresholds, morphology, labelling, watershed, cleanup and metrics."""

from microflow.segmentation.metrics import OverlapScores, score_segmentation
from microflow.segmentation.pipeline import SegmentationResult, segment_image, segment_plane

__all__ = [
    "OverlapScores",
    "SegmentationResult",
    "score_segmentation",
    "segment_image",
    "segment_plane",
]

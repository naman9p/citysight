"""Reproducible, label-driven evaluation utilities for CitySight ANPR."""

from phase1_anpr.evaluation.metrics import (
    DetectionMetrics,
    OCRMetrics,
    character_accuracy,
    evaluate_detections,
    evaluate_ocr,
    intersection_over_union,
)

__all__ = [
    "DetectionMetrics",
    "OCRMetrics",
    "character_accuracy",
    "evaluate_detections",
    "evaluate_ocr",
    "intersection_over_union",
]

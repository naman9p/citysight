"""Deterministic plate-crop quality scoring for Phase 1 (Step 5).

Scores candidate plate crops on factors computable straight from the pixels
(sharpness, size, brightness, contrast) using only OpenCV/NumPy, then selects
the best crops per track. No ML models, no OCR.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ScoredCrop:
    """A candidate crop with its computed quality and provenance."""

    crop: np.ndarray
    quality_score: float
    detector_confidence: float
    frame_number: int


# Default factor weights; overridden by config.yaml's `quality` section.
DEFAULT_WEIGHTS = {
    "sharpness_weight": 0.4,
    "size_weight": 0.3,
    "brightness_weight": 0.15,
    "contrast_weight": 0.15,
}


class QualityScorer:
    """Compute a 0..1 quality score for plate crops and pick the best ones."""

    def __init__(self, weights=None, target_area=8000, top_k=3,
                 min_plate_width=1, min_plate_height=1):
        w = dict(DEFAULT_WEIGHTS)
        if weights:
            w.update({k: v for k, v in weights.items() if k in DEFAULT_WEIGHTS})
        self.weights = w
        self.target_area = max(1, int(target_area))
        self.top_k = max(1, int(top_k))
        self.min_plate_width = min_plate_width
        self.min_plate_height = min_plate_height

    @classmethod
    def from_config(cls, config):
        q = config.get("quality", {})
        weights = {k: q[k] for k in DEFAULT_WEIGHTS if k in q}
        return cls(
            weights=weights,
            target_area=q.get("target_area", 8000),
            top_k=q.get("top_k", 3),
            min_plate_width=q.get("min_plate_width", 1),
            min_plate_height=q.get("min_plate_height", 1),
        )

    def is_valid(self, crop):
        """Reject empty, malformed, or too-small crops."""
        if crop is None or not hasattr(crop, "size") or crop.size == 0:
            return False
        if crop.ndim not in (2, 3):
            return False
        h, w = crop.shape[:2]
        return w >= self.min_plate_width and h >= self.min_plate_height

    @staticmethod
    def _to_gray(crop):
        if crop.ndim == 3:
            return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return crop

    def score(self, crop):
        """Return a deterministic quality score in [0, 1]; 0.0 for invalid crops."""
        if not self.is_valid(crop):
            return 0.0

        gray = self._to_gray(crop)
        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)
        h, w = gray.shape[:2]

        # Sharpness: variance of the Laplacian, squashed into [0, 1].
        sharp_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharpness = sharp_var / (sharp_var + 500.0)

        # Size: crop area relative to a reference area, capped at 1.
        size = min(1.0, (w * h) / float(self.target_area))

        # Brightness: mean intensity, best near mid-gray (128), 0 at extremes.
        brightness = 1.0 - abs(float(gray.mean()) - 128.0) / 128.0
        brightness = max(0.0, brightness)

        # Contrast: std dev normalized against a healthy spread (~64).
        contrast = min(1.0, float(gray.std()) / 64.0)

        w_sum = sum(self.weights.values()) or 1.0
        combined = (
            self.weights["sharpness_weight"] * sharpness
            + self.weights["size_weight"] * size
            + self.weights["brightness_weight"] * brightness
            + self.weights["contrast_weight"] * contrast
        ) / w_sum
        return float(round(combined, 6))

    def select_best(self, candidates):
        """Score candidates and return the top_k ScoredCrop objects.

        `candidates` is an iterable of (frame_number, crop, detector_confidence),
        matching PlateTracker's Track.candidates. Invalid crops are skipped.
        """
        scored = []
        for frame_number, crop, det_conf in candidates:
            if not self.is_valid(crop):
                continue
            scored.append(
                ScoredCrop(
                    crop=crop,
                    quality_score=self.score(crop),
                    detector_confidence=float(det_conf),
                    frame_number=int(frame_number),
                )
            )
        # Sort by quality, then frame_number for deterministic tie-breaking.
        scored.sort(key=lambda s: (-s.quality_score, s.frame_number))
        return scored[: self.top_k]

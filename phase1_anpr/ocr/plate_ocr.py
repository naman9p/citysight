"""OCR over selected plate crops for Phase 1 (Step 6).

Runs a pluggable OCR backend on the top ScoredCrops from the quality step and
returns a stable OCRResult. PaddleOCR is the Phase 1 baseline backend, but the
rest of the code depends only on the small OCRBackend interface below, so it is
never coupled to PaddleOCR directly (and tests use a mock backend).

Cleanup here is intentionally minimal: strip whitespace and uppercase. No plate
grammar, character correction, voting, or calibration (those are later steps).
"""

from dataclasses import dataclass
from typing import List, Protocol, Tuple


@dataclass
class OCRResult:
    """Recognized text for one plate crop, with provenance from earlier steps."""

    text: str
    ocr_confidence: float
    quality_score: float
    detector_confidence: float
    frame_number: int


class OCRBackend(Protocol):
    """Minimal OCR backend contract.

    Implementations return a list of (text, confidence) segments for an image.
    An empty list means "no text found".
    """

    def recognize(self, image) -> List[Tuple[str, float]]:
        ...


class PaddleOCRBackend:
    """PaddleOCR-backed implementation of OCRBackend (lazy import).

    Uses PaddleOCR's recognition-only `TextRecognition` predictor since the
    inputs are already-cropped plates — no text detection stage is run. The
    model is initialized once and reused across crops. Kept thin so nothing
    else imports paddleocr.
    """

    def __init__(self, input_shape=None, **kwargs):
        try:
            from paddleocr import TextRecognition
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "paddleocr is not installed; run pip install -r requirements.txt"
            ) from exc
        self._rec = TextRecognition(input_shape=input_shape, **kwargs)

    def recognize(self, image):  # pragma: no cover - needs model weights
        outputs = self._rec.predict(image) or []
        segments = []
        for item in outputs:
            text = item["rec_text"] if "rec_text" in item else getattr(item, "rec_text", "")
            score = item["rec_score"] if "rec_score" in item else getattr(item, "rec_score", 0.0)
            segments.append((text, float(score)))
        return segments


def _clean(text: str) -> str:
    """Basic cleanup only: trim whitespace and uppercase."""
    return " ".join(text.split()).upper()


class PlateOCR:
    """Recognize plate text from ScoredCrops using an injectable backend."""

    def __init__(self, backend: OCRBackend):
        self.backend = backend

    def read(self, scored_crop) -> OCRResult:
        """OCR a single ScoredCrop and return an OCRResult."""
        crop = getattr(scored_crop, "crop", None)
        if crop is None or getattr(crop, "size", 0) == 0:
            return self._empty(scored_crop)

        segments = self.backend.recognize(crop) or []
        parts = [(_clean(t), c) for (t, c) in segments if _clean(t)]
        if not parts:
            return self._empty(scored_crop)

        text = "".join(t for t, _ in parts)
        # Confidence of the whole plate is only as strong as its weakest segment.
        confidence = min(c for _, c in parts)
        return OCRResult(
            text=text,
            ocr_confidence=float(confidence),
            quality_score=scored_crop.quality_score,
            detector_confidence=scored_crop.detector_confidence,
            frame_number=scored_crop.frame_number,
        )

    def read_many(self, scored_crops) -> List[OCRResult]:
        return [self.read(sc) for sc in scored_crops]

    @staticmethod
    def _empty(scored_crop) -> OCRResult:
        return OCRResult(
            text="",
            ocr_confidence=0.0,
            quality_score=getattr(scored_crop, "quality_score", 0.0),
            detector_confidence=getattr(scored_crop, "detector_confidence", 0.0),
            frame_number=getattr(scored_crop, "frame_number", -1),
        )

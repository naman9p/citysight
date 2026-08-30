"""Track-level ANPR processing for Phase 1 (Step 8).

Wires rectification + OCR over a track's top ScoredCrops and fuses the OCR
results deterministically into ranked plate candidates. Rectifier and OCR are
injected so tests never touch the real Paddle model.

No edit-distance correction, plate grammar, or calibration here — just weighted
evidence fusion over identical OCR strings.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from phase1_anpr.ocr.plate_ocr import OCRResult
from phase1_anpr.quality.quality_scorer import ScoredCrop


@dataclass
class FusedCandidate:
    """One plate-text hypothesis fused across a track's crops."""

    text: str
    score: float
    support: int  # number of crops that produced this text


@dataclass
class TrackResult:
    """Fused ANPR outcome for a single track."""

    track_id: int
    best_text: Optional[str]
    candidates: List[FusedCandidate] = field(default_factory=list)
    evidence: List[OCRResult] = field(default_factory=list)  # per-frame OCR


# Default fusion weights; overridable from config.yaml's `confidence` section.
DEFAULT_WEIGHTS = {"ocr_weight": 0.6, "quality_weight": 0.2, "detector_weight": 0.2}


class TrackProcessor:
    """Rectify → OCR → fuse for a track's selected crops."""

    def __init__(self, rectifier, ocr, weights=None, top_k=3):
        self.rectifier = rectifier
        self.ocr = ocr
        w = dict(DEFAULT_WEIGHTS)
        if weights:
            w.update({k: v for k, v in weights.items() if k in DEFAULT_WEIGHTS})
        self.weights = w
        self.top_k = max(1, int(top_k))

    @classmethod
    def from_config(cls, rectifier, ocr, config):
        conf = config.get("confidence", {})
        weights = {k: conf[k] for k in DEFAULT_WEIGHTS if k in conf}
        return cls(rectifier, ocr, weights=weights)

    def _evidence_score(self, result: OCRResult) -> float:
        w_sum = sum(self.weights.values()) or 1.0
        return (
            self.weights["ocr_weight"] * result.ocr_confidence
            + self.weights["quality_weight"] * result.quality_score
            + self.weights["detector_weight"] * result.detector_confidence
        ) / w_sum

    def process(self, track_id, scored_crops) -> TrackResult:
        """Rectify + OCR each crop (up to top_k), then fuse into ranked candidates."""
        evidence: List[OCRResult] = []
        for sc in scored_crops[: self.top_k]:
            rect = self.rectifier.rectify(sc.crop)
            # OCR the rectified crop while preserving the crop's metadata.
            rectified_sc = ScoredCrop(
                crop=rect.rectified_crop,
                quality_score=sc.quality_score,
                detector_confidence=sc.detector_confidence,
                frame_number=sc.frame_number,
            )
            evidence.append(self.ocr.read(rectified_sc))

        candidates = self._fuse(evidence)
        best_text = candidates[0].text if candidates else None
        return TrackResult(
            track_id=track_id,
            best_text=best_text,
            candidates=candidates,
            evidence=evidence,
        )

    def _fuse(self, evidence: List[OCRResult]) -> List[FusedCandidate]:
        """Group identical non-empty texts and rank by summed weighted evidence."""
        grouped = {}  # text -> [score_sum, support]
        for result in evidence:
            if not result.text:
                continue
            entry = grouped.setdefault(result.text, [0.0, 0])
            entry[0] += self._evidence_score(result)
            entry[1] += 1

        candidates = [
            FusedCandidate(text=text, score=round(score, 6), support=support)
            for text, (score, support) in grouped.items()
        ]
        # Deterministic ranking: score desc, then support desc, then text asc.
        candidates.sort(key=lambda c: (-c.score, -c.support, c.text))
        return candidates

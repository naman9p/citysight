"""ANPR confidence scoring + decision for Phase 1 (Step 10).

Combines the evidence already produced by earlier steps — fused OCR candidates
and per-frame evidence (TrackResult / FusedCandidate / OCRResult) plus format
validity (NormalizationResult) — into a single deterministic composite score
and an accepted / review / abstained decision.

This score is a heuristic confidence, NOT a statistically calibrated
probability; real calibration needs evaluation data (a later step). No OCR
character correction or guessing happens here.
"""

from dataclasses import dataclass, field
from typing import List, Optional


DECISION_ACCEPTED = "accepted"
DECISION_REVIEW = "review"
DECISION_ABSTAINED = "abstained"

DEFAULT_WEIGHTS = {
    "ocr_weight": 0.35,
    "quality_weight": 0.15,
    "detector_weight": 0.15,
    "support_weight": 0.15,
    "validity_weight": 0.2,
}


@dataclass
class ConfidenceResult:
    """Final confidence + decision for a track's plate reading."""

    confidence_score: float          # clamped to [0, 1]
    decision: str                    # accepted | review | abstained
    normalized_text: Optional[str]
    reasons: List[str] = field(default_factory=list)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


class ConfidenceScorer:
    """Score fused ANPR evidence and decide accepted/review/abstained."""

    def __init__(self, weights=None, accept_threshold=0.6, review_threshold=0.4):
        w = dict(DEFAULT_WEIGHTS)
        if weights:
            w.update({k: v for k, v in weights.items() if k in DEFAULT_WEIGHTS})
        self.weights = w
        self.accept_threshold = accept_threshold
        self.review_threshold = review_threshold

    @classmethod
    def from_config(cls, config):
        conf = config.get("confidence", {})
        weights = {k: conf[k] for k in DEFAULT_WEIGHTS if k in conf}
        return cls(
            weights=weights,
            accept_threshold=conf.get("accept_threshold", 0.6),
            review_threshold=conf.get("review_threshold", 0.4),
        )

    def score(self, track_result, normalization) -> ConfidenceResult:
        """Combine track evidence + normalization into a ConfidenceResult."""
        reasons: List[str] = []

        best_text = track_result.best_text
        if not best_text:
            reasons.append("no OCR text produced")
            return ConfidenceResult(0.0, DECISION_ABSTAINED, None, reasons)

        # Evidence rows that produced the winning text.
        supporting = [e for e in track_result.evidence if e.text == best_text]
        if not supporting:  # defensive; best_text should come from evidence
            supporting = list(track_result.evidence)

        n = len(supporting)
        ocr = sum(e.ocr_confidence for e in supporting) / n
        quality = sum(e.quality_score for e in supporting) / n
        detector = sum(e.detector_confidence for e in supporting) / n
        total_evidence = max(1, len(track_result.evidence))
        support_ratio = _clamp01(len(supporting) / total_evidence)
        validity = 1.0 if normalization.is_valid else 0.0

        w = self.weights
        w_sum = sum(w.values()) or 1.0
        composite = (
            w["ocr_weight"] * ocr
            + w["quality_weight"] * quality
            + w["detector_weight"] * detector
            + w["support_weight"] * support_ratio
            + w["validity_weight"] * validity
        ) / w_sum
        composite = _clamp01(round(composite, 6))

        reasons.append(f"ocr={ocr:.2f} quality={quality:.2f} detector={detector:.2f}")
        reasons.append(f"support={len(supporting)}/{total_evidence}")
        reasons.append("format valid" if normalization.is_valid else "format invalid")

        decision = self._decide(composite, normalization.is_valid, reasons)
        return ConfidenceResult(
            confidence_score=composite,
            decision=decision,
            normalized_text=normalization.normalized_text or None,
            reasons=reasons,
        )

    def _decide(self, score, is_valid, reasons) -> str:
        if score >= self.accept_threshold and is_valid:
            reasons.append("accepted: strong evidence, valid format")
            return DECISION_ACCEPTED
        if score >= self.review_threshold:
            # Includes invalid-format-but-strong: never accepted, may review.
            if score >= self.accept_threshold and not is_valid:
                reasons.append("review: strong evidence but invalid format")
            else:
                reasons.append("review: borderline evidence")
            return DECISION_REVIEW
        reasons.append("abstained: evidence below review threshold")
        return DECISION_ABSTAINED

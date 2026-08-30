"""Tests for Step 10 confidence scoring + decisions."""

from phase1_anpr.confidence.confidence_scorer import (
    ConfidenceResult,
    ConfidenceScorer,
)
from phase1_anpr.normalization.plate_normalizer import NormalizationResult
from phase1_anpr.ocr.plate_ocr import OCRResult
from phase1_anpr.pipeline.track_processor import FusedCandidate, TrackResult


def evidence(text, ocr, quality, det, frame):
    return OCRResult(text=text, ocr_confidence=ocr, quality_score=quality,
                     detector_confidence=det, frame_number=frame)


def track(best_text, rows):
    ev = [evidence(*r) for r in rows]
    cands = [FusedCandidate(text=best_text, score=1.0, support=len(ev))] if best_text else []
    return TrackResult(track_id=1, best_text=best_text, candidates=cands, evidence=ev)


def norm(text, is_valid, fmt=None, state=None):
    return NormalizationResult(raw_text=text, normalized_text=text,
                               is_valid=is_valid, format_type=fmt, state_code=state)


def test_strong_valid_multiframe_accepted():
    tr = track("MH12AB1234", [
        ("MH12AB1234", 0.9, 0.8, 0.9, 0),
        ("MH12AB1234", 0.88, 0.82, 0.9, 1),
    ])
    r = ConfidenceScorer().score(tr, norm("MH12AB1234", True, "standard", "MH"))
    assert isinstance(r, ConfidenceResult)
    assert r.decision == "accepted"
    assert 0.0 <= r.confidence_score <= 1.0 and r.confidence_score >= 0.6
    assert r.normalized_text == "MH12AB1234"


def test_borderline_result_review():
    tr = track("MH12AB1234", [
        ("MH12AB1234", 0.45, 0.4, 0.4, 0),
        ("XX", 0.4, 0.4, 0.4, 1),  # dissents -> support ratio 1/2
    ])
    r = ConfidenceScorer().score(tr, norm("MH12AB1234", True, "standard", "MH"))
    assert r.decision == "review"
    assert 0.4 <= r.confidence_score < 0.6


def test_invalid_format_strong_ocr_review_never_accepted():
    tr = track("ZZ12AB1234", [
        ("ZZ12AB1234", 0.95, 0.9, 0.9, 0),
        ("ZZ12AB1234", 0.95, 0.9, 0.9, 1),
    ])
    r = ConfidenceScorer().score(tr, norm("ZZ12AB1234", False))
    assert r.decision == "review"       # strong evidence, but invalid format
    assert r.decision != "accepted"


def test_empty_ocr_abstained():
    tr = track(None, [])
    r = ConfidenceScorer().score(tr, norm("", False))
    assert r.decision == "abstained"
    assert r.confidence_score == 0.0 and r.normalized_text is None


def test_weak_evidence_abstained():
    tr = track("ABCD", [("ABCD", 0.1, 0.1, 0.1, 0)])
    r = ConfidenceScorer().score(tr, norm("ABCD", False))
    assert r.decision == "abstained"
    assert r.confidence_score < 0.4


def test_configurable_thresholds():
    tr = track("MH12AB1234", [("MH12AB1234", 0.5, 0.5, 0.5, 0)])
    n = norm("MH12AB1234", True, "standard", "MH")
    strict = ConfidenceScorer(accept_threshold=0.95, review_threshold=0.9)
    lenient = ConfidenceScorer(accept_threshold=0.3, review_threshold=0.1)
    assert strict.score(tr, n).decision == "abstained"
    assert lenient.score(tr, n).decision == "accepted"


def test_score_bounded_and_deterministic():
    tr = track("MH12AB1234", [
        ("MH12AB1234", 1.0, 1.0, 1.0, 0),
        ("MH12AB1234", 1.0, 1.0, 1.0, 1),
    ])
    n = norm("MH12AB1234", True, "standard", "MH")
    r1 = ConfidenceScorer().score(tr, n)
    r2 = ConfidenceScorer().score(tr, n)
    assert 0.0 <= r1.confidence_score <= 1.0
    assert r1.confidence_score == r2.confidence_score
    assert r1.decision == r2.decision


def test_from_config():
    config = {"confidence": {"accept_threshold": 0.3, "review_threshold": 0.1,
                             "ocr_weight": 0.35, "quality_weight": 0.15,
                             "detector_weight": 0.15, "support_weight": 0.15,
                             "validity_weight": 0.2}}
    scorer = ConfidenceScorer.from_config(config)
    assert scorer.accept_threshold == 0.3

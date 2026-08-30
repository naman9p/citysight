"""Tests for Step 11 observation construction + contract validation."""

import json

import jsonschema
import pytest

from phase1_anpr.confidence.confidence_scorer import ConfidenceResult
from phase1_anpr.normalization.plate_normalizer import NormalizationResult
from phase1_anpr.observation.observation_builder import (
    ObservationBuilder,
    PlateObservation,
)
from phase1_anpr.ocr.plate_ocr import OCRResult
from phase1_anpr.pipeline.track_processor import FusedCandidate, TrackResult

META = dict(camera_id="cam_01", timestamp="2026-08-31T10:00:00+05:30",
            model_version="phase1-anpr-0.1.0")


def make_track(best_text, rows):
    ev = [OCRResult(text=t, ocr_confidence=o, quality_score=q,
                    detector_confidence=d, frame_number=f) for (t, o, q, d, f) in rows]
    cands = [FusedCandidate(best_text, 1.0, len(ev))] if best_text else []
    return TrackResult(track_id=7, best_text=best_text, candidates=cands, evidence=ev)


def norm(text, valid):
    return NormalizationResult(text, text, valid, "standard" if valid else None,
                               "MH" if valid else None)


def conf(score, decision):
    return ConfidenceResult(score, decision, None, ["r"])


@pytest.fixture
def builder():
    return ObservationBuilder()


def test_accepted_observation_valid(builder):
    tr = make_track("MH12AB1234", [("MH12AB1234", 0.9, 0.8, 0.9, 4),
                                    ("MH12AB1234", 0.88, 0.85, 0.9, 9)])
    obs = builder.build(tr, norm("MH12AB1234", True), conf(0.9, "accepted"), **META)
    builder.validate(obs)  # must not raise
    assert obs.status == "accepted"
    assert obs.plate_raw == "MH12AB1234" and obs.plate_normalized == "MH12AB1234"
    assert obs.best_frame_number == 9      # higher-quality agreeing frame
    assert obs.quality_score == 0.85
    assert obs.camera_id == "cam_01" and obs.track_id == 7


def test_review_observation_valid(builder):
    tr = make_track("MH12AB1234", [("MH12AB1234", 0.5, 0.5, 0.5, 2)])
    obs = builder.build(tr, norm("MH12AB1234", True), conf(0.5, "review"), **META)
    assert obs.status == "review"
    assert obs.plate_normalized == "MH12AB1234"
    assert builder.is_valid(obs)


def test_abstained_asserts_no_identity(builder):
    tr = make_track(None, [])
    obs = builder.build(tr, norm("", False), conf(0.0, "abstained"), **META)
    assert obs.status == "abstained"
    assert obs.plate_raw is None and obs.plate_normalized is None
    assert obs.detector_confidence is None and obs.ocr_confidence is None
    assert obs.quality_score is None and obs.best_frame_number is None
    builder.validate(obs)


def test_schema_validation_failure_is_detected(builder):
    tr = make_track("MH12AB1234", [("MH12AB1234", 0.9, 0.8, 0.9, 1)])
    obs = builder.build(tr, norm("MH12AB1234", True), conf(0.9, "accepted"), **META)
    bad = obs.to_dict()
    bad["status"] = "definitely"          # not in enum
    assert not builder.is_valid(bad)
    with pytest.raises(jsonschema.ValidationError):
        builder.validate(bad)

    bad2 = obs.to_dict()
    bad2["confidence"] = 1.5               # out of [0, 1]
    assert not builder.is_valid(bad2)


def test_event_id_is_stable_across_serialization(builder):
    tr = make_track("MH12AB1234", [("MH12AB1234", 0.9, 0.8, 0.9, 1)])
    obs = builder.build(tr, norm("MH12AB1234", True), conf(0.9, "accepted"), **META)
    eid = obs.event_id
    assert obs.to_dict()["event_id"] == eid
    assert json.loads(obs.to_json())["event_id"] == eid
    # A retry that reuses the id keeps it identical.
    obs2 = builder.build(tr, norm("MH12AB1234", True), conf(0.9, "accepted"),
                         event_id=eid, **META)
    assert obs2.event_id == eid


def test_explicit_event_id_used(builder):
    tr = make_track("MH12AB1234", [("MH12AB1234", 0.9, 0.8, 0.9, 1)])
    obs = builder.build(tr, norm("MH12AB1234", True), conf(0.9, "accepted"),
                        event_id="evt-123", **META)
    assert obs.event_id == "evt-123"


def test_output_is_json_serializable(builder):
    tr = make_track("MH12AB1234", [("MH12AB1234", 0.9, 0.8, 0.9, 1)])
    obs = builder.build(tr, norm("MH12AB1234", True), conf(0.9, "accepted"), **META)
    text = obs.to_json()
    parsed = json.loads(text)
    assert isinstance(parsed, dict)
    assert set(parsed) == set(PlateObservation.__dataclass_fields__)

"""Tests for label-driven Phase 1 ANPR evaluation primitives."""

import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from phase1_anpr.evaluation.dataset import (
    EvaluationDatasetError,
    audit_manifest,
    load_evaluation_manifest,
)
from phase1_anpr.evaluation.metrics import (
    aligned_substitutions,
    character_accuracy,
    evaluate_detections,
    evaluate_ocr,
    intersection_over_union,
    levenshtein_distance,
)
from phase1_anpr.evaluation.runner import _character_vote, _fuse, _score_fused_text
from phase1_anpr.evaluation.runner import run_detector, run_end_to_end, run_ocr
from phase1_anpr.ocr.plate_ocr import OCRResult


def _ocr(text, confidence=0.9, quality=0.8, frame=1):
    return OCRResult(
        text=text,
        ocr_confidence=confidence,
        quality_score=quality,
        detector_confidence=0.9,
        frame_number=frame,
    )


def test_iou_and_detection_metrics_use_one_to_one_matching():
    assert intersection_over_union((0, 0, 10, 10), (5, 5, 15, 15)) == pytest.approx(25 / 175)
    result = evaluate_detections(
        {"a": [(0, 0, 10, 10)], "b": [(0, 0, 10, 10)]},
        {"a": [((0, 0, 10, 10), 0.9), ((0, 0, 10, 10), 0.8)]},
        iou_threshold=0.5,
    )
    assert result.metrics.true_positives == 1
    assert result.metrics.false_positives == 1  # duplicate prediction
    assert result.metrics.false_negatives == 1
    assert result.metrics.precision == pytest.approx(0.5)
    assert result.metrics.recall == pytest.approx(0.5)
    assert result.metrics.f1 == pytest.approx(0.5)


@pytest.mark.parametrize("box", [(0, 0, 0, 1), (1, 1, 0, 2), (1, 2, 3)])
def test_invalid_boxes_are_rejected(box):
    with pytest.raises(ValueError, match="bounding box"):
        intersection_over_union(box, (0, 0, 1, 1))


def test_ocr_metrics_use_edit_distance_not_position_zip():
    metrics = evaluate_ocr([("MH12AB1234", "MH12A1234"), ("DL1C0001", "DL1C0001")])
    assert metrics.samples == 2
    assert metrics.exact_matches == 1
    assert metrics.exact_accuracy == 0.5
    assert metrics.character_errors == 1
    assert metrics.ground_truth_characters == 18
    assert metrics.character_accuracy == pytest.approx(17 / 18)
    assert levenshtein_distance("ABC", "ADC") == 1
    assert character_accuracy("ABC", "ADC") == pytest.approx(2 / 3)


def test_substitution_alignment_excludes_insertions_and_deletions():
    assert aligned_substitutions("ABO12", "AB012") == (("O", "0"),)
    assert aligned_substitutions("ABC", "AC") == ()


def test_fusion_strategies_are_deterministic_and_label_independent():
    rows = [_ocr("MH12AB1234", 0.7, 0.9, 1),
            _ocr("MH12AB1234", 0.6, 0.8, 2),
            _ocr("MH12A81234", 0.95, 0.7, 3)]
    assert _fuse(rows, "best_frame") == "MH12AB1234"
    assert _fuse(rows, "majority") == "MH12AB1234"
    assert _fuse(rows, "confidence_weighted") == "MH12AB1234"
    assert _fuse(rows, "quality_confidence_weighted") == "MH12AB1234"
    assert _character_vote(rows) == "MH12AB1234"


def test_synthesized_character_vote_is_never_auto_accepted():
    from phase1_anpr.confidence.confidence_scorer import ConfidenceScorer
    from phase1_anpr.normalization.plate_normalizer import PlateNormalizer

    rows = [_ocr("MH0A0000", 0.9, 0.9),
            _ocr("MH1B1111", 0.9, 0.9),
            _ocr("DL0B1111", 0.9, 0.9)]
    fused = _character_vote(rows)
    assert fused not in {row.text for row in rows}
    _, decision = _score_fused_text(
        fused, rows, PlateNormalizer(), ConfidenceScorer()
    )
    assert decision == "review"


def _write_image(path, value=100):
    assert cv2.imwrite(str(path), np.full((40, 100, 3), value, dtype=np.uint8))


def _manifest(tmp_path, images):
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump({
        "name": "fixture",
        "version": "v1",
        "detection": {"images": images},
        "ocr": {"crops": [{
            "id": "crop-1", "path": "one.jpg", "text": "MH12AB1234",
            "track_id": "track-1", "frame_number": 1,
            "quality_score": 0.8, "detector_confidence": 0.9,
        }]},
    }), encoding="utf-8")
    return path


def test_manifest_resolves_paths_and_audits_duplicates_and_leakage(tmp_path):
    _write_image(tmp_path / "one.jpg")
    _write_image(tmp_path / "two.jpg")  # exact same bytes
    path = _manifest(tmp_path, [
        {"id": "image-1", "path": "one.jpg", "sequence_id": "seq",
         "split": "train", "boxes": [{"bbox": [1, 2, 20, 10]}]},
        {"id": "image-2", "path": "two.jpg", "sequence_id": "seq",
         "split": "test", "boxes": [{"bbox": [2, 3, 22, 11], "tags": ["glare"]}]},
    ])
    manifest = load_evaluation_manifest(path)
    assert manifest.detection_images[0].path.is_absolute()
    assert manifest.ocr_crops[0].expected_text == "MH12AB1234"
    report = audit_manifest(manifest)
    assert report["ground_truth_plates"] == 2
    assert report["sequence_split_leakage"] == {"seq": ["test", "train"]}
    # The OCR crop reuses one.jpg too, so all three IDs are exact duplicates.
    assert sorted(report["exact_duplicate_groups"][0]) == ["crop-1", "image-1", "image-2"]
    assert not report["suitable_for_evaluation"]
    json.dumps(report)  # machine-readable contract


def test_manifest_rejects_duplicate_ids_and_missing_files(tmp_path):
    _write_image(tmp_path / "one.jpg")
    path = _manifest(tmp_path, [
        {"id": "same", "path": "one.jpg", "boxes": []},
        {"id": "same", "path": "missing.jpg", "boxes": []},
    ])
    with pytest.raises(EvaluationDatasetError, match="duplicate detection image id"):
        load_evaluation_manifest(path)


def test_manifest_rejects_unknown_fields(tmp_path):
    path = tmp_path / "manifest.yaml"
    path.write_text("name: fixture\nversion: v1\nunexpected: true\n", encoding="utf-8")
    with pytest.raises(EvaluationDatasetError, match="unknown manifest field"):
        load_evaluation_manifest(path)


def test_detector_runner_writes_stage_correct_reports(tmp_path, monkeypatch):
    _write_image(tmp_path / "one.jpg")
    _write_image(tmp_path / "two.jpg", 150)
    manifest_path = _manifest(tmp_path, [
        {"id": "image-1", "path": "one.jpg", "sequence_id": "seq-1",
         "split": "test", "boxes": [{"bbox": [1, 2, 20, 10]}]},
        {"id": "image-2", "path": "two.jpg", "sequence_id": "seq-2",
         "split": "test", "boxes": [{"bbox": [2, 3, 22, 11]}]},
    ])
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"fixture")

    class FakeDetector:
        def __init__(self, **kwargs):
            self.image_size = kwargs["image_size"]
            self.iou_threshold = kwargs["iou_threshold"]
            self.confidence_threshold = kwargs["confidence_threshold"]

        def detect(self, image, frame_number):
            if frame_number == 0:
                return [SimpleNamespace(bbox=(1, 2, 20, 10), confidence=0.9)]
            return []

    monkeypatch.setattr("phase1_anpr.detection.detector.PlateDetector", FakeDetector)
    output = tmp_path / "reports"
    args = SimpleNamespace(
        manifest=str(manifest_path), weights=str(weights), device="cpu",
        plate_class_id=0, confidence_thresholds=(0.35,),
        nms_iou_thresholds=(0.45,), image_sizes=(640,), match_ious=(0.5,),
        primary_match_iou=0.5, max_failure_images=1,
        allow_sequence_leakage=False, output_dir=str(output),
    )
    report = run_detector(args)
    metrics = report["selected_operating_point"]["metrics"]
    assert metrics["true_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["false_positives"] == 0
    assert (output / "detector_evaluation.json").is_file()
    assert (output / "detector_evaluation.md").is_file()


def test_ocr_runner_benchmarks_preprocessing_fusion_and_coverage(tmp_path, monkeypatch):
    _write_image(tmp_path / "one.jpg")
    manifest_path = _manifest(tmp_path, [])
    model_dir = tmp_path / "ocr-model"
    model_dir.mkdir()
    (model_dir / "model.json").write_text("fixture", encoding="utf-8")

    class FakeBackend:
        def __init__(self, **kwargs):
            assert kwargs["model_dir"] == str(model_dir.resolve())

        def recognize(self, image):
            return [("MH12AB1234", 0.95)]

    monkeypatch.setattr("phase1_anpr.ocr.plate_ocr.PaddleOCRBackend", FakeBackend)
    output = tmp_path / "reports"
    args = SimpleNamespace(
        manifest=str(manifest_path),
        config="phase1_anpr/config/config.yaml",
        ocr_model_dir=str(model_dir), ocr_model_name="fixture-recognizer",
        variants=("original", "rectified"), k_values=(1, 3, 5),
        output_dir=str(output),
    )
    report = run_ocr(args)
    assert report["selected_preprocessing"]["metrics"]["exact_accuracy"] == 1.0
    assert report["selected_fusion"]["metrics"]["exact_accuracy"] == 1.0
    assert report["selected_fusion"]["coverage"] == 1.0
    assert report["selected_track_outcomes"][0]["decision"] == "accepted"
    assert (output / "ocr_evaluation.json").is_file()
    assert (output / "ocr_evaluation.md").is_file()


def test_end_to_end_runner_requires_detection_and_exact_ocr(tmp_path):
    detector_path = tmp_path / "detector.json"
    ocr_path = tmp_path / "ocr.json"
    detector_path.write_text(json.dumps({
        "dataset": "fixture", "dataset_version": "v1",
        "provenance": {"manifest_sha256": "same"},
        "selected_ground_truth_outcomes": [
            {"plate_id": "track-1", "detected": True},
            {"plate_id": "track-2", "detected": False},
        ],
    }), encoding="utf-8")
    ocr_path.write_text(json.dumps({
        "provenance": {"manifest_sha256": "same"},
        "selected_track_outcomes": [
            {"track_id": "track-1", "expected_text": "MH12AB1234",
             "predicted_text": "MH12AB1234", "correct": True,
             "decision": "accepted"},
            {"track_id": "track-2", "expected_text": "DL1C0001",
             "predicted_text": "DL1C0001", "correct": True,
             "decision": "accepted"},
        ],
    }), encoding="utf-8")
    output = tmp_path / "reports"
    report = run_end_to_end(SimpleNamespace(
        detector_report=str(detector_path), ocr_report=str(ocr_path),
        output_dir=str(output),
    ))
    metrics = report["metrics"]
    assert metrics["detector_track_coverage"] == 0.5
    assert metrics["track_ocr_exact_accuracy"] == 1.0
    assert metrics["end_to_end_exact_accuracy"] == 0.5
    assert metrics["accepted_set_accuracy"] == 1.0
    assert metrics["accepted_coverage"] == 0.5
    assert metrics["abstention_rate"] == 0.5
    assert (output / "end_to_end_evaluation.json").is_file()

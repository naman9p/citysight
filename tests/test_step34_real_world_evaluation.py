"""Focused tests for the Step 34 real-world ground-truth benchmark."""

import json

import pytest
import yaml

from phase1_anpr.evaluation.real_world import (
    RealWorldEvaluationError,
    evaluate_real_world,
    load_real_world_ground_truth,
    load_real_world_predictions,
    prepare_real_world_annotation_workspace,
)


def _source(*, source_id="cam01", camera_id="CAM_01", instances=(), frames=(),
            nonplates=(), reviewed=True):
    return {
        "source_id": source_id,
        "camera_id": camera_id,
        "video_path": f"{source_id}.mp4",
        "observations_reviewed": reviewed,
        "non_plate_observation_event_ids": list(nonplates),
        "plate_instances": list(instances),
        "frames": list(frames),
    }


def _instance(instance_id="plate-1", text="GJ04EJ5933", event_id="event-1",
              *, status_reviewed=True):
    return {
        "instance_id": instance_id,
        "plate_text": text,
        "text_evaluable": text is not None,
        "vehicle_identity": None,
        "observation_event_id": event_id,
        "observation_link_reviewed": status_reviewed,
        "notes": None,
    }


def _frame(frame_number=0, plates=()):
    return {
        "frame_number": frame_number,
        "timestamp_seconds": frame_number / 30.0,
        "reviewed": True,
        "plates": list(plates),
    }


def _plate(instance_id="plate-1", bbox=(0, 0, 10, 10)):
    return {"instance_id": instance_id, "bbox": list(bbox), "notes": None}


def _observation(event_id="event-1", text="GJ04EJ5933", status="accepted"):
    return {
        "event_id": event_id,
        "best_frame_number": 0,
        "status": status,
        "ocr_text": text,
        "plate_raw": text,
        "plate_normalized": text,
        "confidence": 0.9,
    }


def _write_inputs(tmp_path, sources, prediction_sources, *, rankings=(), cases=()):
    gt_path = tmp_path / "ground_truth.yaml"
    gt_path.write_text(yaml.safe_dump({
        "schema_version": 1,
        "benchmark_id": "benchmark",
        "sources": list(sources),
        "cross_camera_cases": list(cases),
    }, sort_keys=False), encoding="utf-8")
    prediction_path = tmp_path / "predictions.json"
    prediction_path.write_text(json.dumps({
        "schema_version": 1,
        "benchmark_id": "benchmark",
        "provenance": {"config": "fixed"},
        "sources": list(prediction_sources),
        "candidate_rankings": list(rankings),
    }), encoding="utf-8")
    return (
        load_real_world_ground_truth(gt_path),
        load_real_world_predictions(prediction_path),
    )


def _prediction_source(*, source_id="cam01", camera_id="CAM_01",
                       detections=(), observations=()):
    return {
        "source_id": source_id,
        "camera_id": camera_id,
        "detections": list(detections),
        "observations": list(observations),
    }


def _detection(bbox=(0, 0, 10, 10), confidence=0.9, frame=0):
    return {
        "frame_number": frame,
        "bbox": list(bbox),
        "confidence": confidence,
    }


def test_empty_ground_truth_has_no_claimed_accuracy_and_is_deterministic(tmp_path):
    ground_truth, predictions = _write_inputs(tmp_path, (), ())

    first = evaluate_real_world(ground_truth, predictions)
    second = evaluate_real_world(ground_truth, predictions)

    assert first == second
    assert first["detection"]["ground_truth_plate_boxes"] == 0
    assert first["detection"]["available"] is False
    assert first["ocr"]["exact_match_accuracy"] is None
    assert first["end_to_end"]["available"] is False


def test_perfect_match_reports_stage_correct_metrics(tmp_path):
    ground_truth, predictions = _write_inputs(
        tmp_path,
        [_source(instances=[_instance()], frames=[_frame(plates=[_plate()])])],
        [_prediction_source(
            detections=[_detection()], observations=[_observation()])],
    )

    report = evaluate_real_world(ground_truth, predictions)

    assert report["detection"]["primary"]["true_positives"] == 1
    assert report["detection"]["primary"]["precision"] == 1.0
    assert report["ocr"]["correct_count"] == 1
    assert report["ocr"]["evaluable_count"] == 1
    assert report["end_to_end"]["correct_count"] == 1
    assert report["end_to_end"]["exact_recognition"] == 1.0
    assert report["confidence_status"]["accepted_only_precision"] == 1.0
    assert report["confidence_status"]["coverage"] == 1.0


def test_detector_false_positive_and_false_negative_are_both_counted(tmp_path):
    ground_truth, predictions = _write_inputs(
        tmp_path,
        [_source(
            instances=[_instance(event_id=None)],
            frames=[_frame(plates=[_plate(bbox=(0, 0, 10, 10))])],
            reviewed=False,
        )],
        [_prediction_source(detections=[_detection(bbox=(20, 20, 30, 30))])],
    )

    detection = evaluate_real_world(ground_truth, predictions)["detection"]
    metrics = detection["primary"]

    assert metrics["true_positives"] == 0
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert len(detection["false_negative_boxes"]) == 1


def test_iou_boundary_is_inclusive(tmp_path):
    ground_truth, predictions = _write_inputs(
        tmp_path,
        [_source(
            instances=[_instance(event_id=None)],
            frames=[_frame(plates=[_plate(bbox=(0, 0, 10, 10))])],
            reviewed=False,
        )],
        [_prediction_source(detections=[_detection(bbox=(0, 0, 10, 5))])],
    )

    metrics = evaluate_real_world(
        ground_truth, predictions, primary_iou=0.5)["detection"]["primary"]

    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0


def test_ocr_mismatch_review_and_abstained_behavior_are_separate(tmp_path):
    instances = [
        _instance("plate-1", "GJ04EJ5933", "event-1"),
        _instance("plate-2", "MH12AB1234", "event-2"),
    ]
    frames = [_frame(plates=[
        _plate("plate-1", (0, 0, 10, 10)),
        _plate("plate-2", (20, 0, 30, 10)),
    ])]
    observations = [
        _observation("event-1", "GJ04EJ5938", "review"),
        _observation("event-2", "MH12AB1234", "abstained"),
    ]
    ground_truth, predictions = _write_inputs(
        tmp_path,
        [_source(instances=instances, frames=frames)],
        [_prediction_source(
            detections=[_detection(), _detection((20, 0, 30, 10))],
            observations=observations,
        )],
    )

    report = evaluate_real_world(ground_truth, predictions)

    assert report["ocr"]["correct_count"] == 1
    assert report["ocr"]["evaluable_count"] == 2
    assert report["ocr"]["exact_match_accuracy"] == 0.5
    assert report["ocr"]["character_confusions"] == {"3->8": 1}
    assert report["confidence_status"]["review"] == 1
    assert report["confidence_status"]["abstained"] == 1
    assert len(report["confidence_status"]["correct_sent_to_review_or_abstained"]) == 1
    assert report["confidence_status"]["accepted_only_precision"] is None


def test_duplicate_prediction_observations_are_rejected(tmp_path):
    predictions = _prediction_source(observations=[
        _observation("duplicate"), _observation("duplicate")])

    with pytest.raises(RealWorldEvaluationError, match="duplicate prediction event_id"):
        _write_inputs(tmp_path, [_source()], [predictions])


def test_annotation_preparation_never_overwrites_existing_labels(tmp_path):
    manifest = tmp_path / "ground_truth.yaml"
    manifest.write_text("manual: labels\n", encoding="utf-8")

    with pytest.raises(RealWorldEvaluationError, match="refusing to overwrite"):
        prepare_real_world_annotation_workspace(
            "scenario.yaml", "city.yaml", "config.yaml", manifest)

    assert manifest.read_text(encoding="utf-8") == "manual: labels\n"


def test_single_cross_camera_positive_does_not_produce_percentages(tmp_path):
    sources = [
        _source(
            source_id="cam01", camera_id="CAM_01",
            instances=[_instance("plate-1", event_id="event-1")],
            frames=[_frame(plates=[_plate("plate-1")])],
        ),
        _source(
            source_id="cam02", camera_id="CAM_02",
            instances=[_instance("plate-2", event_id="event-2")],
            frames=[_frame(plates=[_plate("plate-2")])],
        ),
    ]
    prediction_sources = [
        _prediction_source(
            source_id="cam01", camera_id="CAM_01",
            detections=[_detection()],
            observations=[_observation("event-1")],
        ),
        _prediction_source(
            source_id="cam02", camera_id="CAM_02",
            detections=[_detection()],
            observations=[_observation("event-2")],
        ),
    ]
    cases = [{
        "source_event_id": "event-1",
        "relevant_event_ids": ["event-2"],
    }]
    rankings = [{
        "source_event_id": "event-1",
        "candidate_event_ids": ["event-2"],
        "topology_rejections": 0,
        "travel_rejections": 0,
        "decisions": [{
            "candidate_event_id": "event-2",
            "decision": "possible_strong",
        }],
    }]
    ground_truth, predictions = _write_inputs(
        tmp_path, sources, prediction_sources, cases=cases, rankings=rankings)

    cross_camera = evaluate_real_world(
        ground_truth, predictions)["cross_camera"]

    assert cross_camera["available"] is False
    assert cross_camera["labeled_cases"] == 1
    assert cross_camera["metrics"] is None


@pytest.mark.parametrize("mutation, message", [
    (lambda source: source.update({"unknown": True}), "unknown field"),
    (lambda source: source["frames"].append(_frame(0)), "duplicate reviewed frame"),
    (lambda source: source["frames"][0]["plates"][0].update(
        {"bbox": [10, 10, 5, 5]}), "invalid coordinates"),
])
def test_malformed_annotations_are_rejected(tmp_path, mutation, message):
    source = _source(
        instances=[_instance(event_id=None)],
        frames=[_frame(plates=[_plate()])],
        reviewed=False,
    )
    mutation(source)

    with pytest.raises(RealWorldEvaluationError, match=message):
        _write_inputs(tmp_path, [source], [_prediction_source()])

"""Synthetic-only tests for the Step 43A benchmark tooling."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from phase1_anpr.evaluation.step43 import (
    FROZEN_PHASE2_REFERENCE,
    PHASE3_UNORDERED_REASON,
    Step43EvaluationError,
    evaluate_step34_by_camera,
    evaluate_step43,
    load_step43_manifest,
    paired_metric,
    run_step43_benchmark,
    step43_markdown,
)
from phase1_anpr.evaluation.real_world import (
    load_real_world_ground_truth,
    load_real_world_predictions,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value, *, yaml_file=False) -> Path:
    if yaml_file:
        path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    else:
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _instance(instance_id, event_id, text, identity):
    return {
        "instance_id": instance_id,
        "plate_text": text,
        "text_evaluable": text is not None,
        "vehicle_identity": identity,
        "observation_event_id": event_id,
        "observation_link_reviewed": True,
        "notes": None,
    }


def _frame(number, plates):
    return {
        "frame_number": number,
        "timestamp_seconds": number / 10,
        "reviewed": True,
        "plates": plates,
    }


def _plate(instance_id, bbox):
    return {"instance_id": instance_id, "bbox": bbox, "notes": None}


def _observation(event_id, frame, text, status="accepted"):
    return {
        "event_id": event_id,
        "best_frame_number": frame,
        "status": status,
        "ocr_text": text,
        "plate_raw": text,
        "plate_normalized": text,
        "confidence": 0.9,
    }


def _candidate(event_id, *, hybrid="eligible_with_evidence",
               physical="feasible", plate="both_unavailable",
               appearance="available", available=True, cosine=0.8):
    return {
        "event_id": event_id,
        "hybrid_status": hybrid,
        "physical_gate_status": physical,
        "trusted_plate_relation": plate,
        "appearance_status": appearance,
        "appearance_available": available,
        "appearance_cosine_similarity": cosine if available else None,
    }


def _fixture(tmp_path: Path, *, incomplete=False, appearance_provenance=True):
    videos = []
    for name in ("alpha.mp4", "beta.mp4"):
        path = tmp_path / name
        path.write_bytes(("synthetic-" + name).encode())
        videos.append(path)
    ground_truth = {
        "schema_version": 1,
        "benchmark_id": "synthetic-step43",
        "sources": [
            {
                "source_id": "alpha", "camera_id": "CAM_ALPHA",
                "video_path": str(videos[0]),
                "observations_reviewed": not incomplete,
                "non_plate_observation_event_ids": [],
                "plate_instances": [
                    _instance("a-readable", "event-a", "AA11AA1111", "vehicle-1"),
                ],
                "frames": [
                    _frame(0, [_plate("a-readable", [0, 0, 10, 10])]),
                    _frame(1, []),
                ],
            },
            {
                "source_id": "beta", "camera_id": "CAM_BETA",
                "video_path": str(videos[1]),
                "observations_reviewed": True,
                "non_plate_observation_event_ids": [],
                "plate_instances": [
                    _instance("b-unreadable", "event-b", None, "vehicle-1"),
                    _instance("b-readable", "event-c", "CC33CC3333", "vehicle-2"),
                ],
                "frames": [_frame(0, [
                    _plate("b-unreadable", [0, 0, 10, 10]),
                    _plate("b-readable", [20, 0, 30, 10]),
                ])],
            },
        ],
        "cross_camera_cases": [
            {"source_event_id": "event-a", "relevant_event_ids": ["event-b"]},
            {"source_event_id": "event-b", "relevant_event_ids": ["event-a"]},
        ],
    }
    gt_path = _write(tmp_path / "ground_truth.yaml", ground_truth, yaml_file=True)
    predictions = {
        "schema_version": 1,
        "benchmark_id": "synthetic-step43",
        "provenance": {"synthetic": True},
        "sources": [
            {
                "source_id": "alpha", "camera_id": "CAM_ALPHA",
                "detections": [
                    {"frame_number": 0, "bbox": [0, 0, 10, 10], "confidence": 0.9},
                    {"frame_number": 1, "bbox": [30, 30, 40, 40], "confidence": 0.8},
                ],
                "observations": [_observation("event-a", 0, "AA11AA1111")],
            },
            {
                "source_id": "beta", "camera_id": "CAM_BETA",
                "detections": [
                    {"frame_number": 0, "bbox": [0, 0, 10, 10], "confidence": 0.9},
                    {"frame_number": 0, "bbox": [20, 0, 30, 10], "confidence": 0.9},
                ],
                "observations": [
                    _observation("event-b", 0, None),
                    _observation("event-c", 0, "CC33CC3338", "review"),
                ],
            },
        ],
        "candidate_rankings": [
            {
                "source_event_id": "event-a",
                "candidate_event_ids": ["event-c", "event-b"],
                "topology_rejections": 1, "travel_rejections": 0,
                "decisions": [{"candidate_event_id": "event-b",
                               "decision": "possible_strong"}],
            },
            {
                "source_event_id": "event-b",
                "candidate_event_ids": ["event-a"],
                "topology_rejections": 0, "travel_rejections": 1,
                "decisions": [{"candidate_event_id": "event-a",
                               "decision": "possible_weak"}],
            },
        ],
    }
    prediction_path = _write(tmp_path / "phase2_predictions.json", predictions)
    phase3 = {
        "schema_version": 1,
        "benchmark_id": "synthetic-step43",
        "ground_truth_sha256": _sha(gt_path),
        "ordering_semantics": "chronological_retrieval_order",
        "cases": [
            {
                "source_event_id": "event-a",
                "candidates": [
                    _candidate("event-b"),
                    _candidate("event-c", hybrid="eligible_without_evidence",
                               appearance="both_unavailable", available=False),
                ],
                "hypotheses": {
                    "count": 2, "branch_count": 1,
                    "coverage_event_ids": ["event-b"],
                    "paths": [["event-a", "event-b"], ["event-a", "event-c"]],
                    "truncated": False,
                },
            },
            {
                "source_event_id": "event-b",
                "candidates": [
                    _candidate("event-a", hybrid="physically_impossible",
                               physical="impossible", appearance="both_unavailable",
                               available=False),
                    _candidate("event-c", hybrid="trusted_plate_contradiction",
                               plate="contradiction",
                               appearance="incompatible_provenance", available=False),
                ],
                "hypotheses": {
                    "count": 0, "branch_count": 0,
                    "coverage_event_ids": [], "paths": [], "truncated": False,
                },
            },
        ],
    }
    phase3_path = _write(tmp_path / "phase3_results.json", phase3)
    artifacts = {}
    for name in ("config", "city_config", "baseline_replay_config",
                 "benchmark_scenario"):
        path = tmp_path / f"{name}.yaml"
        path.write_text(f"synthetic: {name}\n", encoding="utf-8")
        artifacts[name] = {"path": str(path), "sha256": _sha(path)}
    appearance = (
        {
            "available": True,
            "model_id": "synthetic-model", "model_version": "1",
            "weights_sha256": "1" * 64, "preprocessing_sha256": "2" * 64,
        } if appearance_provenance else {
            "available": False, "unavailable_reason": "synthetic model not loaded",
        }
    )
    manifest = {
        "schema_version": 1,
        "benchmark_id": "synthetic-step43",
        "created_at": "2026-01-02T03:04:05+00:00",
        "evaluation_at": "2026-01-03T03:04:05+00:00",
        "ground_truth": {"path": str(gt_path), "sha256": _sha(gt_path)},
        "sources": [
            {"source_id": "alpha", "camera_id": "CAM_ALPHA",
             "video": {"path": str(videos[0]), "sha256": _sha(videos[0])},
             "metadata": {"device": "synthetic", "fps": 10}},
            {"source_id": "beta", "camera_id": "CAM_BETA",
             "video": {"path": str(videos[1]), "sha256": _sha(videos[1])},
             "metadata": {}},
        ],
        "phase2": {
            "baseline_reference": FROZEN_PHASE2_REFERENCE,
            "matcher_identifier": "citysight-phase2-frozen-step33",
            "predictions": {"path": str(prediction_path),
                            "sha256": _sha(prediction_path)},
            "artifacts": artifacts,
            "retrieval_policy": {"window_seconds": 300, "max_results": 50},
            "physical_policy": {"maximum_speed_kph": 120},
        },
        "phase3": {
            "implementation_reference": "synthetic-phase3-commit",
            "results": {"path": str(phase3_path), "sha256": _sha(phase3_path)},
            "appearance": appearance,
            "retrieval_policy": {"window_seconds": 300, "max_results": 50,
                                 "ordering": "chronological"},
            "physical_policy": {"maximum_speed_kph": 120},
            "hybrid_policy": {"implementation_id": "citysight-phase3-step37-v1",
                              "version": 1},
            "hypothesis_policy": {"implementation_id": "citysight-phase3-step41-v1",
                                  "version": 1, "max_hops": 3},
        },
        "notes": "clearly synthetic fixture",
    }
    manifest_path = _write(tmp_path / "step43.yaml", manifest, yaml_file=True)
    return manifest_path, manifest, gt_path, prediction_path, phase3_path


def test_manifest_validation_and_provenance_snapshots(tmp_path):
    path, _, _, _, _ = _fixture(tmp_path)
    manifest = load_step43_manifest(path)

    assert manifest["benchmark_id"] == "synthetic-step43"
    assert manifest["phase2"]["baseline_reference"] == "a86baa8"
    assert manifest["phase2"]["artifacts"]["config"]["sha256"]
    assert manifest["phase3"]["appearance"]["model_id"] == "synthetic-model"
    assert manifest["sources"][0]["metadata"]["fps"] == 10


def test_source_and_artifact_hash_mismatch_are_rejected(tmp_path):
    path, raw, _, _, _ = _fixture(tmp_path)
    raw["sources"][0]["video"]["sha256"] = "0" * 64
    _write(path, raw, yaml_file=True)
    with pytest.raises(Step43EvaluationError, match="SHA-256 mismatch"):
        load_step43_manifest(path)


@pytest.mark.parametrize("mutation, message", [
    (lambda row: row.update({"schema_version": 99}), "schema_version"),
    (lambda row: row["phase2"].update({"baseline_reference": "new"}),
     "baseline_reference"),
    (lambda row: row.update({"benchmark_id": ""}), "benchmark_id"),
    (lambda row: row["sources"].append(deepcopy(row["sources"][0])),
     "duplicate source_id"),
    (lambda row: row["phase2"]["retrieval_policy"].update({"bad": float("nan")}),
     "NaN or Infinity"),
])
def test_invalid_manifest_values_are_rejected(tmp_path, mutation, message):
    path, raw, _, _, _ = _fixture(tmp_path)
    mutation(raw)
    _write(path, raw, yaml_file=True)
    with pytest.raises(Step43EvaluationError, match=message):
        load_step43_manifest(path)


def test_missing_appearance_provenance_is_explicit(tmp_path):
    path, _, _, _, _ = _fixture(tmp_path, appearance_provenance=False)
    manifest = load_step43_manifest(path)
    assert manifest["phase3"]["appearance"] == {
        "available": False, "unavailable_reason": "synthetic model not loaded"}


def test_same_case_population_is_required(tmp_path):
    path, raw, _, _, phase3_path = _fixture(tmp_path)
    phase3 = json.loads(phase3_path.read_text(encoding="utf-8"))
    phase3["cases"].pop()
    _write(phase3_path, phase3)
    raw["phase3"]["results"]["sha256"] = _sha(phase3_path)
    _write(path, raw, yaml_file=True)
    with pytest.raises(Step43EvaluationError, match="case populations"):
        evaluate_step43(load_step43_manifest(path))


def test_unknown_phase3_candidate_reference_is_rejected(tmp_path):
    path, raw, _, _, phase3_path = _fixture(tmp_path)
    phase3 = json.loads(phase3_path.read_text(encoding="utf-8"))
    phase3["cases"][0]["candidates"][0]["event_id"] = "unknown"
    _write(phase3_path, phase3)
    raw["phase3"]["results"]["sha256"] = _sha(phase3_path)
    _write(path, raw, yaml_file=True)
    with pytest.raises(Step43EvaluationError, match="unknown events"):
        evaluate_step43(load_step43_manifest(path))


def test_phase3_chronological_semantics_are_required(tmp_path):
    path, raw, _, _, phase3_path = _fixture(tmp_path)
    phase3 = json.loads(phase3_path.read_text(encoding="utf-8"))
    phase3["ordering_semantics"] = "cosine_rank"
    _write(phase3_path, phase3)
    raw["phase3"]["results"]["sha256"] = _sha(phase3_path)
    _write(path, raw, yaml_file=True)
    with pytest.raises(Step43EvaluationError, match="chronological_retrieval_order"):
        evaluate_step43(load_step43_manifest(path))


def test_appearance_results_require_manifest_provenance(tmp_path):
    path, _, _, _, _ = _fixture(tmp_path, appearance_provenance=False)
    with pytest.raises(Step43EvaluationError, match="provenance marks it unavailable"):
        evaluate_step43(load_step43_manifest(path))


def test_paired_metric_delta_requires_equal_denominators_and_cases():
    valid = paired_metric(
        name="synthetic", phase2_numerator=1, phase2_denominator=2,
        phase3_numerator=2, phase3_denominator=2,
        phase2_case_ids=("a", "b"), phase3_case_ids=("a", "b"))
    assert valid["available"] is True
    assert valid["absolute_delta"] == 0.5
    assert valid["case_sets_equal"] is True

    denominator_mismatch = paired_metric(
        name="synthetic", phase2_numerator=1, phase2_denominator=2,
        phase3_numerator=2, phase3_denominator=3,
        phase2_case_ids=("a",), phase3_case_ids=("a",))
    assert denominator_mismatch["available"] is False
    assert denominator_mismatch["absolute_delta"] is None
    assert denominator_mismatch["denominators_equal"] is False

    case_mismatch = paired_metric(
        name="synthetic", phase2_numerator=1, phase2_denominator=2,
        phase3_numerator=1, phase3_denominator=2,
        phase2_case_ids=("a",), phase3_case_ids=("b",))
    assert case_mismatch["available"] is False
    assert case_mismatch["unavailable_reason"] == "evaluated case ID sets differ"


def test_unavailable_reason_is_preserved():
    row = paired_metric(
        name="unsupported", phase2_numerator=None, phase2_denominator=2,
        phase3_numerator=None, phase3_denominator=None,
        phase2_case_ids=("a",), phase3_case_ids=("a",),
        unavailable_reason="deliberately unavailable")
    assert row["unavailable_reason"] == "deliberately unavailable"


def test_step34_metrics_are_reused_per_camera_and_combined(tmp_path):
    _, _, gt_path, prediction_path, _ = _fixture(tmp_path)
    ground_truth = load_real_world_ground_truth(gt_path)
    predictions = load_real_world_predictions(prediction_path)
    report = evaluate_step34_by_camera(ground_truth, predictions)

    assert set(report["by_camera"]) == {"CAM_ALPHA", "CAM_BETA"}
    combined = report["combined"]
    assert combined["detection"]["primary"]["true_positives"] == 3
    assert combined["detection"]["primary"]["false_positives"] == 1
    assert combined["detection"]["ground_truth_plate_boxes"] == 3
    assert combined["ocr"]["evaluable_count"] == 2
    assert combined["ocr"]["correct_count"] == 1
    assert combined["end_to_end"]["evaluable_count"] == 2
    assert combined["end_to_end"]["correct_count"] == 1
    assert combined["confidence_status"]["accepted"] == 1
    assert combined["confidence_status"]["review"] == 1
    assert report["by_camera"]["CAM_ALPHA"]["detection"]["primary"][
        "false_positives"] == 1
    assert report["by_camera"]["CAM_BETA"]["detection"][
        "ground_truth_plate_boxes"] == 2


def test_unreadable_plate_is_detector_truth_not_ocr_denominator(tmp_path):
    _, _, gt_path, prediction_path, _ = _fixture(tmp_path)
    report = evaluate_step34_by_camera(
        load_real_world_ground_truth(gt_path),
        load_real_world_predictions(prediction_path))["combined"]
    assert report["detection"]["ground_truth_plate_boxes"] == 3
    assert report["ocr"]["evaluable_count"] == 2


def test_phase2_cross_camera_metrics_are_preserved(tmp_path):
    path, _, _, _, _ = _fixture(tmp_path)
    report, _, _ = evaluate_step43(load_step43_manifest(path))
    cross = report["phase2_cross_camera"]
    assert cross["available"] is True
    assert cross["metrics"]["recall_at_k"]["1"] == 0.5
    assert cross["metrics"]["mean_reciprocal_rank"] == 0.75


def test_unordered_phase3_never_gets_rank_metrics(tmp_path):
    path, _, _, _, _ = _fixture(tmp_path)
    report, _, _ = evaluate_step43(load_step43_manifest(path))
    assert {row["metric"] for row in report["paired_cross_camera_metrics"]} == {
        "recall_at_1", "recall_at_3", "recall_at_5",
        "hit_rate_at_1", "hit_rate_at_3", "hit_rate_at_5",
        "mean_reciprocal_rank"}
    assert all(not row["available"] for row in report["paired_cross_camera_metrics"])
    assert all(row["unavailable_reason"] == PHASE3_UNORDERED_REASON
               for row in report["paired_cross_camera_metrics"])


def test_phase3_evidence_and_hypothesis_diagnostics(tmp_path):
    path, _, _, _, _ = _fixture(tmp_path)
    report, ablations, failures = evaluate_step43(load_step43_manifest(path))
    diagnostics = report["phase3_diagnostics"]
    assert diagnostics["candidate_count"] == 4
    assert diagnostics["appearance_available_count"] == 1
    assert diagnostics["appearance_incompatible_count"] == 1
    assert diagnostics["physical_rejection_count"] == 1
    assert diagnostics["trusted_plate_contradiction_count"] == 1
    assert diagnostics["eligible_with_evidence_count"] == 1
    assert diagnostics["eligible_without_evidence_count"] == 1
    assert diagnostics["hypothesis_count"] == 2
    assert diagnostics["branch_count"] == 1
    assert diagnostics["hypothesis_coverage"] == 0.5
    assert set(report["phase3_diagnostics_by_camera"]) == {
        "CAM_ALPHA", "CAM_BETA"}
    assert report["phase3_diagnostics_by_camera"]["CAM_ALPHA"]["case_count"] == 1
    assert any(row["status"] == "unsupported" for row in ablations["ablations"])
    assert failures["detector_false_positives"]
    assert failures["ocr_wrong_text"]
    assert isinstance(failures["phase2_phase3_disagreements"], list)


def test_incomplete_ground_truth_prevents_final_claim_and_numbers(tmp_path):
    path, _, _, _, _ = _fixture(tmp_path, incomplete=True)
    report, _, failures = evaluate_step43(load_step43_manifest(path))
    assert report["final_claim_ready"] is False
    assert report["phase1_shared_metrics"] is None
    assert report["phase3_diagnostics"] is None
    assert failures["unavailable"]
    assert "Final claim ready: `false`" in step43_markdown(report)


def test_evaluation_does_not_mutate_loaded_inputs(tmp_path):
    path, _, gt_path, prediction_path, _ = _fixture(tmp_path)
    ground_truth = load_real_world_ground_truth(gt_path)
    predictions = load_real_world_predictions(prediction_path)
    before_gt = repr(ground_truth)
    before_predictions = repr(predictions)
    evaluate_step34_by_camera(ground_truth, predictions)
    assert repr(ground_truth) == before_gt
    assert repr(predictions) == before_predictions
    manifest = load_step43_manifest(path)
    before_manifest = deepcopy(manifest)
    evaluate_step43(manifest)
    assert manifest == before_manifest


def test_evaluator_does_not_invoke_production_components(tmp_path, monkeypatch):
    from phase2_city.candidate_collection import ReplayCandidateCollector
    from phase2_city.cross_camera_matching import CrossCameraCandidateMatcher
    from phase3_city.historical_candidate_retrieval import HistoricalCandidateRetriever
    from phase3_city.hybrid_matching import Phase3HybridCandidateMatcher
    from phase3_city.query_history import SQLiteCandidateQueryHistoryRepository
    from phase3_city.trajectory_hypotheses import TrajectoryHypothesisBuilder

    invoked = []

    def forbidden(*args, **kwargs):
        invoked.append((args, kwargs))
        raise AssertionError("production component was invoked")

    monkeypatch.setattr(CrossCameraCandidateMatcher, "compare", forbidden)
    monkeypatch.setattr(ReplayCandidateCollector, "collect", forbidden)
    monkeypatch.setattr(Phase3HybridCandidateMatcher, "compare", forbidden)
    monkeypatch.setattr(HistoricalCandidateRetriever, "retrieve", forbidden)
    monkeypatch.setattr(SQLiteCandidateQueryHistoryRepository, "save_query", forbidden)
    monkeypatch.setattr(TrajectoryHypothesisBuilder, "build", forbidden)

    path, _, _, _, _ = _fixture(tmp_path)
    report, _, _ = evaluate_step43(load_step43_manifest(path))
    assert report["final_claim_ready"] is True
    assert invoked == []


def test_reports_are_deterministic_finite_and_atomically_written(tmp_path):
    path, _, _, _, _ = _fixture(tmp_path)
    output = tmp_path / "outputs"
    first = run_step43_benchmark(path, output)
    first_bytes = {item.name: item.read_bytes() for item in output.iterdir()}
    second = run_step43_benchmark(path, output)
    second_bytes = {item.name: item.read_bytes() for item in output.iterdir()}

    assert first == second
    assert first_bytes == second_bytes
    assert set(first_bytes) == {
        "step43_final_benchmark.json", "step43_final_benchmark.md",
        "step43_ablation.json", "step43_ablation.md", "step43_failures.json"}
    assert not list(output.glob("*.tmp"))
    assert b"NaN" not in first_bytes["step43_final_benchmark.json"]
    assert b"Infinity" not in first_bytes["step43_final_benchmark.json"]
    assert step43_markdown(first) == step43_markdown(first)


def test_current_protected_step34_paths_are_not_inputs(tmp_path):
    path, _, _, _, _ = _fixture(tmp_path)
    manifest = load_step43_manifest(path)
    serialized = json.dumps(manifest)
    assert "step34_ground_truth.yaml" not in serialized
    assert "step34_frames" not in serialized
    assert "step34_quality_check" not in serialized


def test_cli_parser_exposes_step43_command():
    from phase1_anpr.evaluation.runner import build_parser

    args = build_parser().parse_args([
        "step43", "--manifest", "future.yaml", "--output-dir", "future-output"])
    assert args.command == "step43"
    assert args.manifest == "future.yaml"
    assert args.output_dir == "future-output"

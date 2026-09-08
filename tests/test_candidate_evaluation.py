"""Step 30: deterministic evaluation of Step 28 candidate retrieval."""

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
import math

import pytest

from phase1_anpr.observation.observation_builder import PlateObservation
from phase1_anpr.pipeline.anpr_pipeline import PipelineResult
from phase1_anpr.replay import ReplaySource, SourceReplayResult
from phase2_city.candidate_collection import (
    CandidateCollectionPolicy,
    CandidateCollectionResult,
    ReplayCandidateCollector,
)
from phase2_city.candidate_evaluation import (
    CandidateEvaluationError,
    CandidateEvaluationPolicy,
    CandidateGroundTruthCase,
    ReplayCandidateEvaluator,
)
from phase2_city.cross_camera_matching import (
    CrossCameraCandidateMatcher,
    CrossCameraCandidateResult,
    CrossCameraMatchPolicy,
    FingerprintObservation,
)
from phase2_city.fingerprint import FingerprintComparison, VehicleFingerprint
from phase2_city.graph import CityCameraGraph
from phase2_city.models import Camera, CameraLink
from phase2_city.scenario import ScenarioResult


T0 = datetime(2026, 9, 9, 10, 0, tzinfo=timezone.utc)


def _observation(event_id, camera_id="CAM_A", seconds=0, plate="GJ01AB1234"):
    return PlateObservation(
        event_id=event_id,
        camera_id=camera_id,
        track_id=1,
        timestamp=(T0 + timedelta(seconds=seconds)).isoformat(),
        plate_raw=plate,
        plate_normalized=plate,
        confidence=0.9,
        status="accepted",
        detector_confidence=0.9,
        ocr_confidence=0.9,
        quality_score=0.8,
        best_frame_number=1,
        plate_image_path=None,
        model_version="phase1-anpr-0.1.0",
    )


def _pipeline(event_id, camera_id="CAM_A", seconds=0, plate="GJ01AB1234"):
    return PipelineResult(
        observation=_observation(event_id, camera_id, seconds, plate),
        inserted=True,
        alerts=[],
    )


def _scenario(*groups, scenario_id="evaluation-scenario"):
    source_results = []
    for index, group in enumerate(groups, start=1):
        results = list(group)
        camera_id = (results[0].observation.camera_id
                     if results else f"CAM_{index}")
        source_results.append(SourceReplayResult(
            source=ReplaySource(
                video_path=f"source-{index}.mp4",
                camera_id=camera_id,
                source_id=f"source-{index}",
            ),
            results=results,
        ))
    return ScenarioResult(
        scenario_id=scenario_id,
        source_results=source_results,
    )


def _fingerprint_observation(event_id, camera_id="CAM_B", seconds=1):
    return FingerprintObservation(
        event_id=event_id,
        camera_id=camera_id,
        timestamp=T0 + timedelta(seconds=seconds),
        observation_status="accepted",
        fingerprint=VehicleFingerprint(),
    )


def _match_result(source, event_id, decision="possible_strong", index=1):
    candidate = _fingerprint_observation(
        event_id, camera_id=f"CAM_{index + 1}", seconds=index)
    comparison = FingerprintComparison(
        overall_similarity=(0.8 if decision.startswith("possible") else None),
        evidence_coverage=(0.6 if decision.startswith("possible") else 0.0),
        matched_weight=0.0,
        contradicted_weight=0.0,
        net_evidence=0.0,
        outcome=("consistent" if decision.startswith("possible")
                 else "insufficient_evidence"),
        field_comparisons=(),
    )
    return CrossCameraCandidateResult(
        source=source,
        candidate=candidate,
        fingerprint_comparison=comparison,
        topology_status="direct_link",
        link=None,
        elapsed_seconds=float(index),
        minimum_travel_seconds=0.0,
        travel_status="feasible",
        decision=decision,
        reasons=(f"decision: {decision}",),
    )


def _collection_result(
        source_event_id,
        candidate_event_ids=(),
        *,
        rejected_event_ids=(),
        truncated=False,
):
    source = _fingerprint_observation(
        source_event_id, camera_id="CAM_A", seconds=0)
    candidates = tuple(
        _match_result(source, event_id, index=index)
        for index, event_id in enumerate(candidate_event_ids, start=1)
    )
    rejected = tuple(
        _match_result(
            source, event_id, decision="insufficient_evidence",
            index=index + len(candidates),
        )
        for index, event_id in enumerate(rejected_event_ids, start=1)
    )
    ranked_results = candidates + rejected
    return CandidateCollectionResult(
        scenario_id="evaluation-scenario",
        source=source,
        total_observations_considered=10,
        structurally_eligible_count=len(ranked_results) + int(truncated),
        truncated=truncated,
        ranked_results=ranked_results,
        candidates=candidates,
    )


class RecordingCollector:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def collect(self, scenario_result, source_event_id):
        self.calls.append((scenario_result, source_event_id))
        return self.results[source_event_id]


def _evaluate(scenario, cases, results, k_values=(1, 3, 5)):
    collector = RecordingCollector(results)
    evaluator = ReplayCandidateEvaluator(
        collector, CandidateEvaluationPolicy(k_values))
    return evaluator.evaluate(scenario, cases), collector


def _all_events(*event_ids):
    return _scenario(tuple(
        _pipeline(event_id, camera_id=f"CAM_{index + 1}", seconds=index)
        for index, event_id in enumerate(event_ids)
    ))


def test_evaluation_policy_is_frozen_and_canonicalizes_k_order():
    policy = CandidateEvaluationPolicy((5, 1, 3))
    assert policy.k_values == (1, 3, 5)
    assert CandidateEvaluationPolicy((1,)).k_values == (1,)
    assert policy.to_dict() == {"k_values": [1, 3, 5]}
    with pytest.raises(FrozenInstanceError):
        policy.k_values = (1,)


@pytest.mark.parametrize("value", [(), [], None])
def test_empty_or_non_tuple_k_values_are_rejected(value):
    with pytest.raises(CandidateEvaluationError, match="k_values"):
        CandidateEvaluationPolicy(value)


@pytest.mark.parametrize("value", [0, -1, True, 1.0, "1"])
def test_invalid_k_value_is_rejected(value):
    with pytest.raises(CandidateEvaluationError, match="k_values"):
        CandidateEvaluationPolicy((1, value))


def test_duplicate_k_values_are_rejected():
    with pytest.raises(CandidateEvaluationError, match="duplicate K"):
        CandidateEvaluationPolicy((1, 3, 1))


def test_ground_truth_is_frozen_and_canonicalizes_relevant_ids():
    case = CandidateGroundTruthCase("source", ("relevant-b", "relevant-a"))
    assert case.relevant_event_ids == ("relevant-a", "relevant-b")
    assert case.to_dict() == {
        "source_event_id": "source",
        "relevant_event_ids": ["relevant-a", "relevant-b"],
    }
    with pytest.raises(FrozenInstanceError):
        case.source_event_id = "other"


@pytest.mark.parametrize("source_event_id", ["", "   ", None, 1])
def test_invalid_ground_truth_source_is_rejected(source_event_id):
    with pytest.raises(CandidateEvaluationError, match="source_event_id"):
        CandidateGroundTruthCase(source_event_id, ("relevant",))


@pytest.mark.parametrize("relevant_event_ids", [(), [], None])
def test_empty_or_non_tuple_relevant_ids_are_rejected(relevant_event_ids):
    with pytest.raises(CandidateEvaluationError, match="relevant_event_ids"):
        CandidateGroundTruthCase("source", relevant_event_ids)


@pytest.mark.parametrize("event_id", ["", "   ", None, 1])
def test_invalid_relevant_event_id_is_rejected(event_id):
    with pytest.raises(CandidateEvaluationError, match="relevant_event_ids"):
        CandidateGroundTruthCase("source", (event_id,))


def test_duplicate_relevant_ids_and_source_as_relevant_are_rejected():
    with pytest.raises(CandidateEvaluationError, match="duplicate relevant"):
        CandidateGroundTruthCase("source", ("event", "event"))
    with pytest.raises(CandidateEvaluationError, match="cannot appear"):
        CandidateGroundTruthCase("source", ("source",))


def test_missing_source_label_is_rejected_before_collection():
    scenario = _all_events("real-source", "relevant")
    collector = RecordingCollector({})
    evaluator = ReplayCandidateEvaluator(
        collector, CandidateEvaluationPolicy((1,)))
    with pytest.raises(CandidateEvaluationError, match="source event.*missing"):
        evaluator.evaluate(scenario, (
            CandidateGroundTruthCase("missing", ("relevant",)),
        ))
    assert collector.calls == []


def test_missing_relevant_label_is_rejected_before_collection():
    scenario = _all_events("source")
    collector = RecordingCollector({})
    evaluator = ReplayCandidateEvaluator(
        collector, CandidateEvaluationPolicy((1,)))
    with pytest.raises(CandidateEvaluationError, match="relevant event.*missing"):
        evaluator.evaluate(scenario, (
            CandidateGroundTruthCase("source", ("missing",)),
        ))
    assert collector.calls == []


def test_duplicate_scenario_event_ids_are_rejected_globally():
    scenario = _scenario(
        (_pipeline("duplicate", "CAM_A"),),
        (_pipeline("duplicate", "CAM_B"),),
    )
    evaluator = ReplayCandidateEvaluator(
        RecordingCollector({}), CandidateEvaluationPolicy((1,)))
    with pytest.raises(CandidateEvaluationError, match="duplicate scenario"):
        evaluator.evaluate(scenario, (
            CandidateGroundTruthCase("duplicate", ("other",)),
        ))


def test_duplicate_ground_truth_sources_are_rejected_before_collection():
    scenario = _all_events("source", "relevant-a", "relevant-b")
    collector = RecordingCollector({})
    evaluator = ReplayCandidateEvaluator(
        collector, CandidateEvaluationPolicy((1,)))
    with pytest.raises(CandidateEvaluationError, match="duplicate ground-truth"):
        evaluator.evaluate(scenario, (
            CandidateGroundTruthCase("source", ("relevant-a",)),
            CandidateGroundTruthCase("source", ("relevant-b",)),
        ))
    assert collector.calls == []


def test_empty_evaluation_set_is_rejected():
    evaluator = ReplayCandidateEvaluator(
        RecordingCollector({}), CandidateEvaluationPolicy((1,)))
    with pytest.raises(CandidateEvaluationError, match="at least one case"):
        evaluator.evaluate(_all_events("source"), ())


def test_recall_at_k_uses_relevant_item_micro_definition():
    scenario = _all_events(
        "source-a", "source-b", "relevant-a", "relevant-b",
        "relevant-c", "wrong-x", "wrong-y",
    )
    cases = (
        CandidateGroundTruthCase(
            "source-a", ("relevant-a", "relevant-b")),
        CandidateGroundTruthCase("source-b", ("relevant-c",)),
    )
    results = {
        "source-a": _collection_result(
            "source-a", ("relevant-a", "wrong-x", "wrong-y", "relevant-b")),
        "source-b": _collection_result(
            "source-b", ("wrong-x", "relevant-c", "wrong-y")),
    }
    summary, _ = _evaluate(scenario, cases, results)

    assert summary.total_relevant_events == 3
    assert dict(summary.recall_at_k) == {
        1: pytest.approx(1 / 3),
        3: pytest.approx(2 / 3),
        5: 1.0,
    }
    assert dict(summary.hit_rate_at_k) == {1: 0.5, 3: 1.0, 5: 1.0}
    assert summary.mean_reciprocal_rank == 0.75
    assert summary.case_results[0].retrieved_relevant_at_k == (
        (1, 1), (3, 1), (5, 2))


def test_hit_rate_and_mrr_use_first_relevant_candidate_rank():
    scenario = _all_events(
        "source-a", "source-b", "source-c", "relevant-a", "relevant-b",
        "relevant-c", "wrong-x", "wrong-y",
    )
    cases = (
        CandidateGroundTruthCase("source-a", ("relevant-a",)),
        CandidateGroundTruthCase("source-b", ("relevant-b",)),
        CandidateGroundTruthCase("source-c", ("relevant-c",)),
    )
    results = {
        "source-a": _collection_result("source-a", ("relevant-a",)),
        "source-b": _collection_result(
            "source-b", ("wrong-x", "relevant-b", "relevant-a")),
        "source-c": _collection_result(
            "source-c", ("wrong-x", "wrong-y")),
    }
    summary, _ = _evaluate(scenario, cases, results, k_values=(1, 3))

    assert [case.first_relevant_rank for case in summary.case_results] == [
        1, 2, None
    ]
    assert dict(summary.hit_rate_at_k) == {
        1: pytest.approx(1 / 3),
        3: pytest.approx(2 / 3),
    }
    assert summary.mean_reciprocal_rank == 0.5


def test_k_larger_than_available_candidates_uses_available_output():
    scenario = _all_events("source", "wrong", "relevant")
    summary, _ = _evaluate(
        scenario,
        (CandidateGroundTruthCase("source", ("relevant",)),),
        {"source": _collection_result("source", ("wrong", "relevant"))},
        k_values=(5,),
    )
    assert dict(summary.recall_at_k) == {5: 1.0}
    assert dict(summary.hit_rate_at_k) == {5: 1.0}


def test_rejected_diagnostic_ground_truth_result_is_a_retrieval_miss():
    scenario = _all_events("source", "relevant")
    summary, _ = _evaluate(
        scenario,
        (CandidateGroundTruthCase("source", ("relevant",)),),
        {"source": _collection_result(
            "source", rejected_event_ids=("relevant",))},
        k_values=(1,),
    )
    assert summary.case_results[0].candidate_event_ids == ()
    assert summary.case_results[0].evaluated_count == 1
    assert summary.case_results[0].first_relevant_rank is None
    assert dict(summary.recall_at_k) == {1: 0.0}
    assert dict(summary.hit_rate_at_k) == {1: 0.0}


def test_step28_candidate_order_is_preserved_for_rank_measurement():
    scenario = _all_events("source", "wrong-x", "wrong-y", "relevant")
    summary, _ = _evaluate(
        scenario,
        (CandidateGroundTruthCase("source", ("relevant",)),),
        {"source": _collection_result(
            "source", ("wrong-x", "wrong-y", "relevant"))},
        k_values=(1, 3),
    )
    case = summary.case_results[0]
    assert case.candidate_event_ids == ("wrong-x", "wrong-y", "relevant")
    assert case.first_relevant_rank == 3
    assert dict(summary.recall_at_k) == {1: 0.0, 3: 1.0}


def test_truncation_statistics_are_preserved_without_failing():
    scenario = _all_events("source-a", "source-b", "relevant-a", "relevant-b")
    cases = (
        CandidateGroundTruthCase("source-a", ("relevant-a",)),
        CandidateGroundTruthCase("source-b", ("relevant-b",)),
    )
    results = {
        "source-a": _collection_result(
            "source-a", ("relevant-a",), truncated=True),
        "source-b": _collection_result(
            "source-b", ("relevant-b",), truncated=False),
    }
    summary, _ = _evaluate(scenario, cases, results, k_values=(1,))
    assert summary.truncated_case_count == 1
    assert summary.truncated_case_rate == 0.5
    assert [case.truncated for case in summary.case_results] == [True, False]


def test_truncated_miss_remains_in_all_metric_denominators():
    scenario = _all_events("source-a", "source-b", "relevant-a", "relevant-b")
    cases = (
        CandidateGroundTruthCase("source-a", ("relevant-a",)),
        CandidateGroundTruthCase("source-b", ("relevant-b",)),
    )
    results = {
        "source-a": _collection_result("source-a", truncated=True),
        "source-b": _collection_result("source-b", ("relevant-b",)),
    }
    summary, _ = _evaluate(scenario, cases, results, k_values=(1,))

    assert dict(summary.recall_at_k) == {1: 0.5}
    assert dict(summary.hit_rate_at_k) == {1: 0.5}
    assert summary.mean_reciprocal_rank == 0.5
    assert summary.truncated_case_count == 1
    assert summary.truncated_case_rate == 0.5


def test_zero_candidates_produces_finite_zero_metrics():
    scenario = _all_events("source", "relevant")
    summary, _ = _evaluate(
        scenario,
        (CandidateGroundTruthCase("source", ("relevant",)),),
        {"source": _collection_result("source")},
    )
    assert dict(summary.recall_at_k) == {1: 0.0, 3: 0.0, 5: 0.0}
    assert dict(summary.hit_rate_at_k) == {1: 0.0, 3: 0.0, 5: 0.0}
    assert summary.mean_reciprocal_rank == 0.0
    assert summary.truncated_case_rate == 0.0
    assert all(math.isfinite(value) for _, value in summary.recall_at_k)
    assert all(math.isfinite(value) for _, value in summary.hit_rate_at_k)


def test_identical_unlabeled_plate_does_not_manufacture_ground_truth():
    scenario = _scenario((
        _pipeline("source", "CAM_A", 0, plate="GJ01AB1234"),
        _pipeline("same-plate", "CAM_B", 10, plate="GJ01AB1234"),
        _pipeline("labeled-truth", "CAM_C", 20, plate="MH12CD5678"),
    ))
    summary, _ = _evaluate(
        scenario,
        (CandidateGroundTruthCase("source", ("labeled-truth",)),),
        {"source": _collection_result("source", ("same-plate",))},
        k_values=(1,),
    )
    assert summary.case_results[0].candidate_event_ids == ("same-plate",)
    assert dict(summary.recall_at_k) == {1: 0.0}


def test_case_order_metrics_and_serialization_are_permutation_invariant():
    scenario = _all_events(
        "source-a", "source-b", "relevant-a", "relevant-b", "wrong")
    case_a = CandidateGroundTruthCase("source-a", ("relevant-a",))
    case_b = CandidateGroundTruthCase("source-b", ("relevant-b",))
    results = {
        "source-a": _collection_result("source-a", ("relevant-a",)),
        "source-b": _collection_result("source-b", ("wrong", "relevant-b")),
    }

    first, _ = _evaluate(
        scenario, (case_b, case_a), results, k_values=(5, 1, 3))
    second, _ = _evaluate(
        scenario, (case_a, case_b), results, k_values=(3, 5, 1))
    assert [case.source_event_id for case in first.case_results] == [
        "source-a", "source-b"
    ]
    assert first == second
    assert first.to_dict() == second.to_dict()
    assert list(first.to_dict()["recall_at_k"]) == ["1", "3", "5"]


def test_collector_is_called_once_per_case_and_never_by_serialization():
    scenario = _all_events("source-a", "source-b", "relevant-a", "relevant-b")
    cases = (
        CandidateGroundTruthCase("source-b", ("relevant-b",)),
        CandidateGroundTruthCase("source-a", ("relevant-a",)),
    )
    results = {
        "source-a": _collection_result("source-a", ("relevant-a",)),
        "source-b": _collection_result("source-b", ("relevant-b",)),
    }
    summary, collector = _evaluate(scenario, cases, results, k_values=(1,))
    assert [source_id for _, source_id in collector.calls] == [
        "source-a", "source-b"
    ]
    summary.to_dict()
    summary.to_dict()
    assert len(collector.calls) == 2


def test_case_and_summary_results_are_frozen_with_immutable_collections():
    scenario = _all_events("source", "relevant")
    summary, _ = _evaluate(
        scenario,
        (CandidateGroundTruthCase("source", ("relevant",)),),
        {"source": _collection_result("source", ("relevant",))},
        k_values=(1,),
    )
    assert isinstance(summary.case_results, tuple)
    assert isinstance(summary.recall_at_k, tuple)
    assert isinstance(summary.case_results[0].candidate_event_ids, tuple)
    with pytest.raises(FrozenInstanceError):
        summary.total_cases = 2
    with pytest.raises(FrozenInstanceError):
        summary.case_results[0].truncated = True


def test_serialization_has_stable_shape_and_no_sets():
    scenario = _all_events("source", "relevant")
    summary, _ = _evaluate(
        scenario,
        (CandidateGroundTruthCase("source", ("relevant",)),),
        {"source": _collection_result("source", ("relevant",))},
        k_values=(3, 1),
    )
    serialized = summary.to_dict()
    assert list(serialized) == [
        "scenario_id", "policy", "total_cases", "total_relevant_events",
        "recall_at_k", "hit_rate_at_k", "mean_reciprocal_rank",
        "truncated_case_count", "truncated_case_rate", "case_results",
    ]
    assert serialized["policy"] == {"k_values": [1, 3]}
    assert list(serialized["recall_at_k"]) == ["1", "3"]
    assert not _contains_set(serialized)


def _contains_set(value):
    if isinstance(value, set):
        return True
    if isinstance(value, dict):
        return any(_contains_set(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_set(item) for item in value)
    return False


def test_real_step27_step28_step30_integration():
    cameras = [
        Camera("CAM_A", "Camera A", 28.6, 77.2, "Road", 90.0),
        Camera("CAM_B", "Camera B", 28.7, 77.3, "Road", 90.0),
    ]
    graph = CityCameraGraph(cameras, [
        CameraLink("CAM_A", "CAM_B", 1000.0, "Road", "eastbound"),
    ])
    scenario = _scenario(
        (_pipeline("source", "CAM_A", 0),),
        (_pipeline("relevant", "CAM_B", 100),),
    )
    collector = ReplayCandidateCollector(
        CrossCameraCandidateMatcher(
            graph, CrossCameraMatchPolicy(maximum_speed_kph=36)),
        CandidateCollectionPolicy(
            max_time_delta_seconds=300,
            max_candidates=5,
        ),
    )
    summary = ReplayCandidateEvaluator(
        collector, CandidateEvaluationPolicy((1, 3))
    ).evaluate(scenario, (
        CandidateGroundTruthCase("source", ("relevant",)),
    ))

    assert dict(summary.recall_at_k) == {1: 1.0, 3: 1.0}
    assert dict(summary.hit_rate_at_k) == {1: 1.0, 3: 1.0}
    assert summary.mean_reciprocal_rank == 1.0
    assert summary.case_results[0].candidate_event_ids == ("relevant",)
    assert summary.case_results[0].evaluated_count == 1

"""Step 28: bounded replay candidate collection and deterministic ranking."""

import math
import random
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone

import pytest

from phase1_anpr.observation.observation_builder import PlateObservation
from phase1_anpr.pipeline.anpr_pipeline import PipelineResult
from phase1_anpr.replay import ReplaySource, SourceReplayResult
from phase2_city import (
    Camera,
    CameraLink,
    CandidateCollectionError,
    CandidateCollectionPolicy,
    CandidateCollectionResult,
    CityCameraGraph,
    CrossCameraCandidateMatcher,
    CrossCameraCandidateResult,
    CrossCameraMatchPolicy,
    FingerprintComparison,
    ReplayCandidateCollector,
    ScenarioResult,
    VehicleAttributes,
    VehicleFingerprint,
)


T0 = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)


def _timestamp(seconds):
    if seconds is None:
        return None
    return (T0 + timedelta(seconds=seconds)).isoformat()


def _observation(
        event_id,
        camera_id="CAM_A",
        seconds=0,
        *,
        status="accepted",
        plate="GJ01AB1234",
        timestamp_marker=False,
):
    abstained = status == "abstained"
    timestamp = seconds if timestamp_marker else _timestamp(seconds)
    return PlateObservation(
        event_id=event_id,
        camera_id=camera_id,
        track_id=1,
        timestamp=timestamp,
        plate_raw=None if abstained else plate,
        plate_normalized=None if abstained else plate,
        confidence=0.9,
        status=status,
        detector_confidence=None if abstained else 0.9,
        ocr_confidence=None if abstained else 0.9,
        quality_score=None if abstained else 0.8,
        best_frame_number=None if abstained else 5,
        plate_image_path=None,
        model_version="phase1-anpr-0.1.0",
    )


def _pipeline(observation, fingerprint=None, vehicle_attributes=None):
    return PipelineResult(
        observation=observation,
        inserted=True,
        alerts=[],
        vehicle_attributes=vehicle_attributes,
        vehicle_fingerprint=fingerprint,
    )


def _source_result(source_id, results, camera_id="REPLAY_METADATA_ONLY"):
    return SourceReplayResult(
        source=ReplaySource(
            video_path=f"{source_id}.mp4",
            camera_id=camera_id,
            source_id=source_id,
        ),
        results=list(results),
    )


def _scenario(*source_results, scenario_id="scenario-28"):
    return ScenarioResult(
        scenario_id=scenario_id,
        source_results=list(source_results),
    )


def _camera(camera_id):
    return Camera(
        camera_id=camera_id,
        name=f"Camera {camera_id}",
        latitude=28.6,
        longitude=77.2,
        road_name="Demo Road",
        heading_deg=90.0,
    )


def _link(source="CAM_A", candidate="CAM_B", distance_m=1000.0):
    return CameraLink(
        from_camera_id=source,
        to_camera_id=candidate,
        distance_m=distance_m,
        road_name="Demo Road",
        travel_direction="eastbound",
    )


def _graph(*links):
    return CityCameraGraph(
        [_camera(name) for name in ("CAM_A", "CAM_B", "CAM_C", "CAM_D")],
        links,
    )


def _real_matcher(*, graph=None, speed=36.0):
    return CrossCameraCandidateMatcher(
        graph if graph is not None else _graph(_link()),
        CrossCameraMatchPolicy(maximum_speed_kph=speed),
    )


class RecordingMatcher:
    """Small matcher fake returning real Step 27 result models."""

    def __init__(self, specs=None):
        self.specs = specs or {}
        self.calls = []

    def compare(self, source, candidate):
        self.calls.append((source, candidate))
        spec = self.specs.get(candidate.event_id, {})
        similarity = spec.get("similarity", 0.8)
        coverage = spec.get("coverage", 1.0)
        comparison = FingerprintComparison(
            overall_similarity=similarity,
            evidence_coverage=coverage,
            matched_weight=coverage,
            contradicted_weight=0.0,
            net_evidence=coverage,
            outcome="consistent" if similarity is not None
            else "insufficient_evidence",
            field_comparisons=(),
        )
        result_candidate = candidate
        if "result_timestamp" in spec:
            result_candidate = replace(
                candidate, timestamp=spec["result_timestamp"])
        return CrossCameraCandidateResult(
            source=source,
            candidate=result_candidate,
            fingerprint_comparison=comparison,
            topology_status=spec.get("topology_status", "direct_link"),
            link=None,
            elapsed_seconds=spec.get(
                "elapsed",
                (candidate.timestamp - source.timestamp).total_seconds(),
            ),
            minimum_travel_seconds=None,
            travel_status=spec.get("travel_status", "feasible"),
            decision=spec.get("decision", "possible_strong"),
            reasons=(spec.get("reason", f"reason for {candidate.event_id}"),),
        )


def _collector(matcher=None, *, window=300.0, limit=10):
    return ReplayCandidateCollector(
        matcher=matcher or RecordingMatcher(),
        policy=CandidateCollectionPolicy(window, limit),
    )


def _basic_scenario(*candidates):
    source = _pipeline(_observation("source", "CAM_A", 0))
    return _scenario(
        _source_result("source-video", [source]),
        _source_result("candidate-video", [_pipeline(item) for item in candidates]),
    )


# --- policy and model validation -------------------------------------------


def test_policy_is_required_validated_and_frozen():
    policy = CandidateCollectionPolicy(30, 2)
    assert policy.max_time_delta_seconds == 30.0
    assert policy.max_candidates == 2
    assert policy.to_dict() == {
        "max_time_delta_seconds": 30.0,
        "max_candidates": 2,
    }
    with pytest.raises(FrozenInstanceError):
        policy.max_candidates = 3


@pytest.mark.parametrize("value", [0, -1, math.nan, math.inf, -math.inf,
                                   True, "30", None])
def test_invalid_time_window_is_rejected(value):
    with pytest.raises(CandidateCollectionError,
                       match="max_time_delta_seconds"):
        CandidateCollectionPolicy(value, 1)


@pytest.mark.parametrize("value", [0, -1, True, 1.0, "1", None])
def test_invalid_candidate_limit_is_rejected(value):
    with pytest.raises(CandidateCollectionError, match="max_candidates"):
        CandidateCollectionPolicy(30, value)


def test_result_is_frozen_and_serialization_is_deterministic():
    result = _collector().collect(
        _basic_scenario(_observation("candidate", "CAM_B", 10)),
        "source",
    )
    assert isinstance(result, CandidateCollectionResult)
    with pytest.raises(FrozenInstanceError):
        result.truncated = True
    assert result.to_dict() == result.to_dict()
    assert list(result.to_dict()) == [
        "scenario_id",
        "source",
        "total_observations_considered",
        "structurally_eligible_count",
        "truncated",
        "ranked_results",
        "candidates",
    ]
    assert isinstance(result.to_dict()["ranked_results"], list)
    assert result.to_dict()["source"]["event_id"] == "source"


def test_collector_validates_dependencies_and_request_boundary():
    with pytest.raises(CandidateCollectionError, match="matcher"):
        ReplayCandidateCollector(object(), CandidateCollectionPolicy(30, 1))
    with pytest.raises(CandidateCollectionError, match="policy"):
        ReplayCandidateCollector(RecordingMatcher(), object())
    collector = _collector()
    with pytest.raises(CandidateCollectionError, match="ScenarioResult"):
        collector.collect(object(), "source")
    for event_id in (None, 1, "", "   "):
        with pytest.raises(CandidateCollectionError, match="source_event_id"):
            collector.collect(_scenario(), event_id)


# --- replay extraction and Step 27 adaptation ------------------------------


def test_real_replay_hierarchy_is_flattened_across_sources():
    scenario = _scenario(
        _source_result("one", [
            _pipeline(_observation("source", "CAM_A", 0)),
            _pipeline(_observation("same-camera", "CAM_A", 5)),
        ]),
        _source_result("empty", []),
        _source_result("two", [
            _pipeline(_observation("candidate", "CAM_B", 10)),
        ]),
    )
    result = _collector().collect(scenario, "source")
    assert result.total_observations_considered == 3
    assert result.structurally_eligible_count == 1
    assert [item.candidate.event_id for item in result.ranked_results] == [
        "candidate"
    ]


@pytest.mark.parametrize("scenario", [
    _scenario(),
    _scenario(_source_result("empty", [])),
])
def test_empty_scenario_or_sources_report_source_not_found(scenario):
    with pytest.raises(CandidateCollectionError, match="not found"):
        _collector().collect(scenario, "source")


@pytest.mark.parametrize("event_id", [None, "", "  ", 7])
def test_invalid_observation_event_id_is_rejected(event_id):
    scenario = _scenario(_source_result("one", [
        _pipeline(_observation(event_id)),
    ]))
    with pytest.raises(CandidateCollectionError, match="event_id"):
        _collector().collect(scenario, "source")


def test_duplicate_event_ids_anywhere_are_rejected_without_deduplication():
    scenario = _scenario(
        _source_result("one", [
            _pipeline(_observation("duplicate", "CAM_A", 0)),
        ]),
        _source_result("two", [
            _pipeline(_observation("duplicate", "CAM_B", 10)),
        ]),
    )
    with pytest.raises(CandidateCollectionError, match="duplicate.*duplicate"):
        _collector().collect(scenario, "duplicate")


def test_source_event_must_exist():
    with pytest.raises(CandidateCollectionError, match="missing.*not found"):
        _collector().collect(
            _basic_scenario(_observation("candidate", "CAM_B", 10)),
            "missing",
        )


def test_adapter_preserves_plate_only_and_empty_fingerprint_semantics():
    scenario = _basic_scenario(
        _observation("review", "CAM_B", 10, status="review"),
        _observation("abstained", "CAM_C", 20, status="abstained"),
    )
    result = _collector().collect(scenario, "source")
    assert result.source.fingerprint == VehicleFingerprint(
        normalized_plate="GJ01AB1234")
    fingerprints = {
        item.candidate.event_id: item.candidate.fingerprint
        for item in result.ranked_results
    }
    assert fingerprints["review"] == VehicleFingerprint()
    assert fingerprints["abstained"] == VehicleFingerprint()
    assert all(fp.vehicle_colour is None and fp.vehicle_class is None
               for fp in fingerprints.values())


@pytest.mark.parametrize("status", ["accepted", "review", "abstained"])
def test_explicit_vehicle_fingerprint_is_preserved(status):
    fingerprint = VehicleFingerprint(
        vehicle_colour="red", vehicle_class="car")
    observation = _observation(
        "candidate", "CAM_B", 10, status=status)
    scenario = _scenario(
        _source_result("source", [
            _pipeline(_observation("source", "CAM_A", 0)),
        ]),
        _source_result("candidate", [
            _pipeline(observation, fingerprint),
        ]),
    )
    result = _collector().collect(scenario, "source")
    assert result.ranked_results[0].candidate.fingerprint is fingerprint


def test_vehicle_attributes_are_not_an_independent_adaptation_path():
    attributes = VehicleAttributes(
        vehicle_colour="red",
        colour_confidence=0.9,
        vehicle_class="car",
        class_confidence=0.9,
    )
    scenario = _scenario(
        _source_result("source", [
            _pipeline(_observation("source", "CAM_A", 0)),
        ]),
        _source_result("candidate", [
            _pipeline(
                _observation("candidate", "CAM_B", 10, status="review"),
                vehicle_attributes=attributes,
            ),
        ]),
    )
    result = _collector().collect(scenario, "source")
    assert result.ranked_results[0].candidate.fingerprint == \
        VehicleFingerprint()


def test_malformed_timestamp_fails_clearly_with_event_id():
    malformed = _observation(
        "corrupt-time", "CAM_B", "not-a-time", timestamp_marker=True)
    with pytest.raises(CandidateCollectionError, match="corrupt-time.*timestamp"):
        _collector().collect(_basic_scenario(malformed), "source")


def test_source_missing_timestamp_raises_but_candidate_missing_time_is_excluded():
    missing_source = _scenario(_source_result("one", [
        _pipeline(_observation("source", "CAM_A", None)),
    ]))
    with pytest.raises(CandidateCollectionError, match="source.*timestamp"):
        _collector().collect(missing_source, "source")

    matcher = RecordingMatcher()
    result = _collector(matcher).collect(
        _basic_scenario(_observation("no-time", "CAM_B", None)),
        "source",
    )
    assert result.structurally_eligible_count == 0
    assert matcher.calls == []


# --- structural filtering and bounded evaluation ---------------------------


def test_only_the_five_structural_rules_are_applied_before_matching():
    matcher = RecordingMatcher()
    scenario = _basic_scenario(
        _observation("earlier", "CAM_B", -1),
        _observation("equal", "CAM_B", 0),
        _observation("later", "CAM_B", 1, status="review"),
        _observation("boundary", "CAM_C", 100, status="abstained"),
        _observation("outside", "CAM_D", 100.001),
        _observation("same-camera", "CAM_A", 10),
        _observation("missing-time", "CAM_B", None),
    )
    result = _collector(matcher, window=100, limit=10).collect(
        scenario, "source")
    assert [call[1].event_id for call in matcher.calls] == [
        "later", "boundary"
    ]
    assert result.total_observations_considered == 8
    assert result.structurally_eligible_count == 2


def test_different_replay_source_with_same_observation_camera_is_excluded():
    matcher = RecordingMatcher()
    scenario = _scenario(
        _source_result("one", [
            _pipeline(_observation("source", "CAM_A", 0)),
        ], camera_id="CAM_A"),
        _source_result("two", [
            _pipeline(_observation("same-camera", "CAM_A", 10)),
            _pipeline(_observation("different-camera", "CAM_B", 20)),
        ], camera_id="CAM_Z"),
    )
    _collector(matcher).collect(scenario, "source")
    assert [call[1].event_id for call in matcher.calls] == [
        "different-camera"
    ]


def test_reverse_only_no_link_and_unknown_camera_all_reach_step27():
    matcher = _real_matcher(graph=_graph(_link("CAM_B", "CAM_A")))
    result = _collector(matcher).collect(
        _basic_scenario(
            _observation("reverse", "CAM_B", 10),
            _observation("no-link", "CAM_C", 20),
            _observation("unknown", "CAM_UNKNOWN", 30),
        ),
        "source",
    )
    assert {
        item.candidate.event_id: item.topology_status
        for item in result.ranked_results
    } == {
        "reverse": "reverse_only",
        "no-link": "no_direct_link",
        "unknown": "unknown_camera",
    }


def test_evaluation_is_capped_before_matching_and_earliest_order_is_stable():
    matcher = RecordingMatcher()
    observations = [
        _observation(f"event-{i:02d}", "CAM_B", 20 - i)
        for i in range(20)
    ]
    random.Random(28).shuffle(observations)
    result = _collector(matcher, window=30, limit=5).collect(
        _basic_scenario(*observations), "source")
    assert len(matcher.calls) == 5
    assert [call[1].event_id for call in matcher.calls] == [
        "event-19", "event-18", "event-17", "event-16", "event-15"
    ]
    assert result.structurally_eligible_count == 20
    assert result.truncated is True
    assert len(result.ranked_results) == 5
    assert len(result.candidates) == 5


def test_preselection_ties_use_camera_then_event_id_and_not_input_order():
    observations = [
        _observation("z", "CAM_B", 10),
        _observation("a", "CAM_B", 10),
        _observation("m", "CAM_C", 10),
    ]
    outputs = []
    for ordered in (observations, list(reversed(observations))):
        matcher = RecordingMatcher()
        _collector(matcher, limit=2).collect(
            _basic_scenario(*ordered), "source")
        outputs.append([call[1].event_id for call in matcher.calls])
    assert outputs == [["a", "z"], ["a", "z"]]


def test_not_truncated_and_zero_eligible_have_exact_call_counts():
    matcher = RecordingMatcher()
    result = _collector(matcher, limit=5).collect(
        _basic_scenario(_observation("candidate", "CAM_B", 10)),
        "source",
    )
    assert len(matcher.calls) == 1
    assert result.truncated is False

    empty_matcher = RecordingMatcher()
    empty = _collector(empty_matcher).collect(
        _basic_scenario(_observation("old", "CAM_B", -1)),
        "source",
    )
    assert empty_matcher.calls == []
    assert empty.ranked_results == ()
    assert empty.candidates == ()


# --- Step 27 authority and physical-feasibility regression -----------------


def test_real_step27_plate_conflict_remains_identity_conflict():
    result = _collector(_real_matcher()).collect(
        _basic_scenario(_observation(
            "conflict", "CAM_B", 100, plate="MH12CD5678")),
        "source",
    )
    match = result.ranked_results[0]
    assert match.decision == "identity_conflict"
    assert "identity_conflict" in match.reasons[-1]
    assert result.candidates == ()


def test_strong_evidence_cannot_bypass_step27_physical_gate():
    result = _collector(_real_matcher()).collect(
        _basic_scenario(_observation("too-fast", "CAM_B", 99.999)),
        "source",
    )
    match = result.ranked_results[0]
    assert match.fingerprint_comparison.outcome == "consistent"
    assert match.decision == "physically_impossible"
    assert match.travel_status == "too_fast"


@pytest.mark.parametrize(("seconds", "decision", "travel_status"), [
    (100.0, "possible_strong", "feasible"),
    (99.999, "physically_impossible", "too_fast"),
])
def test_finite_36_kph_1000_m_boundary_stays_owned_by_step27(
        seconds, decision, travel_status):
    result = _collector(_real_matcher()).collect(
        _basic_scenario(_observation("candidate", "CAM_B", seconds)),
        "source",
    ).ranked_results[0]
    assert result.minimum_travel_seconds == pytest.approx(100.0)
    assert result.decision == decision
    assert result.travel_status == travel_status


def test_matcher_errors_propagate_without_silently_dropping_candidate():
    class FailingMatcher:
        def compare(self, source, candidate):
            raise RuntimeError(f"failure for {candidate.event_id}")

    with pytest.raises(RuntimeError, match="failure for candidate"):
        _collector(FailingMatcher()).collect(
            _basic_scenario(_observation("candidate", "CAM_B", 10)),
            "source",
        )


# --- final ranking and candidate subset ------------------------------------


def test_decision_priority_and_candidate_subset_are_exact():
    decisions = [
        "ineligible",
        "physically_impossible",
        "identity_conflict",
        "insufficient_evidence",
        "possible_weak",
        "possible_strong",
    ]
    specs = {
        decision: {"decision": decision, "reason": f"original {decision}"}
        for decision in decisions
    }
    matcher = RecordingMatcher(specs)
    observations = [
        _observation(decision, "CAM_B", index + 1)
        for index, decision in enumerate(decisions)
    ]
    result = _collector(matcher).collect(
        _basic_scenario(*observations), "source")
    assert [item.decision for item in result.ranked_results] == [
        "possible_strong",
        "possible_weak",
        "insufficient_evidence",
        "identity_conflict",
        "physically_impossible",
        "ineligible",
    ]
    assert [item.decision for item in result.candidates] == [
        "possible_strong", "possible_weak"
    ]
    assert [item.reasons[0] for item in result.ranked_results] == [
        f"original {decision}" for decision in reversed(decisions)
    ]


@pytest.mark.parametrize(("left_spec", "right_spec"), [
    ({"similarity": 0.9}, {"similarity": 0.8}),
    ({"similarity": 0.8, "coverage": 0.7},
     {"similarity": 0.8, "coverage": 0.6}),
    ({"elapsed": 8}, {"elapsed": 9}),
    ({"elapsed": 8}, {"elapsed": None}),
    ({"similarity": 0.1}, {"similarity": None}),
])
def test_same_decision_numeric_tie_breakers_put_left_first(
        left_spec, right_spec):
    common = {"decision": "possible_weak"}
    matcher = RecordingMatcher({
        "left": {**common, **left_spec},
        "right": {**common, **right_spec},
    })
    result = _collector(matcher).collect(
        _basic_scenario(
            _observation("right", "CAM_B", 10),
            _observation("left", "CAM_C", 20),
        ),
        "source",
    )
    assert [item.candidate.event_id for item in result.ranked_results] == [
        "left", "right"
    ]


def test_timestamp_camera_and_event_id_complete_the_ranking_order():
    common = {
        "decision": "possible_weak",
        "similarity": 0.5,
        "coverage": 0.5,
        "elapsed": 10,
    }
    matcher = RecordingMatcher({name: common for name in ("z", "a", "early")})
    scenario = _basic_scenario(
        _observation("z", "CAM_B", 20),
        _observation("a", "CAM_B", 20),
        _observation("early", "CAM_C", 10),
    )
    result = _collector(matcher).collect(scenario, "source")
    assert [item.candidate.event_id for item in result.ranked_results] == [
        "early", "a", "z"
    ]


def test_camera_id_breaks_a_timestamp_tie_before_event_id():
    common = {
        "decision": "possible_weak", "similarity": 0.5,
        "coverage": 0.5, "elapsed": 10,
    }
    matcher = RecordingMatcher({"a-z": common, "z-a": common})
    result = _collector(matcher).collect(
        _basic_scenario(
            _observation("a-z", "CAM_C", 10),
            _observation("z-a", "CAM_B", 10),
        ),
        "source",
    )
    assert [item.candidate.event_id for item in result.ranked_results] == [
        "z-a", "a-z"
    ]


def test_missing_candidate_timestamp_is_last_in_final_ranking():
    common = {
        "decision": "possible_weak", "similarity": 0.5,
        "coverage": 0.5, "elapsed": 10,
    }
    matcher = RecordingMatcher({
        "present": {**common, "result_timestamp": T0 + timedelta(seconds=20)},
        "missing": {**common, "result_timestamp": None},
    })
    result = _collector(matcher).collect(
        _basic_scenario(
            _observation("missing", "CAM_B", 10),
            _observation("present", "CAM_C", 20),
        ),
        "source",
    )
    assert [item.candidate.event_id for item in result.ranked_results] == [
        "present", "missing"
    ]


def test_equivalent_source_and_observation_permutations_rank_identically():
    candidates = [
        _pipeline(_observation("z", "CAM_C", 10)),
        _pipeline(_observation("a", "CAM_B", 10)),
        _pipeline(_observation("middle", "CAM_D", 5)),
    ]
    source = _pipeline(_observation("source", "CAM_A", 0))
    first = _scenario(
        _source_result("one", [source, candidates[0]]),
        _source_result("two", candidates[1:]),
    )
    second = _scenario(
        _source_result("two", list(reversed(candidates[1:]))),
        _source_result("one", [candidates[0], source]),
    )
    outputs = []
    for scenario in (first, second):
        result = _collector(RecordingMatcher()).collect(scenario, "source")
        outputs.append([item.candidate.event_id
                        for item in result.ranked_results])
    assert outputs[0] == outputs[1]

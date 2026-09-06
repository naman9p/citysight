"""Step 27: pure explainable cross-camera candidate matching."""

import math
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

from phase1_anpr.observation.observation_builder import PlateObservation
from phase2_city import (
    Camera,
    CameraLink,
    CityCameraGraph,
    CrossCameraCandidateMatcher,
    CrossCameraCandidateResult,
    CrossCameraMatchPolicy,
    CrossCameraMatchValidationError,
    FingerprintObservation,
    VehicleFingerprint,
    VehicleFingerprintScorer,
)


T0 = datetime(2026, 8, 31, 10, 0, tzinfo=timezone.utc)


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
        [_camera("CAM_A"), _camera("CAM_B"), _camera("CAM_C")],
        links,
    )


def _fingerprint(plate="GJ01AB1234", colour="white", vehicle_class="car"):
    return VehicleFingerprint(
        normalized_plate=plate,
        vehicle_colour=colour,
        vehicle_class=vehicle_class,
    )


def _identity_observation(
        event_id,
        camera_id,
        timestamp,
        fingerprint=None,
        status="accepted",
):
    return FingerprintObservation(
        event_id=event_id,
        camera_id=camera_id,
        timestamp=timestamp,
        observation_status=status,
        fingerprint=fingerprint if fingerprint is not None else _fingerprint(),
    )


def _plate_observation(
        *,
        event_id="event-1",
        camera_id="CAM_A",
        timestamp="2026-08-31T10:00:00+00:00",
        status="accepted",
        plate="GJ01AB1234",
):
    abstained = status == "abstained"
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


def _matcher(
        *,
        graph=None,
        speed=36.0,
        tolerance=0.0,
        minimum_coverage=0.25,
        strong_threshold=0.70,
        scorer=None,
):
    return CrossCameraCandidateMatcher(
        graph if graph is not None else _graph(_link()),
        CrossCameraMatchPolicy(
            maximum_speed_kph=speed,
            travel_time_tolerance_seconds=tolerance,
            minimum_evidence_coverage=minimum_coverage,
            strong_similarity_threshold=strong_threshold,
        ),
        fingerprint_scorer=scorer,
    )


def _pair(
        *,
        source_fingerprint=None,
        candidate_fingerprint=None,
        source_time=T0,
        candidate_time=T0 + timedelta(seconds=100),
        source_camera="CAM_A",
        candidate_camera="CAM_B",
        source_event="event-a",
        candidate_event="event-b",
):
    source = _identity_observation(
        source_event,
        source_camera,
        source_time,
        source_fingerprint or _fingerprint(),
    )
    candidate = _identity_observation(
        candidate_event,
        candidate_camera,
        candidate_time,
        candidate_fingerprint or _fingerprint(),
    )
    return source, candidate


# --- models and validation ---------------------------------------------------


def test_fingerprint_observation_is_frozen():
    value = _pair()[0]
    with pytest.raises(FrozenInstanceError):
        value.camera_id = "CAM_C"


def test_policy_is_frozen():
    value = CrossCameraMatchPolicy(maximum_speed_kph=60)
    with pytest.raises(FrozenInstanceError):
        value.maximum_speed_kph = 80


def test_result_is_frozen():
    result = _matcher().compare(*_pair())
    assert isinstance(result, CrossCameraCandidateResult)
    with pytest.raises(FrozenInstanceError):
        result.decision = "possible_weak"


@pytest.mark.parametrize(("field", "value"), [
    ("event_id", ""),
    ("event_id", "   "),
    ("event_id", 1),
    ("camera_id", ""),
    ("camera_id", "\t"),
    ("camera_id", None),
])
def test_empty_or_invalid_identity_fields_are_rejected(field, value):
    kwargs = {
        "event_id": "event-a",
        "camera_id": "CAM_A",
        "timestamp": T0,
        "observation_status": "accepted",
        "fingerprint": _fingerprint(),
    }
    kwargs[field] = value
    with pytest.raises(CrossCameraMatchValidationError, match=field):
        FingerprintObservation(**kwargs)


@pytest.mark.parametrize("status", ["", "pending", None, 1, []])
def test_invalid_observation_status_is_rejected(status):
    with pytest.raises(CrossCameraMatchValidationError,
                       match="observation_status"):
        FingerprintObservation(
            "event-a", "CAM_A", T0, status, _fingerprint())


def test_invalid_fingerprint_type_is_rejected():
    with pytest.raises(CrossCameraMatchValidationError, match="fingerprint"):
        FingerprintObservation(
            "event-a", "CAM_A", T0, "accepted", {"plate": "x"})


def test_aware_offset_timestamp_is_normalized_to_utc():
    moment = datetime(2026, 8, 31, 15, 30,
                      tzinfo=timezone(timedelta(hours=5, minutes=30)))
    value = _identity_observation("event-a", "CAM_A", moment)
    assert value.timestamp == T0
    assert value.timestamp.tzinfo is timezone.utc


def test_naive_timestamp_is_rejected():
    with pytest.raises(CrossCameraMatchValidationError, match="timezone"):
        _identity_observation(
            "event-a", "CAM_A", datetime(2026, 8, 31, 10, 0))


def test_non_datetime_timestamp_is_rejected():
    with pytest.raises(CrossCameraMatchValidationError, match="datetime"):
        _identity_observation(
            "event-a", "CAM_A", "2026-08-31T10:00:00Z")


def test_malformed_adapter_timestamp_is_rejected():
    with pytest.raises(CrossCameraMatchValidationError,
                       match="timestamp is invalid"):
        FingerprintObservation.from_plate_observation(
            _plate_observation(timestamp="not-a-timestamp"))


@pytest.mark.parametrize(("overrides", "field"), [
    ({"maximum_speed_kph": 0}, "maximum_speed_kph"),
    ({"maximum_speed_kph": -1}, "maximum_speed_kph"),
    ({"maximum_speed_kph": float("nan")}, "maximum_speed_kph"),
    ({"maximum_speed_kph": float("inf")}, "maximum_speed_kph"),
    ({"maximum_speed_kph": "60"}, "maximum_speed_kph"),
    ({"maximum_speed_kph": True}, "maximum_speed_kph"),
    ({"travel_time_tolerance_seconds": -1},
     "travel_time_tolerance_seconds"),
    ({"travel_time_tolerance_seconds": float("nan")},
     "travel_time_tolerance_seconds"),
    ({"travel_time_tolerance_seconds": False},
     "travel_time_tolerance_seconds"),
    ({"minimum_evidence_coverage": -0.01}, "minimum_evidence_coverage"),
    ({"minimum_evidence_coverage": 1.01}, "minimum_evidence_coverage"),
    ({"minimum_evidence_coverage": True}, "minimum_evidence_coverage"),
    ({"strong_similarity_threshold": -0.01},
     "strong_similarity_threshold"),
    ({"strong_similarity_threshold": 1.01},
     "strong_similarity_threshold"),
    ({"strong_similarity_threshold": False},
     "strong_similarity_threshold"),
])
def test_invalid_policy_values_are_rejected(overrides, field):
    values = {"maximum_speed_kph": 60}
    values.update(overrides)
    with pytest.raises(CrossCameraMatchValidationError, match=field):
        CrossCameraMatchPolicy(**values)


def test_policy_boundaries_are_valid_and_normalized_to_float():
    policy = CrossCameraMatchPolicy(
        maximum_speed_kph=60,
        travel_time_tolerance_seconds=0,
        minimum_evidence_coverage=0,
        strong_similarity_threshold=1,
    )
    assert policy.to_dict() == {
        "maximum_speed_kph": 60.0,
        "travel_time_tolerance_seconds": 0.0,
        "minimum_evidence_coverage": 0.0,
        "strong_similarity_threshold": 1.0,
    }


def test_maximum_speed_is_required_and_has_no_default():
    with pytest.raises(TypeError, match="maximum_speed_kph"):
        CrossCameraMatchPolicy()


def test_subnormal_maximum_speed_that_underflows_to_zero_mps_is_rejected():
    speed = float.fromhex("0x0.0000000000001p-1022")
    assert speed > 0.0
    assert math.isfinite(speed)
    assert speed / 3.6 == 0.0

    with pytest.raises(CrossCameraMatchValidationError,
                       match="maximum_speed_kph"):
        CrossCameraMatchPolicy(maximum_speed_kph=speed)


@pytest.mark.parametrize("status", ["accepted", "review", "abstained"])
def test_all_canonical_statuses_are_supported(status):
    value = FingerprintObservation.from_plate_observation(
        _plate_observation(status=status))
    assert value.observation_status == status


# --- PlateObservation adaptation --------------------------------------------


def test_accepted_observation_fallback_includes_plate_only():
    value = FingerprintObservation.from_plate_observation(
        _plate_observation(plate=" gj-01 ab 1234 "))
    assert value.fingerprint == VehicleFingerprint(
        normalized_plate="GJ01AB1234")
    assert value.fingerprint.vehicle_colour is None
    assert value.fingerprint.vehicle_class is None


@pytest.mark.parametrize("status", ["review", "abstained"])
def test_review_and_abstained_fallback_omit_plate(status):
    value = FingerprintObservation.from_plate_observation(
        _plate_observation(status=status))
    assert value.fingerprint == VehicleFingerprint()


def test_explicit_step26_fingerprint_is_preserved_exactly():
    supplied = VehicleFingerprint(vehicle_colour="white", vehicle_class="car")
    value = FingerprintObservation.from_plate_observation(
        _plate_observation(status="review"), supplied)
    assert value.fingerprint is supplied


@pytest.mark.parametrize("status", ["review", "abstained"])
def test_review_and_abstained_attribute_fingerprints_can_match(status):
    supplied = VehicleFingerprint(vehicle_colour="white", vehicle_class="car")
    source = FingerprintObservation.from_plate_observation(
        _plate_observation(
            event_id="event-a", camera_id="CAM_A", status=status),
        supplied,
    )
    candidate = FingerprintObservation.from_plate_observation(
        _plate_observation(
            event_id="event-b",
            camera_id="CAM_B",
            timestamp="2026-08-31T10:01:40+00:00",
            status=status,
        ),
        supplied,
    )
    result = _matcher().compare(source, candidate)
    assert result.fingerprint_comparison.overall_similarity == 0.7
    assert result.decision == "possible_strong"


def test_adapter_rejects_non_observation_and_non_fingerprint():
    with pytest.raises(CrossCameraMatchValidationError, match="observation"):
        FingerprintObservation.from_plate_observation({"event_id": "e"})
    with pytest.raises(CrossCameraMatchValidationError, match="fingerprint"):
        FingerprintObservation.from_plate_observation(
            _plate_observation(), object())


def test_adapter_wraps_invalid_observation_fingerprint_evidence():
    with pytest.raises(CrossCameraMatchValidationError,
                       match="fingerprint evidence is invalid"):
        FingerprintObservation.from_plate_observation(
            _plate_observation(plate=1234))


def test_missing_adapter_timestamp_remains_missing():
    value = FingerprintObservation.from_plate_observation(
        _plate_observation(timestamp=None))
    assert value.timestamp is None


# --- fingerprint decisions ---------------------------------------------------


def test_full_match_and_feasible_direct_travel_is_possible_strong():
    result = _matcher().compare(*_pair())
    assert result.decision == "possible_strong"
    assert result.fingerprint_comparison.overall_similarity == 1.0


def test_plate_only_match_is_possible_strong():
    plate_only = VehicleFingerprint(normalized_plate="GJ01AB1234")
    result = _matcher().compare(*_pair(
        source_fingerprint=plate_only,
        candidate_fingerprint=plate_only,
    ))
    assert result.fingerprint_comparison.overall_similarity == 0.8
    assert result.fingerprint_comparison.evidence_coverage == 0.6
    assert result.decision == "possible_strong"


def test_missing_plate_with_matching_colour_and_class_is_possible_strong():
    left = _fingerprint()
    right = VehicleFingerprint(vehicle_colour="white", vehicle_class="car")
    result = _matcher().compare(*_pair(
        source_fingerprint=left,
        candidate_fingerprint=right,
    ))
    assert result.fingerprint_comparison.overall_similarity == 0.7
    assert result.fingerprint_comparison.evidence_coverage == 0.4
    assert result.decision == "possible_strong"


def test_colour_only_match_is_possible_weak():
    value = VehicleFingerprint(vehicle_colour="white")
    result = _matcher().compare(*_pair(
        source_fingerprint=value,
        candidate_fingerprint=value,
    ))
    assert result.fingerprint_comparison.overall_similarity == 0.625
    assert result.fingerprint_comparison.evidence_coverage == 0.25
    assert result.decision == "possible_weak"


def test_class_only_match_is_insufficient_by_default():
    value = VehicleFingerprint(vehicle_class="car")
    result = _matcher().compare(*_pair(
        source_fingerprint=value,
        candidate_fingerprint=value,
    ))
    assert result.fingerprint_comparison.evidence_coverage == 0.15
    assert result.decision == "insufficient_evidence"


def test_both_empty_fingerprints_are_insufficient():
    empty = VehicleFingerprint()
    result = _matcher().compare(*_pair(
        source_fingerprint=empty,
        candidate_fingerprint=empty,
    ))
    assert result.fingerprint_comparison.outcome == "insufficient_evidence"
    assert result.decision == "insufficient_evidence"


def test_disjoint_fields_are_insufficient():
    result = _matcher().compare(*_pair(
        source_fingerprint=VehicleFingerprint(
            normalized_plate="GJ01AB1234"),
        candidate_fingerprint=VehicleFingerprint(vehicle_colour="white"),
    ))
    assert result.fingerprint_comparison.overall_similarity is None
    assert result.decision == "insufficient_evidence"


def test_plate_mismatch_with_matching_attributes_is_possible_weak():
    result = _matcher().compare(*_pair(
        source_fingerprint=_fingerprint(plate="GJ01AB1234"),
        candidate_fingerprint=_fingerprint(plate="MH12CD5678"),
    ))
    comparison = result.fingerprint_comparison
    assert comparison.outcome == "mixed"
    assert comparison.overall_similarity == 0.4
    assert result.decision == "possible_weak"


def test_pure_plate_mismatch_is_identity_conflict_not_physical_failure():
    result = _matcher().compare(*_pair(
        source_fingerprint=VehicleFingerprint(
            normalized_plate="GJ01AB1234"),
        candidate_fingerprint=VehicleFingerprint(
            normalized_plate="MH12CD5678"),
    ))
    assert result.fingerprint_comparison.outcome == "contradictory"
    assert result.decision == "identity_conflict"
    assert result.travel_status == "feasible"


def test_colour_and_class_contradictions_remain_visible():
    result = _matcher().compare(*_pair(
        source_fingerprint=VehicleFingerprint(
            vehicle_colour="white", vehicle_class="car"),
        candidate_fingerprint=VehicleFingerprint(
            vehicle_colour="black", vehicle_class="truck"),
    ))
    states = {
        item.field: item.state
        for item in result.fingerprint_comparison.field_comparisons
    }
    assert states == {
        "normalized_plate": "missing",
        "vehicle_colour": "mismatch",
        "vehicle_class": "mismatch",
    }
    assert result.decision == "identity_conflict"


def test_matcher_returns_the_actual_step23_comparison():
    source, candidate = _pair(
        source_fingerprint=_fingerprint(plate=None),
        candidate_fingerprint=_fingerprint(vehicle_class="truck"),
    )
    scorer = VehicleFingerprintScorer()
    result = _matcher(scorer=scorer).compare(source, candidate)
    assert result.fingerprint_comparison == scorer.compare(
        source.fingerprint, candidate.fingerprint)


def test_repeated_match_and_serialization_are_deterministic():
    matcher = _matcher()
    source, candidate = _pair()
    first = matcher.compare(source, candidate)
    second = matcher.compare(source, candidate)
    assert first == second
    assert first.to_dict() == second.to_dict()
    assert first.reasons == second.reasons


# --- topology ---------------------------------------------------------------


def test_forward_direct_link_is_detected_and_retained():
    link = _link()
    result = _matcher(graph=_graph(link)).compare(*_pair())
    assert result.topology_status == "direct_link"
    assert result.link is link
    assert result.link.distance_m == 1000.0
    assert result.link.road_name == "Demo Road"
    assert result.link.travel_direction == "eastbound"


def test_infinite_forward_link_distance_is_rejected():
    link = _link(distance_m=float("inf"))
    with pytest.raises(CrossCameraMatchValidationError, match="distance_m"):
        _matcher(graph=_graph(link)).compare(*_pair())


def test_infinite_forward_link_distance_is_rejected_without_timestamps():
    link = _link(distance_m=float("inf"))
    with pytest.raises(CrossCameraMatchValidationError, match="distance_m"):
        _matcher(graph=_graph(link)).compare(*_pair(
            source_time=None,
            candidate_time=None,
        ))


def test_reverse_only_is_explicit_and_reverse_distance_is_not_borrowed():
    result = _matcher(graph=_graph(_link("CAM_B", "CAM_A", 9000))).compare(
        *_pair())
    assert result.topology_status == "reverse_only"
    assert result.link is None
    assert result.minimum_travel_seconds is None
    assert result.travel_status == "unknown"
    assert result.decision == "possible_strong"


def test_known_cameras_without_link_are_unverified_not_impossible():
    result = _matcher(graph=_graph()).compare(*_pair())
    assert result.topology_status == "no_direct_link"
    assert result.travel_status == "unknown"
    assert result.decision == "possible_strong"


def test_unknown_camera_keeps_physical_feasibility_unknown():
    result = _matcher().compare(*_pair(candidate_camera="CAM_UNKNOWN"))
    assert result.topology_status == "unknown_camera"
    assert result.link is None
    assert result.travel_status == "unknown"
    assert result.decision == "possible_strong"


def test_same_camera_is_ineligible():
    result = _matcher().compare(*_pair(candidate_camera="CAM_A"))
    assert result.topology_status == "same_camera"
    assert result.link is None
    assert result.travel_status == "not_applicable"
    assert result.decision == "ineligible"


def test_swapping_direction_changes_topology_without_changing_fingerprint():
    matcher = _matcher(graph=_graph(_link()))
    source, candidate = _pair()
    forward = matcher.compare(source, candidate)
    reverse = matcher.compare(candidate, source)
    assert forward.topology_status == "direct_link"
    assert reverse.topology_status == "reverse_only"
    assert forward.fingerprint_comparison == reverse.fingerprint_comparison


# --- time and travel ---------------------------------------------------------


def test_plausible_forward_direct_travel_is_feasible():
    result = _matcher().compare(*_pair(candidate_time=T0 + timedelta(seconds=120)))
    assert result.elapsed_seconds == 120.0
    assert result.minimum_travel_seconds == 100.0
    assert result.travel_status == "feasible"


def test_exact_minimum_travel_boundary_is_feasible():
    result = _matcher().compare(*_pair(candidate_time=T0 + timedelta(seconds=100)))
    assert result.elapsed_seconds == result.minimum_travel_seconds == 100.0
    assert result.travel_status == "feasible"


def test_just_below_minimum_travel_is_too_fast():
    result = _matcher().compare(*_pair(
        candidate_time=T0 + timedelta(seconds=99, microseconds=999999)))
    assert result.elapsed_seconds < result.minimum_travel_seconds
    assert result.travel_status == "too_fast"
    assert result.decision == "physically_impossible"


def test_tolerance_changes_boundary_deterministically():
    source, candidate = _pair(candidate_time=T0 + timedelta(seconds=99))
    without = _matcher(tolerance=0).compare(source, candidate)
    with_tolerance = _matcher(tolerance=1).compare(source, candidate)
    assert without.travel_status == "too_fast"
    assert with_tolerance.travel_status == "feasible"


def test_negative_elapsed_time_is_physically_impossible():
    result = _matcher().compare(*_pair(
        candidate_time=T0 - timedelta(seconds=1)))
    assert result.elapsed_seconds == -1.0
    assert result.travel_status == "invalid_order"
    assert result.decision == "physically_impossible"


def test_equal_timestamps_on_positive_distance_link_are_too_fast():
    result = _matcher().compare(*_pair(candidate_time=T0))
    assert result.elapsed_seconds == 0.0
    assert result.travel_status == "too_fast"
    assert result.decision == "physically_impossible"


@pytest.mark.parametrize(("missing_side", "source_time", "candidate_time"), [
    ("source", None, T0 + timedelta(seconds=100)),
    ("candidate", T0, None),
    ("both", None, None),
])
def test_missing_timestamp_keeps_travel_unknown(
        missing_side, source_time, candidate_time):
    result = _matcher().compare(*_pair(
        source_time=source_time,
        candidate_time=candidate_time,
    ))
    assert missing_side
    assert result.elapsed_seconds is None
    assert result.minimum_travel_seconds is None
    assert result.travel_status == "unknown"
    assert result.decision == "possible_strong"


def test_equivalent_instants_with_different_offsets_compare_as_equal():
    india = timezone(timedelta(hours=5, minutes=30))
    candidate_time = datetime(2026, 8, 31, 15, 30, tzinfo=india)
    result = _matcher(graph=_graph()).compare(*_pair(
        candidate_time=candidate_time))
    assert result.candidate.timestamp == T0
    assert result.elapsed_seconds == 0.0
    assert result.travel_status == "unknown"
    assert result.decision == "possible_strong"


def test_no_maximum_duration_rejection():
    result = _matcher().compare(*_pair(
        candidate_time=T0 + timedelta(days=365)))
    assert result.travel_status == "feasible"
    assert result.decision == "possible_strong"


def test_no_direct_link_never_runs_distance_based_too_fast_check():
    result = _matcher(graph=_graph()).compare(*_pair(
        candidate_time=T0))
    assert result.topology_status == "no_direct_link"
    assert result.elapsed_seconds == 0.0
    assert result.minimum_travel_seconds is None
    assert result.travel_status == "unknown"
    assert result.decision == "possible_strong"


# --- hard-gate precedence and collaborator behavior -------------------------


def test_same_event_is_ineligible_but_keeps_fingerprint_comparison():
    result = _matcher().compare(*_pair(candidate_event="event-a"))
    assert result.decision == "ineligible"
    assert result.fingerprint_comparison.outcome == "consistent"
    assert result.fingerprint_comparison.overall_similarity == 1.0


def test_too_fast_overrides_possible_strong_fingerprint_evidence():
    result = _matcher().compare(*_pair(candidate_time=T0))
    assert result.fingerprint_comparison.outcome == "consistent"
    assert result.fingerprint_comparison.overall_similarity == 1.0
    assert result.decision == "physically_impossible"


def test_physical_gate_overrides_empty_fingerprint_evidence():
    empty = VehicleFingerprint()
    result = _matcher().compare(*_pair(
        source_fingerprint=empty,
        candidate_fingerprint=empty,
        candidate_time=T0,
    ))
    assert result.fingerprint_comparison.outcome == "insufficient_evidence"
    assert result.decision == "physically_impossible"


def test_plate_contradiction_alone_never_means_physically_impossible():
    result = _matcher(graph=_graph()).compare(*_pair(
        source_fingerprint=VehicleFingerprint(
            normalized_plate="GJ01AB1234"),
        candidate_fingerprint=VehicleFingerprint(
            normalized_plate="MH12CD5678"),
    ))
    assert result.decision == "identity_conflict"
    assert result.travel_status == "unknown"


@pytest.mark.parametrize("topology_graph", [
    _graph(),
    _graph(_link("CAM_B", "CAM_A")),
])
def test_missing_forward_topology_never_means_physically_impossible(
        topology_graph):
    result = _matcher(graph=topology_graph).compare(*_pair())
    assert result.decision == "possible_strong"


def test_injected_scorer_errors_are_not_swallowed():
    class FailingScorer(VehicleFingerprintScorer):
        def compare(self, left, right):
            raise RuntimeError("programmer error")

    with pytest.raises(RuntimeError, match="programmer error"):
        _matcher(scorer=FailingScorer()).compare(*_pair())


def test_invalid_matcher_inputs_fail_clearly():
    policy = CrossCameraMatchPolicy(maximum_speed_kph=36)
    with pytest.raises(CrossCameraMatchValidationError, match="city_graph"):
        CrossCameraCandidateMatcher(None, policy)
    with pytest.raises(CrossCameraMatchValidationError, match="policy"):
        CrossCameraCandidateMatcher(_graph(), None)
    matcher = _matcher()
    with pytest.raises(CrossCameraMatchValidationError, match="source"):
        matcher.compare(object(), _pair()[1])
    with pytest.raises(CrossCameraMatchValidationError, match="candidate"):
        matcher.compare(_pair()[0], object())


# --- stable serialization ----------------------------------------------------


def test_to_dict_has_stable_top_level_shape_and_utc_datetimes():
    payload = _matcher().compare(*_pair()).to_dict()
    assert list(payload) == [
        "source",
        "candidate",
        "fingerprint_comparison",
        "topology_status",
        "link",
        "elapsed_seconds",
        "minimum_travel_seconds",
        "travel_status",
        "decision",
        "reasons",
    ]
    assert payload["source"]["timestamp"] == "2026-08-31T10:00:00+00:00"
    assert payload["candidate"]["timestamp"] == \
        "2026-08-31T10:01:40+00:00"
    assert payload["fingerprint_comparison"]["field_comparisons"]
    assert payload["link"] == {
        "from_camera_id": "CAM_A",
        "to_camera_id": "CAM_B",
        "distance_m": 1000.0,
        "road_name": "Demo Road",
        "travel_direction": "eastbound",
    }


def test_link_serializes_as_none_without_a_forward_link():
    payload = _matcher(graph=_graph(_link("CAM_B", "CAM_A"))).compare(
        *_pair()).to_dict()
    assert payload["topology_status"] == "reverse_only"
    assert payload["link"] is None
    assert payload["minimum_travel_seconds"] is None


def test_serialized_result_has_no_probability_or_confidence_field():
    payload = _matcher().compare(*_pair()).to_dict()
    assert "probability" not in payload
    assert "confidence" not in payload
    assert payload["decision"] == "possible_strong"

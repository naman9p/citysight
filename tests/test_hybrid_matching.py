"""Focused tests for Step 37 parallel Phase 3 hybrid matching."""

import inspect
import subprocess
import sys
from dataclasses import FrozenInstanceError, fields
from datetime import datetime, timedelta, timezone

import pytest

from phase2_city import (
    Camera,
    CameraLink,
    CityCameraGraph,
    CrossCameraCandidateMatcher,
    CrossCameraMatchPolicy,
    FingerprintObservation,
    VehicleFingerprint,
)
from phase3_city import (
    AppearanceComparisonStatus,
    HybridCandidateMatchResult,
    HybridCandidateStatus,
    Phase3HybridCandidateMatcher,
    PhysicalGateStatus,
    TrustedPlateRelation,
    VehicleAppearanceEmbedding,
)


_T0 = datetime(2026, 9, 20, 10, 0, tzinfo=timezone.utc)
_WEIGHTS_A = "a" * 64
_WEIGHTS_B = "b" * 64
_PREPROCESSING = "c" * 64


def _camera(camera_id):
    return Camera(
        camera_id=camera_id,
        name=f"Camera {camera_id}",
        latitude=28.6,
        longitude=77.2,
        road_name="Test Road",
        heading_deg=90.0,
    )


def _graph(*links):
    return CityCameraGraph(
        [_camera(name) for name in ("CAM_A", "CAM_B", "CAM_C")],
        links,
    )


def _link(distance_m=1000.0):
    return CameraLink(
        from_camera_id="CAM_A",
        to_camera_id="CAM_B",
        distance_m=distance_m,
        road_name="Test Road",
        travel_direction="eastbound",
    )


def _fingerprint(
        plate="GJ01AB1234",
        colour="white",
        vehicle_class="car",
):
    return VehicleFingerprint(
        normalized_plate=plate,
        vehicle_colour=colour,
        vehicle_class=vehicle_class,
    )


def _observation(
        event_id,
        camera_id,
        seconds,
        *,
        status="accepted",
        fingerprint=None,
):
    timestamp = None if seconds is None else _T0 + timedelta(seconds=seconds)
    return FingerprintObservation(
        event_id=event_id,
        camera_id=camera_id,
        timestamp=timestamp,
        observation_status=status,
        fingerprint=fingerprint if fingerprint is not None else _fingerprint(),
    )


def _pair(
        *,
        source_fingerprint=None,
        candidate_fingerprint=None,
        source_status="accepted",
        candidate_status="accepted",
        source_seconds=0.0,
        candidate_seconds=100.0,
        source_camera="CAM_A",
        candidate_camera="CAM_B",
        source_event="source",
        candidate_event="candidate",
):
    return (
        _observation(
            source_event,
            source_camera,
            source_seconds,
            status=source_status,
            fingerprint=source_fingerprint,
        ),
        _observation(
            candidate_event,
            candidate_camera,
            candidate_seconds,
            status=candidate_status,
            fingerprint=candidate_fingerprint,
        ),
    )


def _embedding(vector=(1.0, 0.0), *, weights_sha256=_WEIGHTS_A):
    return VehicleAppearanceEmbedding(
        vector=vector,
        dimension=len(vector),
        model_id="test-reid",
        model_version="1",
        weights_sha256=weights_sha256,
        preprocessing_sha256=_PREPROCESSING,
    )


def _matcher(*, graph=None, speed=36.0, tolerance=0.0):
    return Phase3HybridCandidateMatcher(
        graph if graph is not None else _graph(_link()),
        CrossCameraMatchPolicy(
            maximum_speed_kph=speed,
            travel_time_tolerance_seconds=tolerance,
        ),
    )


def _compare(
        *,
        source_appearance=None,
        candidate_appearance=None,
        matcher=None,
        **pair_kwargs,
):
    source, candidate = _pair(**pair_kwargs)
    result = (matcher or _matcher()).compare(
        source,
        candidate,
        source_appearance,
        candidate_appearance,
    )
    return result


def _field_states(result):
    return {
        detail.field: detail.state
        for detail in result.attribute_evidence.field_comparisons
    }


def _components(result):
    return {item.component: item for item in result.components}


def test_physical_impossibility_rejects_despite_maximum_appearance():
    appearance = _embedding()

    result = _compare(
        candidate_seconds=99.999,
        source_appearance=appearance,
        candidate_appearance=appearance,
    )

    assert result.status is HybridCandidateStatus.PHYSICALLY_IMPOSSIBLE
    assert result.physical_feasibility.gate_status is PhysicalGateStatus.IMPOSSIBLE
    assert result.appearance_comparison.cosine_similarity == pytest.approx(1.0)
    assert "speed ceiling" in result.hard_gate_reason


def test_physical_impossibility_rejects_despite_identical_trusted_plate():
    result = _compare(candidate_seconds=99.999)

    assert result.trusted_plate_evidence.relation is TrustedPlateRelation.AGREEMENT
    assert result.status is HybridCandidateStatus.PHYSICALLY_IMPOSSIBLE
    assert result.hard_gate_reason.startswith("physically_impossible:")


def test_trusted_plate_contradiction_rejects_matching_colour_and_class():
    result = _compare(
        source_fingerprint=_fingerprint(plate="GJ01AB1234"),
        candidate_fingerprint=_fingerprint(plate="MH12CD5678"),
    )

    assert result.status is HybridCandidateStatus.TRUSTED_PLATE_CONTRADICTION
    assert result.trusted_plate_evidence.relation is (
        TrustedPlateRelation.CONTRADICTION)
    assert _field_states(result) == {
        "vehicle_colour": "match",
        "vehicle_class": "match",
    }


def test_trusted_plate_contradiction_rejects_maximum_appearance():
    appearance = _embedding()
    result = _compare(
        source_fingerprint=_fingerprint(plate="GJ01AB1234"),
        candidate_fingerprint=_fingerprint(plate="MH12CD5678"),
        source_appearance=appearance,
        candidate_appearance=appearance,
    )

    assert result.appearance_comparison.cosine_similarity == pytest.approx(1.0)
    assert result.status is HybridCandidateStatus.TRUSTED_PLATE_CONTRADICTION
    assert result.hard_gate_reason == (
        "trusted_plate_contradiction: accepted normalized plates differ")


def test_trusted_plate_agreement_is_explicit_strong_component_evidence():
    result = _compare(
        source_fingerprint=_fingerprint(colour=None, vehicle_class=None),
        candidate_fingerprint=_fingerprint(colour=None, vehicle_class=None),
    )

    plate = _components(result)["trusted_plate"]
    assert result.trusted_plate_evidence.relation is TrustedPlateRelation.AGREEMENT
    assert plate.available is True
    assert plate.evidence_value == 1.0
    assert plate.hard_gate is False
    assert result.status is HybridCandidateStatus.ELIGIBLE_WITH_EVIDENCE


def test_missing_source_plate_is_neutral():
    result = _compare(
        source_fingerprint=_fingerprint(plate=None, colour=None,
                                        vehicle_class=None),
        candidate_fingerprint=_fingerprint(colour=None, vehicle_class=None),
    )

    assert result.trusted_plate_evidence.relation is (
        TrustedPlateRelation.SOURCE_UNAVAILABLE)
    plate = _components(result)["trusted_plate"]
    assert plate.available is False
    assert plate.evidence_value is None
    assert plate.hard_gate is False


def test_missing_candidate_plate_is_neutral():
    result = _compare(
        source_fingerprint=_fingerprint(colour=None, vehicle_class=None),
        candidate_fingerprint=_fingerprint(plate=None, colour=None,
                                           vehicle_class=None),
    )

    assert result.trusted_plate_evidence.relation is (
        TrustedPlateRelation.CANDIDATE_UNAVAILABLE)
    assert result.hard_gate_reason is None


@pytest.mark.parametrize("untrusted_status", ["review", "abstained"])
def test_non_trusted_plate_never_becomes_authoritative_contradiction(
        untrusted_status):
    result = _compare(
        source_status=untrusted_status,
        source_fingerprint=_fingerprint(
            plate="GJ01AB1234", colour=None, vehicle_class=None),
        candidate_fingerprint=_fingerprint(
            plate="MH12CD5678", colour=None, vehicle_class=None),
    )

    assert result.fingerprint_comparison.field_comparisons[0].state == "mismatch"
    assert result.trusted_plate_evidence.relation is (
        TrustedPlateRelation.SOURCE_UNAVAILABLE)
    assert result.status is HybridCandidateStatus.ELIGIBLE_WITHOUT_EVIDENCE
    assert result.hard_gate_reason is None


def test_matching_colour_evidence_and_exact_phase2_weight_are_preserved():
    result = _compare(
        source_fingerprint=_fingerprint(plate=None, colour="white",
                                        vehicle_class=None),
        candidate_fingerprint=_fingerprint(plate=None, colour="white",
                                           vehicle_class=None),
    )

    colour = result.attribute_evidence.field_comparisons[0]
    assert colour.field == "vehicle_colour"
    assert colour.state == "match"
    assert colour.weight == pytest.approx(0.25)
    assert colour.contribution == pytest.approx(0.25)


def test_contradictory_colour_evidence_is_preserved_without_hard_gate():
    result = _compare(
        source_fingerprint=_fingerprint(plate=None, colour="white",
                                        vehicle_class=None),
        candidate_fingerprint=_fingerprint(plate=None, colour="black",
                                           vehicle_class=None),
    )

    colour = result.attribute_evidence.field_comparisons[0]
    assert colour.state == "mismatch"
    assert colour.contribution == pytest.approx(-0.25)
    assert result.attribute_evidence.outcome == "contradictory"
    assert result.status is HybridCandidateStatus.ELIGIBLE_WITH_EVIDENCE
    assert result.hard_gate_reason is None


def test_matching_class_evidence_and_exact_phase2_weight_are_preserved():
    result = _compare(
        source_fingerprint=_fingerprint(plate=None, colour=None,
                                        vehicle_class="car"),
        candidate_fingerprint=_fingerprint(plate=None, colour=None,
                                           vehicle_class="car"),
    )

    vehicle_class = result.attribute_evidence.field_comparisons[1]
    assert vehicle_class.field == "vehicle_class"
    assert vehicle_class.state == "match"
    assert vehicle_class.weight == pytest.approx(0.15)
    assert vehicle_class.contribution == pytest.approx(0.15)


def test_contradictory_class_evidence_is_preserved_without_hard_gate():
    result = _compare(
        source_fingerprint=_fingerprint(plate=None, colour=None,
                                        vehicle_class="car"),
        candidate_fingerprint=_fingerprint(plate=None, colour=None,
                                           vehicle_class="truck"),
    )

    vehicle_class = result.attribute_evidence.field_comparisons[1]
    assert vehicle_class.state == "mismatch"
    assert vehicle_class.contribution == pytest.approx(-0.15)
    assert result.hard_gate_reason is None


def test_missing_attributes_remain_neutral():
    empty = _fingerprint(plate=None, colour=None, vehicle_class=None)
    result = _compare(
        source_fingerprint=empty,
        candidate_fingerprint=empty,
    )

    assert _field_states(result) == {
        "vehicle_colour": "missing",
        "vehicle_class": "missing",
    }
    assert result.attribute_evidence.available is False
    assert result.attribute_evidence.net_evidence == 0.0
    assert _components(result)["vehicle_attributes"].evidence_value is None
    assert result.status is HybridCandidateStatus.ELIGIBLE_WITHOUT_EVIDENCE


def test_available_compatible_appearance_contributes_continuous_evidence():
    empty = _fingerprint(plate=None, colour=None, vehicle_class=None)
    result = _compare(
        source_fingerprint=empty,
        candidate_fingerprint=empty,
        source_appearance=_embedding((1.0, 0.0)),
        candidate_appearance=_embedding((0.6, 0.8)),
    )

    appearance = _components(result)["appearance"]
    assert result.appearance_comparison.status is AppearanceComparisonStatus.AVAILABLE
    assert appearance.available is True
    assert appearance.evidence_value == pytest.approx(0.6)
    assert appearance.hard_gate is False
    assert result.status is HybridCandidateStatus.ELIGIBLE_WITH_EVIDENCE


@pytest.mark.parametrize(("source_appearance", "candidate_appearance", "status"), [
    (None, _embedding(), AppearanceComparisonStatus.SOURCE_UNAVAILABLE),
    (_embedding(), None, AppearanceComparisonStatus.CANDIDATE_UNAVAILABLE),
    (None, None, AppearanceComparisonStatus.BOTH_UNAVAILABLE),
])
def test_unavailable_appearance_is_neutral(
        source_appearance, candidate_appearance, status):
    result = _compare(
        source_appearance=source_appearance,
        candidate_appearance=candidate_appearance,
    )

    component = _components(result)["appearance"]
    assert result.appearance_comparison.status is status
    assert component.available is False
    assert component.evidence_value is None
    assert component.hard_gate is False


def test_incompatible_appearance_provenance_is_neutral():
    result = _compare(
        source_appearance=_embedding(weights_sha256=_WEIGHTS_A),
        candidate_appearance=_embedding(weights_sha256=_WEIGHTS_B),
    )

    assert result.appearance_comparison.status is (
        AppearanceComparisonStatus.INCOMPATIBLE_PROVENANCE)
    assert result.appearance_comparison.cosine_similarity is None
    assert _components(result)["appearance"].available is False
    assert result.hard_gate_reason is None


def test_result_contains_no_identity_probability_or_opaque_confidence():
    names = {field.name for field in fields(HybridCandidateMatchResult)}
    keys = set(_compare().to_dict())
    forbidden = {
        "identity_probability",
        "same_vehicle",
        "probability",
        "confidence",
        "global_vehicle_id",
        "hybrid_score",
    }

    assert names.isdisjoint(forbidden)
    assert keys.isdisjoint(forbidden)


def test_repeated_result_and_serialization_are_deterministic():
    matcher = _matcher()
    source, candidate = _pair()
    appearance = _embedding((3.0, 4.0))

    first = matcher.compare(source, candidate, appearance, appearance)
    second = matcher.compare(source, candidate, appearance, appearance)

    assert first == second
    assert first.to_dict() == second.to_dict()


def test_result_and_all_new_nested_objects_are_immutable():
    result = _compare()

    with pytest.raises(FrozenInstanceError):
        result.status = HybridCandidateStatus.INELIGIBLE
    with pytest.raises(FrozenInstanceError):
        result.physical_feasibility.travel_status = "unknown"
    with pytest.raises(FrozenInstanceError):
        result.trusted_plate_evidence.source_trusted = False
    with pytest.raises(FrozenInstanceError):
        result.attribute_evidence.outcome = "mixed"
    with pytest.raises(FrozenInstanceError):
        result.components[0].evidence_value = 0.0


def test_no_source_candidate_or_appearance_input_is_mutated():
    source, candidate = _pair()
    source_appearance = _embedding((3.0, 4.0))
    candidate_appearance = _embedding((4.0, 3.0))
    before = (
        source.to_dict(),
        candidate.to_dict(),
        source_appearance.to_dict(),
        candidate_appearance.to_dict(),
    )

    _matcher().compare(
        source, candidate, source_appearance, candidate_appearance)

    assert before == (
        source.to_dict(),
        candidate.to_dict(),
        source_appearance.to_dict(),
        candidate_appearance.to_dict(),
    )


def test_component_breakdown_is_explicit_and_not_aggregated():
    result = _compare(
        source_appearance=_embedding((1.0, 0.0)),
        candidate_appearance=_embedding((0.0, 1.0)),
    )

    assert [component.component for component in result.components] == [
        "trusted_plate",
        "vehicle_attributes",
        "appearance",
    ]
    assert [component.evidence_value for component in result.components] == [
        1.0,
        pytest.approx(0.4),
        pytest.approx(0.0),
    ]
    assert "hybrid_score" not in result.to_dict()


def test_hard_gate_reason_is_deterministic():
    first = _compare(
        source_fingerprint=_fingerprint(plate="GJ01AB1234"),
        candidate_fingerprint=_fingerprint(plate="MH12CD5678"),
    )
    second = _compare(
        source_fingerprint=_fingerprint(plate="GJ01AB1234"),
        candidate_fingerprint=_fingerprint(plate="MH12CD5678"),
    )

    assert first.hard_gate_reason == second.hard_gate_reason
    assert first.reasons == second.reasons


def test_trusted_plate_contradiction_has_priority_over_opposite_evidence():
    result = _compare(
        source_fingerprint=_fingerprint(plate="GJ01AB1234"),
        candidate_fingerprint=_fingerprint(plate="MH12CD5678"),
        source_appearance=_embedding((1.0, 0.0)),
        candidate_appearance=_embedding((1.0, 0.0)),
    )

    assert _components(result)["appearance"].evidence_value == pytest.approx(1.0)
    assert result.status is HybridCandidateStatus.TRUSTED_PLATE_CONTRADICTION


def test_physical_impossibility_has_priority_over_every_evidence_type():
    appearance = _embedding()
    result = _compare(
        candidate_seconds=-1.0,
        source_fingerprint=_fingerprint(plate="GJ01AB1234"),
        candidate_fingerprint=_fingerprint(plate="MH12CD5678"),
        source_appearance=appearance,
        candidate_appearance=appearance,
    )

    assert result.physical_feasibility.travel_status == "invalid_order"
    assert result.trusted_plate_evidence.relation is (
        TrustedPlateRelation.CONTRADICTION)
    assert result.appearance_comparison.cosine_similarity == pytest.approx(1.0)
    assert result.status is HybridCandidateStatus.PHYSICALLY_IMPOSSIBLE
    assert "precedes source" in result.hard_gate_reason


def test_unknown_topology_is_not_physical_impossibility():
    empty = _fingerprint(plate=None, colour=None, vehicle_class=None)
    result = _compare(
        matcher=_matcher(graph=_graph()),
        source_fingerprint=empty,
        candidate_fingerprint=empty,
    )

    assert result.physical_feasibility.gate_status is PhysicalGateStatus.UNKNOWN
    assert result.physical_feasibility.travel_status == "unknown"
    assert result.status is HybridCandidateStatus.ELIGIBLE_WITHOUT_EVIDENCE


def test_same_camera_remains_structurally_ineligible():
    result = _compare(candidate_camera="CAM_A")

    assert result.physical_feasibility.gate_status is PhysicalGateStatus.INELIGIBLE
    assert result.status is HybridCandidateStatus.INELIGIBLE
    assert "different cameras" in result.hard_gate_reason


def test_phase2_mixed_plate_mismatch_semantics_remain_frozen_and_distinct():
    source, candidate = _pair(
        source_fingerprint=_fingerprint(plate="GJ01AB1234"),
        candidate_fingerprint=_fingerprint(plate="MH12CD5678"),
    )
    policy = CrossCameraMatchPolicy(maximum_speed_kph=36.0)
    graph = _graph(_link())

    phase2 = CrossCameraCandidateMatcher(graph, policy).compare(
        source, candidate)
    phase3 = Phase3HybridCandidateMatcher(graph, policy).compare(
        source, candidate)

    assert phase2.fingerprint_comparison.outcome == "mixed"
    assert phase2.decision == "possible_weak"
    assert phase3.fingerprint_comparison == phase2.fingerprint_comparison
    assert phase3.status is HybridCandidateStatus.TRUSTED_PLATE_CONTRADICTION


def test_matcher_has_no_new_numeric_policy_or_threshold_surface():
    signature = inspect.signature(Phase3HybridCandidateMatcher)

    assert tuple(signature.parameters) == (
        "city_graph", "physical_policy", "fingerprint_scorer")
    assert all("threshold" not in field.name
               for field in fields(HybridCandidateMatchResult))


def test_hybrid_module_import_requires_no_torch_or_model_weights():
    command = (
        "import sys; "
        "assert 'torch' not in sys.modules; "
        "import phase3_city.hybrid_matching; "
        "assert 'torch' not in sys.modules"
    )

    completed = subprocess.run(
        [sys.executable, "-c", command],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr

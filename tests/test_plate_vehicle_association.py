"""Synthetic tests for deterministic Step 24 plate/vehicle association."""

from dataclasses import FrozenInstanceError

import pytest

from phase1_anpr.detection.detector import Detection
from phase2_city.plate_vehicle_association import (
    PlateVehicleAssociation,
    VehicleAssociationError,
    associate_plate_to_vehicle,
)
from phase2_city.vehicle_detection import VehicleDetection


def _plate(bbox=(10, 10, 20, 20), frame_number=3):
    return Detection(
        bbox=bbox, confidence=0.9, class_id=0, frame_number=frame_number)


def _vehicle(bbox=(0, 0, 30, 30), confidence=0.8, class_id=2,
             vehicle_class="car", frame_number=3):
    return VehicleDetection(
        bbox=bbox,
        confidence=confidence,
        class_id=class_id,
        vehicle_class=vehicle_class,
        frame_number=frame_number,
    )


def test_one_valid_vehicle_is_associated():
    vehicle = _vehicle()

    result = associate_plate_to_vehicle(_plate(), [vehicle])

    assert result == PlateVehicleAssociation(
        plate_bbox=(10, 10, 20, 20),
        vehicle_detection=vehicle,
        containment_ratio=1.0,
        outcome="associated",
    )


def test_no_vehicle_candidates_returns_explicit_unmatched_result():
    result = associate_plate_to_vehicle(_plate(), [])

    assert result.outcome == "unmatched"
    assert result.vehicle_detection is None
    assert result.containment_ratio is None


def test_plate_center_outside_vehicle_is_unmatched_even_with_overlap():
    result = associate_plate_to_vehicle(
        _plate(bbox=(0, 0, 10, 10)),
        [_vehicle(bbox=(0, 0, 4, 10))],
    )
    assert result.outcome == "unmatched"


def test_containment_below_minimum_is_unmatched_with_center_inside():
    result = associate_plate_to_vehicle(
        _plate(bbox=(0, 0, 10, 10)),
        [_vehicle(bbox=(3, 0, 13, 10))],
    )
    assert result.outcome == "unmatched"


def test_containment_exactly_minimum_is_accepted():
    vehicle = _vehicle(bbox=(2, 0, 12, 10))

    result = associate_plate_to_vehicle(
        _plate(bbox=(0, 0, 10, 10)), [vehicle])

    assert result.vehicle_detection is vehicle
    assert result.containment_ratio == pytest.approx(0.8)


def test_highest_containment_wins_over_smaller_vehicle():
    complete = _vehicle(bbox=(-10, -10, 30, 30), confidence=0.4)
    partial = _vehicle(bbox=(12, 5, 22, 25), confidence=0.9)

    result = associate_plate_to_vehicle(_plate(), [partial, complete])

    assert result.vehicle_detection is complete
    assert result.containment_ratio == 1.0


def test_smallest_vehicle_area_breaks_equal_containment_tie():
    large = _vehicle(bbox=(0, 0, 30, 30), confidence=0.9)
    small = _vehicle(bbox=(5, 5, 25, 25), confidence=0.4)

    result = associate_plate_to_vehicle(_plate(), [large, small])

    assert result.vehicle_detection is small


def test_highest_confidence_breaks_equal_geometry_tie():
    low = _vehicle(confidence=0.4)
    high = _vehicle(confidence=0.9)

    result = associate_plate_to_vehicle(_plate(), [low, high])

    assert result.vehicle_detection is high


def test_lexicographically_smallest_bbox_breaks_reachable_tie():
    first = _vehicle(bbox=(0, 0, 30, 30))
    second = _vehicle(bbox=(0, 1, 30, 31))

    result = associate_plate_to_vehicle(_plate(), [second, first])

    assert result.vehicle_detection is first


def test_lexicographically_smallest_class_name_breaks_reachable_tie():
    car = _vehicle(class_id=2, vehicle_class="car")
    bus = _vehicle(class_id=5, vehicle_class="bus")

    result = associate_plate_to_vehicle(_plate(), [car, bus])

    assert result.vehicle_detection is bus


def test_different_frame_candidate_is_excluded():
    result = associate_plate_to_vehicle(
        _plate(frame_number=3), [_vehicle(frame_number=4)])
    assert result.outcome == "unmatched"


def test_partially_visible_vehicle_box_can_associate():
    vehicle = _vehicle(bbox=(-20, -20, 22, 22))
    result = associate_plate_to_vehicle(_plate(), [vehicle])
    assert result.vehicle_detection is vehicle


def test_motorcycle_can_be_associated():
    motorcycle = _vehicle(class_id=3, vehicle_class="motorcycle")
    result = associate_plate_to_vehicle(_plate(), [motorcycle])
    assert result.vehicle_detection is motorcycle


def test_multiple_plates_independently_associate_to_same_vehicle():
    vehicle = _vehicle(bbox=(0, 0, 40, 40))

    first = associate_plate_to_vehicle(
        _plate(bbox=(5, 5, 10, 10)), [vehicle])
    second = associate_plate_to_vehicle(
        _plate(bbox=(25, 25, 35, 35)), [vehicle])

    assert first.vehicle_detection is vehicle
    assert second.vehicle_detection is vehicle


@pytest.mark.parametrize("bbox", [
    (0, 0, 0, 1),
    (0, 0, 1, 0),
    (2, 0, 1, 1),
    (0.0, 0, 1, 1),
    (0, 0, 1),
])
def test_invalid_target_plate_geometry_is_rejected(bbox):
    with pytest.raises(VehicleAssociationError):
        associate_plate_to_vehicle(_plate(bbox=bbox), [])


@pytest.mark.parametrize("minimum", [
    True, -0.1, 1.1, float("nan"), float("inf"), 10 ** 400,
])
def test_invalid_containment_threshold_is_rejected(minimum):
    with pytest.raises(VehicleAssociationError):
        associate_plate_to_vehicle(
            _plate(), [], min_containment_ratio=minimum)


def test_invalid_candidate_collection_and_members_are_rejected():
    with pytest.raises(VehicleAssociationError):
        associate_plate_to_vehicle(_plate(), None)
    with pytest.raises(VehicleAssociationError):
        associate_plate_to_vehicle(_plate(), [_vehicle(), object()])


def test_repeated_results_are_deterministic_and_inputs_are_not_mutated():
    candidates = [
        _vehicle(bbox=(0, 1, 30, 31)),
        _vehicle(bbox=(0, 0, 30, 30)),
    ]
    original_order = list(candidates)

    results = [associate_plate_to_vehicle(_plate(), candidates) for _ in range(5)]

    assert all(result == results[0] for result in results)
    assert candidates == original_order


def test_association_result_is_immutable_and_enforces_outcome_invariants():
    result = associate_plate_to_vehicle(_plate(), [])
    with pytest.raises(FrozenInstanceError):
        result.outcome = "associated"

    with pytest.raises(VehicleAssociationError):
        PlateVehicleAssociation(
            plate_bbox=(0, 0, 1, 1),
            vehicle_detection=_vehicle(),
            containment_ratio=None,
            outcome="associated",
        )
    with pytest.raises(VehicleAssociationError):
        PlateVehicleAssociation(
            plate_bbox=(0, 0, 1, 1),
            vehicle_detection=_vehicle(),
            containment_ratio=1.0,
            outcome="unmatched",
        )

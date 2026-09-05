"""Deterministic synthetic tests for Step 25 vehicle attributes."""

from dataclasses import FrozenInstanceError

import cv2
import numpy as np
import pytest

from phase1_anpr.observation.observation_builder import PlateObservation
from phase2_city import (
    PlateVehicleAssociation,
    VehicleAttributes,
    VehicleAttributesValidationError,
    VehicleDetection,
    VehicleFingerprint,
    VehicleFingerprintScorer,
    extract_vehicle_attributes,
    extract_vehicle_colour,
    fingerprint_from_observation_and_attributes,
    vehicle_attributes_from_association,
)


_HSV_BY_COLOUR = {
    "black": (0, 0, 40),
    "white": (0, 10, 220),
    "grey": (0, 20, 120),
    "red-low": (5, 200, 200),
    "red-high": (175, 200, 200),
    "brown": (15, 200, 100),
    "yellow": (25, 200, 200),
    "green": (60, 200, 200),
    "blue": (110, 200, 200),
    "unsupported": (150, 200, 200),
}


def _solid_hsv(hsv, shape=(100, 100)):
    image = np.empty((*shape, 3), dtype=np.uint8)
    image[:] = hsv
    return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)


def _mixed_400_pixel_roi(colour_counts):
    """Return a crop whose specified body ROI is exactly 20x20 pixels."""
    hsv = np.empty((37, 25, 3), dtype=np.uint8)
    hsv[:] = _HSV_BY_COLOUR["unsupported"]
    roi = np.empty((20, 20, 3), dtype=np.uint8)
    roi[:] = _HSV_BY_COLOUR["unsupported"]
    flattened = roi.reshape((-1, 3))
    offset = 0
    for name, count in colour_counts:
        flattened[offset:offset + count] = _HSV_BY_COLOUR[name]
        offset += count
    assert offset <= 400
    hsv[9:29, 2:22] = roi
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _detection(vehicle_class="car", confidence=0.8,
               bbox=(0, 0, 100, 100), frame_number=3):
    class_ids = {"car": 2, "motorcycle": 3, "bus": 5, "truck": 7}
    return VehicleDetection(
        bbox=bbox,
        confidence=confidence,
        class_id=class_ids[vehicle_class],
        vehicle_class=vehicle_class,
        frame_number=frame_number,
    )


def _association(detection=None, plate_bbox=(40, 55, 60, 65)):
    if detection is None:
        return PlateVehicleAssociation(
            plate_bbox=plate_bbox,
            vehicle_detection=None,
            containment_ratio=None,
            outcome="unmatched",
        )
    return PlateVehicleAssociation(
        plate_bbox=plate_bbox,
        vehicle_detection=detection,
        containment_ratio=1.0,
        outcome="associated",
    )


def _observation(status="accepted", plate="GJ01AB1234"):
    return PlateObservation(
        event_id="event-step25",
        camera_id="CAM_01",
        track_id=1,
        timestamp="2026-08-31T10:00:00+00:00",
        plate_raw=plate,
        plate_normalized=plate,
        confidence=0.9,
        status=status,
        detector_confidence=0.9,
        ocr_confidence=0.9,
        quality_score=0.8,
        best_frame_number=10,
        plate_image_path=None,
        model_version="phase1-anpr-0.1.0",
    )


def test_fully_populated_vehicle_attributes_are_valid_and_immutable():
    attributes = VehicleAttributes("white", 0.75, "car", 0.91)

    assert attributes.vehicle_colour == "white"
    assert attributes.colour_confidence == 0.75
    assert attributes.vehicle_class == "car"
    assert attributes.class_confidence == 0.91
    with pytest.raises(FrozenInstanceError):
        attributes.vehicle_colour = "black"


@pytest.mark.parametrize("attributes", [
    VehicleAttributes(vehicle_colour="blue", colour_confidence=0.8),
    VehicleAttributes(vehicle_class="bus", class_confidence=0.7),
    VehicleAttributes(),
])
def test_partial_and_absent_attribute_combinations_are_valid(attributes):
    assert isinstance(attributes, VehicleAttributes)


@pytest.mark.parametrize("kwargs", [
    {"vehicle_colour": "red"},
    {"colour_confidence": 0.8},
    {"vehicle_class": "car"},
    {"class_confidence": 0.8},
])
def test_attribute_value_and_confidence_must_appear_together(kwargs):
    with pytest.raises(VehicleAttributesValidationError):
        VehicleAttributes(**kwargs)


@pytest.mark.parametrize("field", ["colour_confidence", "class_confidence"])
@pytest.mark.parametrize("value", [
    True, "0.5", float("nan"), float("inf"), float("-inf"),
    -0.01, 1.01, 10 ** 400,
])
def test_attribute_confidences_reject_invalid_values(field, value):
    kwargs = {
        "vehicle_colour": "red",
        "colour_confidence": 0.5,
        "vehicle_class": "car",
        "class_confidence": 0.5,
    }
    kwargs[field] = value
    with pytest.raises(VehicleAttributesValidationError):
        VehicleAttributes(**kwargs)


@pytest.mark.parametrize("colour", [
    "other", "silver", "orange", "White", " white", "", 1,
])
def test_unsupported_or_noncanonical_colour_is_rejected(colour):
    with pytest.raises(VehicleAttributesValidationError):
        VehicleAttributes(vehicle_colour=colour, colour_confidence=0.8)


@pytest.mark.parametrize("vehicle_class", [
    "suv", "sedan", "van", "Car", " car", "", 2,
])
def test_unsupported_or_noncanonical_vehicle_class_is_rejected(vehicle_class):
    with pytest.raises(VehicleAttributesValidationError):
        VehicleAttributes(vehicle_class=vehicle_class, class_confidence=0.8)


@pytest.mark.parametrize("name, expected", [
    ("black", "black"),
    ("white", "white"),
    ("grey", "grey"),
    ("red-low", "red"),
    ("red-high", "red"),
    ("brown", "brown"),
    ("yellow", "yellow"),
    ("green", "green"),
    ("blue", "blue"),
])
def test_each_supported_hsv_colour_bin(name, expected):
    assert extract_vehicle_colour(_solid_hsv(_HSV_BY_COLOUR[name])) == (
        expected, 1.0)


def test_silver_like_neutral_evidence_maps_to_grey():
    crop = _solid_hsv((0, 25, 150))
    assert extract_vehicle_colour(crop) == ("grey", 1.0)


def test_unsupported_hue_abstains_without_creating_other():
    assert extract_vehicle_colour(
        _solid_hsv(_HSV_BY_COLOUR["unsupported"])) == (None, None)


@pytest.mark.parametrize("crop", [
    None,
    [],
    np.array([]),
    np.zeros((20, 20), dtype=np.uint8),
    np.zeros((20, 20, 4), dtype=np.uint8),
    np.zeros((20, 20, 3), dtype=np.float32),
])
def test_invalid_vehicle_crop_abstains(crop):
    assert extract_vehicle_colour(crop) == (None, None)


def test_invalid_or_too_small_body_roi_abstains():
    assert extract_vehicle_colour(
        np.zeros((1, 1, 3), dtype=np.uint8)) == (None, None)


def test_fewer_than_256_usable_pixels_abstains():
    crop = _solid_hsv(_HSV_BY_COLOUR["blue"], shape=(20, 20))
    assert extract_vehicle_colour(crop) == (None, None)


def test_at_least_70_percent_severe_underexposure_abstains():
    crop = _solid_hsv((0, 0, 10))
    assert extract_vehicle_colour(crop) == (None, None)


def test_at_least_70_percent_clipped_highlights_abstains():
    crop = _solid_hsv((0, 0, 255))
    assert extract_vehicle_colour(crop) == (None, None)


def test_body_roi_uses_exact_floor_converted_boundaries():
    hsv = np.empty((37, 25, 3), dtype=np.uint8)
    hsv[:] = _HSV_BY_COLOUR["red-low"]
    hsv[9:29, 2:22] = _HSV_BY_COLOUR["blue"]
    crop = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    assert extract_vehicle_colour(crop) == ("blue", 1.0)


def test_dominant_share_immediately_below_055_abstains():
    crop = _mixed_400_pixel_roi([
        ("blue", 219), ("unsupported", 181),
    ])
    assert extract_vehicle_colour(crop) == (None, None)


def test_dominant_share_exactly_055_is_accepted():
    crop = _mixed_400_pixel_roi([
        ("blue", 220), ("unsupported", 180),
    ])
    colour, confidence = extract_vehicle_colour(crop)
    assert colour == "blue"
    assert confidence == 0.55


def test_second_bin_margin_immediately_below_015_abstains():
    crop = _mixed_400_pixel_roi([
        ("blue", 220), ("green", 161), ("unsupported", 19),
    ])
    assert extract_vehicle_colour(crop) == (None, None)


def test_second_bin_margin_exactly_015_is_accepted():
    crop = _mixed_400_pixel_roi([
        ("blue", 220), ("green", 160), ("unsupported", 20),
    ])
    assert extract_vehicle_colour(crop) == ("blue", 0.55)


def test_plate_is_translated_to_crop_coordinates_and_excluded():
    detection = _detection(bbox=(100, 50, 200, 150))
    association = _association(
        detection=detection,
        plate_bbox=(120, 80, 180, 125),
    )
    hsv = np.empty((100, 100, 3), dtype=np.uint8)
    hsv[:] = _HSV_BY_COLOUR["blue"]
    hsv[30:75, 20:80] = _HSV_BY_COLOUR["red-low"]
    crop = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    assert extract_vehicle_colour(crop)[0] == "red"
    attributes = extract_vehicle_attributes(crop, association)

    assert attributes.vehicle_colour == "blue"
    assert attributes.colour_confidence == 1.0
    assert attributes.vehicle_class == "car"


def test_plate_exclusion_includes_conservative_two_pixel_margin():
    detection = _detection()
    association = _association(detection=detection)
    hsv = np.empty((100, 100, 3), dtype=np.uint8)
    hsv[:] = _HSV_BY_COLOUR["blue"]
    hsv[53:67, 38:62] = _HSV_BY_COLOUR["red-low"]
    crop = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    attributes = extract_vehicle_attributes(crop, association)

    assert attributes.vehicle_colour == "blue"
    assert attributes.colour_confidence == 1.0


def test_colour_extraction_is_deterministic_and_does_not_mutate_input():
    crop = _mixed_400_pixel_roi([
        ("green", 260), ("blue", 100), ("unsupported", 40),
    ])
    original = crop.copy()

    results = [extract_vehicle_colour(crop) for _ in range(5)]

    assert all(result == results[0] for result in results)
    assert np.array_equal(crop, original)


@pytest.mark.parametrize("vehicle_class", [
    "car", "motorcycle", "bus", "truck",
])
def test_class_attributes_preserve_supported_detection_evidence(vehicle_class):
    detection = _detection(vehicle_class=vehicle_class, confidence=0.731)

    attributes = vehicle_attributes_from_association(
        _association(detection=detection))

    assert attributes.vehicle_class == vehicle_class
    assert attributes.class_confidence == detection.confidence
    assert attributes.vehicle_colour is None


def test_unmatched_association_produces_no_class_or_colour_evidence():
    association = _association()

    assert vehicle_attributes_from_association(association) == VehicleAttributes()
    assert extract_vehicle_attributes(
        _solid_hsv(_HSV_BY_COLOUR["red-low"]), association,
    ) == VehicleAttributes()


def test_invalid_crop_abstains_for_colour_but_preserves_associated_class():
    attributes = extract_vehicle_attributes(None, _association(_detection()))
    assert attributes == VehicleAttributes(
        vehicle_class="car", class_confidence=0.8)


def test_class_extraction_rejects_non_association_input():
    with pytest.raises(VehicleAttributesValidationError):
        vehicle_attributes_from_association({"outcome": "unmatched"})


@pytest.mark.parametrize("attributes, expected", [
    (
        VehicleAttributes("white", 0.8, "car", 0.9),
        VehicleFingerprint("GJ01AB1234", "white", "car"),
    ),
    (
        VehicleAttributes(vehicle_colour="white", colour_confidence=0.8),
        VehicleFingerprint("GJ01AB1234", "white", None),
    ),
    (
        VehicleAttributes(vehicle_class="car", class_confidence=0.9),
        VehicleFingerprint("GJ01AB1234", None, "car"),
    ),
    (VehicleAttributes(), VehicleFingerprint("GJ01AB1234")),
])
def test_fingerprint_adapter_combines_accepted_plate_and_available_attributes(
        attributes, expected):
    assert fingerprint_from_observation_and_attributes(
        _observation(), attributes) == expected


@pytest.mark.parametrize("status", ["review", "abstained"])
def test_fingerprint_adapter_preserves_existing_plate_status_behavior(status):
    plate = None if status == "abstained" else "GJ01AB1234"
    result = fingerprint_from_observation_and_attributes(
        _observation(status=status, plate=plate),
        VehicleAttributes("blue", 0.8, "bus", 0.7),
    )
    assert result == VehicleFingerprint(None, "blue", "bus")


def test_fingerprint_adapter_returns_new_value_without_mutating_inputs():
    observation = _observation()
    attributes = VehicleAttributes("green", 0.8, "truck", 0.7)
    observation_before = observation.to_dict()

    result = fingerprint_from_observation_and_attributes(
        observation, attributes)

    assert result is not attributes
    assert observation.to_dict() == observation_before
    assert attributes == VehicleAttributes("green", 0.8, "truck", 0.7)


def test_fingerprint_adapter_leaves_step23_scoring_semantics_unchanged():
    left = fingerprint_from_observation_and_attributes(
        _observation(), VehicleAttributes("white", 0.8, "car", 0.9))
    right = fingerprint_from_observation_and_attributes(
        _observation(), VehicleAttributes("white", 0.6, "car", 0.7))

    result = VehicleFingerprintScorer().compare(left, right)

    assert result.overall_similarity == 1.0
    assert result.evidence_coverage == 1.0
    assert result.outcome == "consistent"


def test_malformed_attributes_cannot_reach_fingerprint_adapter():
    with pytest.raises(VehicleAttributesValidationError):
        VehicleAttributes(vehicle_colour="white")
    with pytest.raises(VehicleAttributesValidationError):
        fingerprint_from_observation_and_attributes(
            _observation(), {"vehicle_colour": "white"})

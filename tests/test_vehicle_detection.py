"""Synthetic tests for Step 24 whole-vehicle detection."""

import subprocess
import sys
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from phase2_city.vehicle_detection import (
    COCO_VEHICLE_CLASSES,
    DEFAULT_VEHICLE_CONFIDENCE_THRESHOLD,
    UltralyticsVehicleDetector,
    VehicleDetection,
    VehicleDetectionValidationError,
    VehicleDetectorError,
    crop_vehicle,
)


class _FakeBox:
    def __init__(self, xyxy, confidence, class_id):
        self.xyxy = [np.asarray(xyxy)]
        self.conf = [confidence]
        self.cls = [class_id]


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeModel:
    names = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    def __init__(self, boxes=()):
        self.results = [_FakeResult(list(boxes))]
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return self.results


def _vehicle(bbox=(1, 2, 11, 12), confidence=0.8, class_id=2,
             vehicle_class="car", frame_number=4):
    return VehicleDetection(
        bbox=bbox,
        confidence=confidence,
        class_id=class_id,
        vehicle_class=vehicle_class,
        frame_number=frame_number,
    )


def test_vehicle_detection_is_immutable_and_canonical():
    detection = _vehicle(bbox=[1, 2, 11, 12], vehicle_class="  CAR ")

    assert detection.bbox == (1, 2, 11, 12)
    assert detection.vehicle_class == "car"
    with pytest.raises(FrozenInstanceError):
        detection.confidence = 0.1


@pytest.mark.parametrize("class_id, vehicle_class", [
    (2, "car"),
    (3, "motorcycle"),
    (5, "bus"),
    (7, "truck"),
])
def test_all_supported_classes_are_valid(class_id, vehicle_class):
    detection = _vehicle(class_id=class_id, vehicle_class=vehicle_class)
    assert detection.class_id == class_id
    assert detection.vehicle_class == vehicle_class


def test_coco_vehicle_mapping_is_exact_and_read_only():
    assert dict(COCO_VEHICLE_CLASSES) == {
        2: "car", 3: "motorcycle", 5: "bus", 7: "truck",
    }
    with pytest.raises(TypeError):
        COCO_VEHICLE_CLASSES[0] = "person"


@pytest.mark.parametrize("class_id, vehicle_class", [
    (1, "bicycle"),
    (2, "truck"),
    (True, "car"),
    (2.0, "car"),
])
def test_vehicle_detection_rejects_invalid_class_identity(
        class_id, vehicle_class):
    with pytest.raises(VehicleDetectionValidationError):
        _vehicle(class_id=class_id, vehicle_class=vehicle_class)


@pytest.mark.parametrize("confidence", [
    True, False, "0.5", -0.01, 1.01, float("nan"), float("inf"),
    float("-inf"), 10 ** 400,
])
def test_confidence_validation_rejects_invalid_values(confidence):
    with pytest.raises(VehicleDetectionValidationError):
        _vehicle(confidence=confidence)


@pytest.mark.parametrize("bbox", [
    None,
    (0, 0, 1),
    (0, 0, 1, 1, 2),
    (0.0, 0, 1, 1),
    (False, 0, 1, 1),
    (0, 0, 0, 1),
    (0, 0, 1, 0),
    (2, 0, 1, 1),
])
def test_bbox_validation_rejects_invalid_geometry_or_type(bbox):
    with pytest.raises(VehicleDetectionValidationError):
        _vehicle(bbox=bbox)


@pytest.mark.parametrize("frame_number", [True, -1, 1.5, "1"])
def test_frame_number_validation_rejects_invalid_values(frame_number):
    with pytest.raises(VehicleDetectionValidationError):
        _vehicle(frame_number=frame_number)


def test_detector_filters_unsupported_and_below_threshold_at_boundary():
    model = _FakeModel([
        _FakeBox((0, 0, 10, 10), 0.90, 0),
        _FakeBox((10, 0, 20, 10), 0.349, 2),
        _FakeBox((20, 0, 30, 10), 0.35, 3),
        _FakeBox((30, 0, 40, 10), 0.80, 7),
    ])
    detector = UltralyticsVehicleDetector(model=model)

    detections = detector.detect(np.zeros((50, 50, 3)), frame_number=9)

    assert [(item.class_id, item.confidence) for item in detections] == [
        (3, 0.35), (7, 0.80),
    ]
    assert model.calls[0]["classes"] == [2, 3, 5, 7]
    assert model.calls[0]["conf"] == DEFAULT_VEHICLE_CONFIDENCE_THRESHOLD


def test_float_model_coordinates_follow_existing_integer_conversion():
    model = _FakeModel([_FakeBox((1.9, 2.1, 8.9, 9.7), 0.8, 2)])
    detector = UltralyticsVehicleDetector(model=model)

    [detection] = detector.detect(np.zeros((12, 12, 3)), frame_number=0)

    assert detection.bbox == (1, 2, 8, 9)


def test_results_have_deterministic_bbox_class_confidence_order():
    model = _FakeModel([
        _FakeBox((20, 0, 30, 10), 0.9, 7),
        _FakeBox((0, 0, 10, 10), 0.6, 5),
        _FakeBox((0, 0, 10, 10), 0.8, 2),
        _FakeBox((0, 0, 10, 10), 0.9, 2),
    ])
    detector = UltralyticsVehicleDetector(model=model)

    detections = detector.detect(np.zeros((40, 40, 3)), frame_number=0)

    assert [(d.bbox, d.class_id, d.confidence) for d in detections] == [
        ((0, 0, 10, 10), 2, 0.9),
        ((0, 0, 10, 10), 2, 0.8),
        ((0, 0, 10, 10), 5, 0.6),
        ((20, 0, 30, 10), 7, 0.9),
    ]


def test_injected_model_factory_is_lazy_and_reused(tmp_path):
    weights = tmp_path / "yolov8n.pt"
    weights.write_bytes(b"test placeholder")
    model = _FakeModel([_FakeBox((0, 0, 5, 5), 0.8, 2)])
    constructed = []

    def factory(path):
        constructed.append(path)
        return model

    detector = UltralyticsVehicleDetector(
        weights_path=weights, model_factory=factory)
    assert constructed == []

    detector.detect(np.zeros((5, 5, 3)), 0)
    detector.detect(np.zeros((5, 5, 3)), 1)

    assert constructed == [str(weights)]


def test_missing_local_weights_fail_before_factory_is_called(tmp_path):
    constructed = []
    detector = UltralyticsVehicleDetector(
        weights_path=tmp_path / "missing.pt",
        model_factory=lambda path: constructed.append(path),
    )

    with pytest.raises(VehicleDetectorError, match="weights not found"):
        detector.detect(np.zeros((5, 5, 3)), 0)
    assert constructed == []


def test_incompatible_taxonomy_fails_clearly():
    model = _FakeModel()
    model.names = {2: "sedan", 3: "motorcycle", 5: "bus", 7: "truck"}
    detector = UltralyticsVehicleDetector(model=model)

    with pytest.raises(VehicleDetectorError, match="COCO-compatible"):
        detector.detect(np.zeros((5, 5, 3)), 0)
    assert model.calls == []


def test_module_import_does_not_import_ultralytics_or_initialize_a_model():
    command = (
        "import sys; "
        "sys.modules['ultralytics'] = None; "
        "import phase2_city.vehicle_detection; "
        "assert sys.modules['ultralytics'] is None"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_crop_inside_frame_returns_independent_copy_without_mutation():
    frame = np.arange(8 * 10 * 3, dtype=np.uint8).reshape((8, 10, 3))
    before = frame.copy()

    crop = crop_vehicle(frame, _vehicle(bbox=(2, 1, 7, 6)))

    assert np.array_equal(crop, before[1:6, 2:7])
    crop[:] = 0
    assert np.array_equal(frame, before)


def test_crop_clamps_partially_outside_box_without_negative_wrapping():
    frame = np.arange(6 * 8, dtype=np.uint8).reshape((6, 8))

    crop = crop_vehicle(frame, _vehicle(bbox=(-3, -2, 4, 3)))

    assert np.array_equal(crop, frame[0:3, 0:4])


def test_crop_completely_outside_frame_returns_none():
    frame = np.zeros((6, 8, 3), dtype=np.uint8)
    assert crop_vehicle(frame, _vehicle(bbox=(-5, -4, -1, -1))) is None


@pytest.mark.parametrize("frame", [None, [], np.array([]), np.zeros((2,))])
def test_crop_rejects_invalid_images(frame):
    with pytest.raises(VehicleDetectionValidationError):
        crop_vehicle(frame, _vehicle())

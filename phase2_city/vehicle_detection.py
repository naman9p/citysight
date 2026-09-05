"""Optional whole-vehicle detection foundation for Step 24.

The existing Phase 1 detector identifies license plates only.  This module
provides a separate, lazy Ultralytics wrapper for coarse COCO vehicle classes.
Weights must already exist locally; importing this module never imports YOLO,
loads a model, or downloads weights.

Bounding boxes follow the project's existing detector convention: integer
``(x1, y1, x2, y2)`` coordinates in original-frame pixels.  Ultralytics float
coordinates are converted with ``int()`` while parsing, exactly as the current
plate detector does.  Clamping to image bounds happens only in ``crop_vehicle``.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType

import numpy as np


COCO_VEHICLE_CLASSES = MappingProxyType({
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
})

DEFAULT_VEHICLE_WEIGHTS_PATH = Path("weights/yolov8n.pt")
DEFAULT_VEHICLE_CONFIDENCE_THRESHOLD = 0.35
DEFAULT_VEHICLE_IOU_THRESHOLD = 0.45
DEFAULT_VEHICLE_IMAGE_SIZE = 640


class VehicleDetectorError(Exception):
    """Raised when vehicle-detector configuration or loading fails."""


class VehicleDetectionValidationError(ValueError):
    """Raised when a vehicle detection or detector input is invalid."""


def canonicalize_bbox(bbox) -> tuple:
    """Validate and return an immutable integer ``(x1, y1, x2, y2)`` box."""
    if isinstance(bbox, (str, bytes)) or not isinstance(bbox, Sequence):
        raise VehicleDetectionValidationError(
            "bbox must be a sequence of four integer coordinates")
    if len(bbox) != 4:
        raise VehicleDetectionValidationError(
            "bbox must contain exactly four coordinates")

    coordinates = []
    for value in bbox:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise VehicleDetectionValidationError(
                "bbox coordinates must be finite integers")
        coordinates.append(int(value))

    x1, y1, x2, y2 = coordinates
    if x2 <= x1:
        raise VehicleDetectionValidationError("bbox must satisfy x2 > x1")
    if y2 <= y1:
        raise VehicleDetectionValidationError("bbox must satisfy y2 > y1")
    return tuple(coordinates)


def validate_frame_number(frame_number) -> int:
    """Return a canonical non-negative integer frame number."""
    if isinstance(frame_number, bool) or not isinstance(frame_number, Integral):
        raise VehicleDetectionValidationError(
            "frame_number must be a non-negative integer")
    value = int(frame_number)
    if value < 0:
        raise VehicleDetectionValidationError(
            "frame_number must be a non-negative integer")
    return value


def validate_confidence(value, name="confidence") -> float:
    """Return a finite, non-Boolean confidence in the inclusive range 0..1."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise VehicleDetectionValidationError(
            f"{name} must be a finite number between 0 and 1")
    try:
        numeric = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise VehicleDetectionValidationError(
            f"{name} must be a finite number between 0 and 1") from exc
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise VehicleDetectionValidationError(
            f"{name} must be a finite number between 0 and 1")
    return numeric


@dataclass(frozen=True)
class VehicleDetection:
    """One supported whole-vehicle detection in original-frame coordinates."""

    bbox: tuple
    confidence: float
    class_id: int
    vehicle_class: str
    frame_number: int

    def __post_init__(self):
        bbox = canonicalize_bbox(self.bbox)
        confidence = validate_confidence(self.confidence)
        if isinstance(self.class_id, bool) or not isinstance(
                self.class_id, Integral):
            raise VehicleDetectionValidationError(
                "class_id must be a supported integer COCO class ID")
        class_id = int(self.class_id)
        expected_class = COCO_VEHICLE_CLASSES.get(class_id)
        if expected_class is None:
            raise VehicleDetectionValidationError(
                f"unsupported COCO vehicle class ID: {class_id}")
        if not isinstance(self.vehicle_class, str):
            raise VehicleDetectionValidationError(
                "vehicle_class must match the supported COCO class name")
        vehicle_class = " ".join(self.vehicle_class.split()).lower()
        if vehicle_class != expected_class:
            raise VehicleDetectionValidationError(
                f"class ID {class_id} must map to {expected_class!r}, "
                f"got {self.vehicle_class!r}")
        frame_number = validate_frame_number(self.frame_number)

        object.__setattr__(self, "bbox", bbox)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "class_id", class_id)
        object.__setattr__(self, "vehicle_class", vehicle_class)
        object.__setattr__(self, "frame_number", frame_number)


def _resolve_device(device) -> str:
    requested = str(device or "cpu").strip().lower()
    if requested in ("cuda", "gpu"):
        return "cuda"
    if requested == "cpu":
        return "cpu"
    if requested != "auto":
        raise VehicleDetectorError(
            "device must be one of: cpu, cuda, gpu, auto")
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _taxonomy_name(names, class_id):
    if isinstance(names, Mapping):
        if class_id in names:
            return names[class_id]
        return names.get(str(class_id))
    if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        return names[class_id] if class_id < len(names) else None
    return None


def _validate_taxonomy(model) -> None:
    names = getattr(model, "names", None)
    for class_id, expected in COCO_VEHICLE_CLASSES.items():
        actual = _taxonomy_name(names, class_id)
        actual_name = str(actual).strip().lower() if actual is not None else None
        if actual_name != expected:
            raise VehicleDetectorError(
                "vehicle model taxonomy is not COCO-compatible: "
                f"class ID {class_id} must be {expected!r}, got {actual!r}")


def _raw_number(value, name) -> float:
    try:
        scalar = value[0]
    except (IndexError, TypeError, KeyError) as exc:
        raise VehicleDetectionValidationError(
            f"raw {name} must contain one numeric value") from exc
    if isinstance(scalar, (bool, np.bool_)):
        raise VehicleDetectionValidationError(
            f"raw {name} must be a finite number")
    try:
        numeric = float(scalar)
    except (OverflowError, TypeError, ValueError) as exc:
        raise VehicleDetectionValidationError(
            f"raw {name} must be a finite number") from exc
    if not math.isfinite(numeric):
        raise VehicleDetectionValidationError(
            f"raw {name} must be a finite number")
    return numeric


class UltralyticsVehicleDetector:
    """Lazy local YOLOv8 vehicle detector restricted to supported COCO classes."""

    def __init__(self, weights_path=DEFAULT_VEHICLE_WEIGHTS_PATH, *,
                 device="cpu",
                 confidence_threshold=DEFAULT_VEHICLE_CONFIDENCE_THRESHOLD,
                 iou_threshold=DEFAULT_VEHICLE_IOU_THRESHOLD,
                 image_size=DEFAULT_VEHICLE_IMAGE_SIZE,
                 model=None, model_factory=None):
        if model is not None and model_factory is not None:
            raise VehicleDetectorError(
                "provide either model or model_factory, not both")
        try:
            self.weights_path = Path(weights_path)
        except TypeError as exc:
            raise VehicleDetectorError("weights_path must be path-like") from exc
        self.device = _resolve_device(device)
        self.confidence_threshold = validate_confidence(
            confidence_threshold, "confidence_threshold")
        self.iou_threshold = validate_confidence(
            iou_threshold, "iou_threshold")
        if isinstance(image_size, bool) or not isinstance(image_size, Integral):
            raise VehicleDetectionValidationError(
                "image_size must be a positive integer")
        self.image_size = int(image_size)
        if self.image_size <= 0:
            raise VehicleDetectionValidationError(
                "image_size must be a positive integer")

        self._model = model
        self._model_factory = model_factory
        self._taxonomy_validated = False

    def _get_model(self):
        if self._model is None:
            if not self.weights_path.is_file():
                raise VehicleDetectorError(
                    f"Vehicle detector weights not found: {self.weights_path}. "
                    "Provide a local COCO YOLOv8 nano weights file; automatic "
                    "download is disabled.")
            factory = self._model_factory
            if factory is None:
                try:
                    from ultralytics import YOLO
                except ImportError as exc:  # pragma: no cover - dependency guard
                    raise VehicleDetectorError(
                        "ultralytics is not installed; run pip install -r "
                        "requirements.txt") from exc
                factory = YOLO
            try:
                self._model = factory(str(self.weights_path))
            except Exception as exc:
                raise VehicleDetectorError(
                    f"Failed to load vehicle YOLO model: {exc}") from exc

        if not self._taxonomy_validated:
            _validate_taxonomy(self._model)
            self._taxonomy_validated = True
        return self._model

    def detect(self, frame, frame_number):
        """Run one frame and return supported detections in deterministic order."""
        frame_number = validate_frame_number(frame_number)
        model = self._get_model()
        results = model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            device=self.device,
            classes=list(COCO_VEHICLE_CLASSES),
            verbose=False,
        )
        return self._parse_results(results, frame_number)

    def _parse_results(self, results, frame_number):
        frame_number = validate_frame_number(frame_number)
        detections = []
        for result in results or []:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                raw_class_id = _raw_number(getattr(box, "cls", None), "class_id")
                if not raw_class_id.is_integer():
                    raise VehicleDetectionValidationError(
                        "raw class_id must be an integer")
                class_id = int(raw_class_id)
                vehicle_class = COCO_VEHICLE_CLASSES.get(class_id)
                if vehicle_class is None:
                    continue

                confidence = validate_confidence(
                    _raw_number(getattr(box, "conf", None), "confidence"))
                if confidence < self.confidence_threshold:
                    continue

                try:
                    raw_bbox = box.xyxy[0]
                    if len(raw_bbox) != 4:
                        raise ValueError
                    float_bbox = tuple(float(value) for value in raw_bbox)
                except (AttributeError, IndexError, TypeError, ValueError) as exc:
                    raise VehicleDetectionValidationError(
                        "raw bbox must contain four finite coordinates") from exc
                if not all(math.isfinite(value) for value in float_bbox):
                    raise VehicleDetectionValidationError(
                        "raw bbox must contain four finite coordinates")
                # Preserve the established PlateDetector conversion convention.
                bbox = tuple(int(value) for value in float_bbox)
                detections.append(VehicleDetection(
                    bbox=bbox,
                    confidence=confidence,
                    class_id=class_id,
                    vehicle_class=vehicle_class,
                    frame_number=frame_number,
                ))

        detections.sort(key=lambda detection: (
            detection.bbox,
            detection.class_id,
            -detection.confidence,
        ))
        return detections


def crop_vehicle(frame, detection: VehicleDetection):
    """Return a clamped copy of a vehicle crop, or ``None`` when off-frame.

    Vehicle boxes are already integers.  This boundary only clamps them to the
    NumPy image dimensions, preventing negative-index wrapping.  The source
    image is never modified and returned crops never alias its storage.
    """
    if not isinstance(detection, VehicleDetection):
        raise VehicleDetectionValidationError(
            "detection must be a VehicleDetection")
    if not isinstance(frame, np.ndarray) or frame.ndim not in (2, 3) \
            or frame.size == 0:
        raise VehicleDetectionValidationError(
            "frame must be a non-empty 2D or 3D NumPy image")

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = detection.bbox
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


__all__ = [
    "COCO_VEHICLE_CLASSES",
    "DEFAULT_VEHICLE_WEIGHTS_PATH",
    "DEFAULT_VEHICLE_CONFIDENCE_THRESHOLD",
    "VehicleDetectorError",
    "VehicleDetectionValidationError",
    "VehicleDetection",
    "UltralyticsVehicleDetector",
    "crop_vehicle",
]

"""Deterministic in-memory vehicle attribute extraction (Step 25).

Colour evidence is derived only from an already-associated vehicle crop.  The
inner body ROI uses floor-converted half-open pixel bounds: horizontal 10%-90%
and vertical 25%-80%.  The associated plate box is translated from frame to
crop coordinates and excluded with a 10% (minimum two-pixel) margin.

The HSV result is a deterministic heuristic, not a calibrated probability.
No detector, association, persistence, or fingerprint scoring occurs here.
"""

import math
from dataclasses import dataclass
from numbers import Real
from typing import Optional, Tuple

import cv2
import numpy as np

from phase1_anpr.observation.observation_builder import PlateObservation
from phase2_city.fingerprint import VehicleFingerprint
from phase2_city.plate_vehicle_association import PlateVehicleAssociation
from phase2_city.vehicle_detection import canonicalize_bbox


SUPPORTED_VEHICLE_COLOURS = frozenset({
    "black",
    "white",
    "grey",
    "red",
    "blue",
    "green",
    "yellow",
    "brown",
})
SUPPORTED_VEHICLE_CLASSES = frozenset({
    "car",
    "motorcycle",
    "bus",
    "truck",
})

MIN_USABLE_COLOUR_PIXELS = 256
PLATE_EXCLUSION_PADDING_RATIO = 0.10
MIN_PLATE_EXCLUSION_PADDING = 2


class VehicleAttributesValidationError(ValueError):
    """Raised when vehicle attribute data or extraction input is malformed."""


def _validate_confidence(value, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise VehicleAttributesValidationError(
            f"{field_name} must be a finite number between 0 and 1")
    try:
        numeric = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise VehicleAttributesValidationError(
            f"{field_name} must be a finite number between 0 and 1") from exc
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise VehicleAttributesValidationError(
            f"{field_name} must be a finite number between 0 and 1")
    return numeric


def _validate_attribute_pair(value, confidence, *, value_name,
                             confidence_name, supported_values):
    if value is None:
        if confidence is not None:
            raise VehicleAttributesValidationError(
                f"{confidence_name} must be None when {value_name} is missing")
        return None, None
    if confidence is None:
        raise VehicleAttributesValidationError(
            f"{confidence_name} is required when {value_name} is present")
    if not isinstance(value, str) or value not in supported_values:
        allowed = ", ".join(sorted(supported_values))
        raise VehicleAttributesValidationError(
            f"{value_name} must be one of: {allowed}")
    return value, _validate_confidence(confidence, confidence_name)


@dataclass(frozen=True)
class VehicleAttributes:
    """Validated optional colour and coarse COCO class evidence."""

    vehicle_colour: Optional[str] = None
    colour_confidence: Optional[float] = None
    vehicle_class: Optional[str] = None
    class_confidence: Optional[float] = None

    def __post_init__(self):
        colour, colour_confidence = _validate_attribute_pair(
            self.vehicle_colour,
            self.colour_confidence,
            value_name="vehicle_colour",
            confidence_name="colour_confidence",
            supported_values=SUPPORTED_VEHICLE_COLOURS,
        )
        vehicle_class, class_confidence = _validate_attribute_pair(
            self.vehicle_class,
            self.class_confidence,
            value_name="vehicle_class",
            confidence_name="class_confidence",
            supported_values=SUPPORTED_VEHICLE_CLASSES,
        )
        object.__setattr__(self, "vehicle_colour", colour)
        object.__setattr__(self, "colour_confidence", colour_confidence)
        object.__setattr__(self, "vehicle_class", vehicle_class)
        object.__setattr__(self, "class_confidence", class_confidence)


def _body_roi_bounds(width: int, height: int) -> tuple:
    """Return the specified half-open body ROI with floor pixel conversion."""
    return (
        int(width * 0.10),
        int(height * 0.25),
        int(width * 0.90),
        int(height * 0.80),
    )


def _usable_body_pixels(vehicle_crop, plate_bbox):
    if not isinstance(vehicle_crop, np.ndarray) \
            or vehicle_crop.dtype != np.uint8 \
            or vehicle_crop.ndim != 3 \
            or vehicle_crop.shape[2] != 3 \
            or vehicle_crop.size == 0:
        return None

    height, width = vehicle_crop.shape[:2]
    roi_x1, roi_y1, roi_x2, roi_y2 = _body_roi_bounds(width, height)
    if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
        return None

    roi = vehicle_crop[roi_y1:roi_y2, roi_x1:roi_x2]
    usable_mask = np.ones(roi.shape[:2], dtype=bool)

    if plate_bbox is not None:
        try:
            plate_x1, plate_y1, plate_x2, plate_y2 = canonicalize_bbox(
                plate_bbox)
        except ValueError as exc:
            raise VehicleAttributesValidationError(
                "plate_bbox must be a valid integer bounding box") from exc

        padding_x = max(
            MIN_PLATE_EXCLUSION_PADDING,
            math.ceil((plate_x2 - plate_x1)
                      * PLATE_EXCLUSION_PADDING_RATIO),
        )
        padding_y = max(
            MIN_PLATE_EXCLUSION_PADDING,
            math.ceil((plate_y2 - plate_y1)
                      * PLATE_EXCLUSION_PADDING_RATIO),
        )
        excluded_x1 = max(roi_x1, plate_x1 - padding_x)
        excluded_y1 = max(roi_y1, plate_y1 - padding_y)
        excluded_x2 = min(roi_x2, plate_x2 + padding_x)
        excluded_y2 = min(roi_y2, plate_y2 + padding_y)
        if excluded_x2 > excluded_x1 and excluded_y2 > excluded_y1:
            usable_mask[
                excluded_y1 - roi_y1:excluded_y2 - roi_y1,
                excluded_x1 - roi_x1:excluded_x2 - roi_x1,
            ] = False

    return roi[usable_mask]


def _colour_counts(hsv_pixels) -> dict:
    hue = hsv_pixels[:, 0]
    saturation = hsv_pixels[:, 1]
    value = hsv_pixels[:, 2]
    assigned = np.zeros(len(hsv_pixels), dtype=bool)
    counts = {}

    def assign(name, condition):
        selected = condition & ~assigned
        counts[name] = int(np.count_nonzero(selected))
        assigned[selected] = True

    # Precedence resolves the intentional V=50..55 overlap in the given bins.
    assign("black", (value >= 20) & (value <= 55))
    assign("white", (saturation <= 35) & (value >= 180) & (value <= 250))
    assign("grey", (saturation <= 45) & (value >= 56) & (value < 180))
    assign(
        "red",
        ((hue <= 9) | (hue >= 170))
        & (saturation >= 50) & (value >= 50),
    )
    assign(
        "brown",
        (hue >= 10) & (hue <= 19)
        & (saturation >= 50) & (value >= 50) & (value < 170),
    )
    assign(
        "yellow",
        (hue >= 20) & (hue <= 35)
        & (saturation >= 50) & (value >= 50),
    )
    assign(
        "green",
        (hue >= 36) & (hue <= 85)
        & (saturation >= 50) & (value >= 50),
    )
    assign(
        "blue",
        (hue >= 86) & (hue <= 135)
        & (saturation >= 50) & (value >= 50),
    )
    return counts


def extract_vehicle_colour(
        vehicle_crop,
        plate_bbox=None,
) -> Tuple[Optional[str], Optional[float]]:
    """Return dominant supported colour evidence from one vehicle crop.

    ``plate_bbox`` is optional crop-relative geometry. Invalid image evidence
    abstains. Malformed supplied geometry raises a validation error.
    """
    usable_bgr = _usable_body_pixels(vehicle_crop, plate_bbox)
    if usable_bgr is None or len(usable_bgr) < MIN_USABLE_COLOUR_PIXELS:
        return None, None

    hsv_pixels = cv2.cvtColor(
        usable_bgr.reshape((-1, 1, 3)), cv2.COLOR_BGR2HSV,
    ).reshape((-1, 3))
    values = hsv_pixels[:, 2]
    usable_count = len(hsv_pixels)

    underexposed_count = int(np.count_nonzero(values < 20))
    if underexposed_count * 100 >= usable_count * 70:
        return None, None
    clipped_count = int(np.count_nonzero(values > 250))
    if clipped_count * 100 >= usable_count * 70:
        return None, None

    counts = _colour_counts(hsv_pixels)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    dominant_colour, dominant_count = ranked[0]
    second_count = ranked[1][1]

    # Integer cross-products make both stated boundaries exact and repeatable.
    if dominant_count * 100 < usable_count * 55:
        return None, None
    if (dominant_count - second_count) * 100 < usable_count * 15:
        return None, None
    return dominant_colour, dominant_count / usable_count


def vehicle_attributes_from_association(
        association: PlateVehicleAssociation) -> VehicleAttributes:
    """Convert the existing Step 24 association into class evidence only."""
    if not isinstance(association, PlateVehicleAssociation):
        raise VehicleAttributesValidationError(
            "association must be a PlateVehicleAssociation")
    detection = association.vehicle_detection
    if detection is None:
        return VehicleAttributes()
    return VehicleAttributes(
        vehicle_class=detection.vehicle_class,
        class_confidence=detection.confidence,
    )


def extract_vehicle_attributes(
        vehicle_crop,
        association: PlateVehicleAssociation,
) -> VehicleAttributes:
    """Derive colour and class from an existing association and its crop.

    The crop must be the output corresponding to Step 24 ``crop_vehicle``. Its
    origin is therefore the vehicle box clamped at frame coordinate zero.
    Invalid crop evidence abstains for colour while preserving valid class
    evidence from the association.
    """
    class_attributes = vehicle_attributes_from_association(association)
    detection = association.vehicle_detection
    if detection is None:
        return class_attributes

    crop_origin_x = max(0, detection.bbox[0])
    crop_origin_y = max(0, detection.bbox[1])
    plate_x1, plate_y1, plate_x2, plate_y2 = association.plate_bbox
    plate_bbox_in_crop = (
        plate_x1 - crop_origin_x,
        plate_y1 - crop_origin_y,
        plate_x2 - crop_origin_x,
        plate_y2 - crop_origin_y,
    )
    colour, colour_confidence = extract_vehicle_colour(
        vehicle_crop, plate_bbox=plate_bbox_in_crop)
    return VehicleAttributes(
        vehicle_colour=colour,
        colour_confidence=colour_confidence,
        vehicle_class=class_attributes.vehicle_class,
        class_confidence=class_attributes.class_confidence,
    )


def fingerprint_from_observation_and_attributes(
        observation: PlateObservation,
        attributes: VehicleAttributes,
) -> VehicleFingerprint:
    """Return a new fingerprint combining Step 23 plate and Step 25 evidence."""
    if not isinstance(attributes, VehicleAttributes):
        raise VehicleAttributesValidationError(
            "attributes must be a VehicleAttributes")
    plate_fingerprint = VehicleFingerprint.from_plate_observation(observation)
    return VehicleFingerprint(
        normalized_plate=plate_fingerprint.normalized_plate,
        vehicle_colour=attributes.vehicle_colour,
        vehicle_class=attributes.vehicle_class,
    )


__all__ = [
    "SUPPORTED_VEHICLE_COLOURS",
    "SUPPORTED_VEHICLE_CLASSES",
    "VehicleAttributesValidationError",
    "VehicleAttributes",
    "extract_vehicle_colour",
    "vehicle_attributes_from_association",
    "extract_vehicle_attributes",
    "fingerprint_from_observation_and_attributes",
]

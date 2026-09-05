"""Pure deterministic plate-to-vehicle geometry association (Step 24)."""

import math
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real
from typing import Optional

from phase2_city.vehicle_detection import (
    VehicleDetection,
    canonicalize_bbox,
    validate_frame_number,
)


DEFAULT_MIN_PLATE_CONTAINMENT = 0.80


class VehicleAssociationError(ValueError):
    """Raised when plate association input is invalid."""


def _validate_ratio(value, name) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise VehicleAssociationError(
            f"{name} must be a finite number between 0 and 1")
    try:
        numeric = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise VehicleAssociationError(
            f"{name} must be a finite number between 0 and 1") from exc
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise VehicleAssociationError(
            f"{name} must be a finite number between 0 and 1")
    return numeric


@dataclass(frozen=True)
class PlateVehicleAssociation:
    """Auditable association result with no synthetic combined confidence."""

    plate_bbox: tuple
    vehicle_detection: Optional[VehicleDetection]
    containment_ratio: Optional[float]
    outcome: str

    def __post_init__(self):
        try:
            plate_bbox = canonicalize_bbox(self.plate_bbox)
        except ValueError as exc:
            raise VehicleAssociationError(str(exc)) from exc
        object.__setattr__(self, "plate_bbox", plate_bbox)

        if self.outcome == "associated":
            if not isinstance(self.vehicle_detection, VehicleDetection):
                raise VehicleAssociationError(
                    "associated result requires a VehicleDetection")
            if self.containment_ratio is None:
                raise VehicleAssociationError(
                    "associated result requires containment_ratio")
            ratio = _validate_ratio(
                self.containment_ratio, "containment_ratio")
            object.__setattr__(self, "containment_ratio", ratio)
        elif self.outcome == "unmatched":
            if self.vehicle_detection is not None \
                    or self.containment_ratio is not None:
                raise VehicleAssociationError(
                    "unmatched result cannot contain vehicle evidence")
        else:
            raise VehicleAssociationError(
                "outcome must be 'associated' or 'unmatched'")


def _plate_input(plate_detection):
    try:
        bbox = canonicalize_bbox(plate_detection.bbox)
        frame_number = validate_frame_number(plate_detection.frame_number)
    except (AttributeError, ValueError) as exc:
        raise VehicleAssociationError(
            "plate detection must provide a valid bbox and frame_number") from exc
    return bbox, frame_number


def _intersection_area(left, right) -> int:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def associate_plate_to_vehicle(
        plate_detection,
        vehicle_detections: Iterable[VehicleDetection],
        min_containment_ratio=DEFAULT_MIN_PLATE_CONTAINMENT,
) -> PlateVehicleAssociation:
    """Associate one plate independently using frame identity and geometry.

    Detection confidence is deliberately not filtered here.  It is used only
    as the approved deterministic tie-break among candidates already emitted by
    the vehicle-detector boundary.
    """
    plate_bbox, plate_frame = _plate_input(plate_detection)
    minimum = _validate_ratio(
        min_containment_ratio, "min_containment_ratio")
    if isinstance(vehicle_detections, (str, bytes)) \
            or not isinstance(vehicle_detections, Iterable):
        raise VehicleAssociationError(
            "vehicle_detections must be an iterable of VehicleDetection")

    plate_area = ((plate_bbox[2] - plate_bbox[0])
                  * (plate_bbox[3] - plate_bbox[1]))
    center_x = (plate_bbox[0] + plate_bbox[2]) / 2.0
    center_y = (plate_bbox[1] + plate_bbox[3]) / 2.0

    qualified = []
    for vehicle in vehicle_detections:
        if not isinstance(vehicle, VehicleDetection):
            raise VehicleAssociationError(
                "vehicle_detections must contain only VehicleDetection values")
        if vehicle.frame_number != plate_frame:
            continue

        vx1, vy1, vx2, vy2 = vehicle.bbox
        if not (vx1 <= center_x < vx2 and vy1 <= center_y < vy2):
            continue
        containment = _intersection_area(plate_bbox, vehicle.bbox) / plate_area
        if containment < minimum:
            continue
        vehicle_area = (vx2 - vx1) * (vy2 - vy1)
        qualified.append((vehicle, containment, vehicle_area))

    if not qualified:
        return PlateVehicleAssociation(
            plate_bbox=plate_bbox,
            vehicle_detection=None,
            containment_ratio=None,
            outcome="unmatched",
        )

    qualified.sort(key=lambda item: (
        -item[1],
        item[2],
        -item[0].confidence,
        item[0].bbox,
        item[0].vehicle_class,
        item[0].class_id,
    ))
    vehicle, containment, _ = qualified[0]
    return PlateVehicleAssociation(
        plate_bbox=plate_bbox,
        vehicle_detection=vehicle,
        containment_ratio=containment,
        outcome="associated",
    )


__all__ = [
    "DEFAULT_MIN_PLATE_CONTAINMENT",
    "VehicleAssociationError",
    "PlateVehicleAssociation",
    "associate_plate_to_vehicle",
]

"""Phase 2 domain models: Camera and directed CameraLink (Step 17).

Validation is plain Python (no new framework) via dataclass ``__post_init__``,
matching the lightweight style of the Phase 1 code. Both models are frozen so a
stored/looked-up value cannot be mutated in place behind the repository's back.
"""

from dataclasses import dataclass
from typing import Optional


class CameraValidationError(ValueError):
    """Raised when a Camera or CameraLink fails validation."""


@dataclass(frozen=True)
class Camera:
    """A physical city camera at a fixed location and heading."""

    camera_id: str
    name: str
    latitude: float
    longitude: float
    road_name: str
    heading_deg: float
    zone: Optional[str] = None
    enabled: bool = True

    def __post_init__(self):
        if not self.camera_id or not str(self.camera_id).strip():
            raise CameraValidationError("camera_id must not be empty")
        if not self.name or not str(self.name).strip():
            raise CameraValidationError("name must not be empty")
        if not self.road_name or not str(self.road_name).strip():
            raise CameraValidationError("road_name must not be empty")
        if not (-90.0 <= float(self.latitude) <= 90.0):
            raise CameraValidationError(
                f"latitude must be between -90 and 90, got {self.latitude}"
            )
        if not (-180.0 <= float(self.longitude) <= 180.0):
            raise CameraValidationError(
                f"longitude must be between -180 and 180, got {self.longitude}"
            )
        if not (0.0 <= float(self.heading_deg) < 360.0):
            raise CameraValidationError(
                f"heading_deg must satisfy 0 <= heading_deg < 360, got {self.heading_deg}"
            )


@dataclass(frozen=True)
class CameraLink:
    """A directed road connection from one camera to another.

    Identity is the ordered pair ``(from_camera_id, to_camera_id)``; A -> B and
    B -> A are distinct links.
    """

    from_camera_id: str
    to_camera_id: str
    distance_m: float
    road_name: str
    travel_direction: Optional[str] = None

    def __post_init__(self):
        if not self.from_camera_id or not str(self.from_camera_id).strip():
            raise CameraValidationError("from_camera_id must not be empty")
        if not self.to_camera_id or not str(self.to_camera_id).strip():
            raise CameraValidationError("to_camera_id must not be empty")
        if self.from_camera_id == self.to_camera_id:
            raise CameraValidationError(
                "from_camera_id cannot equal to_camera_id (self-link)"
            )
        if not self.road_name or not str(self.road_name).strip():
            raise CameraValidationError("road_name must not be empty")
        if not (float(self.distance_m) > 0):
            raise CameraValidationError(
                f"distance_m must be > 0, got {self.distance_m}"
            )

    @property
    def key(self):
        """Composite directed identity."""
        return (self.from_camera_id, self.to_camera_id)

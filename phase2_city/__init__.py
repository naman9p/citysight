"""Phase 2: city camera registry and topology graph (Step 17)."""

from phase2_city.models import (
    Camera,
    CameraLink,
    CameraValidationError,
)
from phase2_city.repository import (
    CityRepository,
    SQLiteCityRepository,
    CameraNotFoundError,
    InvalidCameraLinkError,
    CameraInUseError,
)
from phase2_city.graph import CityCameraGraph
from phase2_city.scenario import (
    Scenario,
    ScenarioSource,
    ScenarioResult,
    ScenarioError,
    load_scenario,
)
from phase2_city.trajectory import (
    Trajectory,
    TrajectorySighting,
    TrajectoryTransition,
    TrajectoryReconstructor,
    TrajectoryError,
    TrajectoryQueryError,
    TrajectoryDataError,
)
from phase2_city.fingerprint import (
    DEFAULT_FINGERPRINT_WEIGHTS,
    FingerprintValidationError,
    VehicleFingerprint,
    FingerprintFieldComparison,
    FingerprintComparison,
    VehicleFingerprintScorer,
)
from phase2_city.vehicle_detection import (
    COCO_VEHICLE_CLASSES,
    DEFAULT_VEHICLE_WEIGHTS_PATH,
    DEFAULT_VEHICLE_CONFIDENCE_THRESHOLD,
    VehicleDetectorError,
    VehicleDetectionValidationError,
    VehicleDetection,
    UltralyticsVehicleDetector,
    crop_vehicle,
)
from phase2_city.plate_vehicle_association import (
    DEFAULT_MIN_PLATE_CONTAINMENT,
    VehicleAssociationError,
    PlateVehicleAssociation,
    associate_plate_to_vehicle,
)
from phase2_city.vehicle_attributes import (
    SUPPORTED_VEHICLE_COLOURS,
    SUPPORTED_VEHICLE_CLASSES,
    VehicleAttributesValidationError,
    VehicleAttributes,
    extract_vehicle_colour,
    vehicle_attributes_from_association,
    extract_vehicle_attributes,
    fingerprint_from_observation_and_attributes,
)
from phase2_city.vehicle_enrichment import TrackVehicleEnricher
from phase2_city.cross_camera_matching import (
    CrossCameraMatchValidationError,
    FingerprintObservation,
    CrossCameraMatchPolicy,
    CrossCameraCandidateResult,
    CrossCameraCandidateMatcher,
)

__all__ = [
    "Camera",
    "CameraLink",
    "CameraValidationError",
    "CityRepository",
    "SQLiteCityRepository",
    "CameraNotFoundError",
    "InvalidCameraLinkError",
    "CameraInUseError",
    "CityCameraGraph",
    "Scenario",
    "ScenarioSource",
    "ScenarioResult",
    "ScenarioError",
    "load_scenario",
    "Trajectory",
    "TrajectorySighting",
    "TrajectoryTransition",
    "TrajectoryReconstructor",
    "TrajectoryError",
    "TrajectoryQueryError",
    "TrajectoryDataError",
    "DEFAULT_FINGERPRINT_WEIGHTS",
    "FingerprintValidationError",
    "VehicleFingerprint",
    "FingerprintFieldComparison",
    "FingerprintComparison",
    "VehicleFingerprintScorer",
    "COCO_VEHICLE_CLASSES",
    "DEFAULT_VEHICLE_WEIGHTS_PATH",
    "DEFAULT_VEHICLE_CONFIDENCE_THRESHOLD",
    "VehicleDetectorError",
    "VehicleDetectionValidationError",
    "VehicleDetection",
    "UltralyticsVehicleDetector",
    "crop_vehicle",
    "DEFAULT_MIN_PLATE_CONTAINMENT",
    "VehicleAssociationError",
    "PlateVehicleAssociation",
    "associate_plate_to_vehicle",
    "SUPPORTED_VEHICLE_COLOURS",
    "SUPPORTED_VEHICLE_CLASSES",
    "VehicleAttributesValidationError",
    "VehicleAttributes",
    "extract_vehicle_colour",
    "vehicle_attributes_from_association",
    "extract_vehicle_attributes",
    "fingerprint_from_observation_and_attributes",
    "TrackVehicleEnricher",
    "CrossCameraMatchValidationError",
    "FingerprintObservation",
    "CrossCameraMatchPolicy",
    "CrossCameraCandidateResult",
    "CrossCameraCandidateMatcher",
]

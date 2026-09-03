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
]

"""Phase 2 multi-camera replay scenario loading + fail-fast validation (Step 19).

A "scenario" is an explicit, curated multi-camera recorded-video replay for the
Phase 2 city view. Unlike the Phase 1 single-video demo (which may derive a
source_id from a filename), a scenario requires every field to be stated
explicitly and validates the WHOLE scenario before any inference runs:

  * scenario_id present and non-empty
  * at least one source
  * each source: source_id (non-empty, non-whitespace, unique across the
    scenario), camera_id, video_path, start_time
  * camera_id must exist in the Step 17 city configuration
  * start_time must be a timezone-aware ISO-8601 datetime (reuses
    parse_video_start_time from Step 18 — no duplicated datetime parsing)
  * video_path resolves relative to the scenario YAML file and must exist

All of the above is checked in load_scenario, so if any source is invalid no
source is processed. This layer sits on top of the existing phase1_anpr replay
orchestration; it does not re-implement detection/OCR/tracking.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import yaml

from phase1_anpr.pipeline.anpr_pipeline import parse_video_start_time


class ScenarioError(ValueError):
    """Raised when a replay scenario is malformed or fails preflight validation."""


@dataclass(frozen=True)
class ScenarioSource:
    """One explicitly specified recorded-video source in a scenario."""

    source_id: str
    camera_id: str
    video_path: str        # absolute, resolved relative to the scenario file
    start_time: datetime   # timezone-aware UTC (validated at load time)


@dataclass(frozen=True)
class Scenario:
    """A fully validated multi-camera replay scenario."""

    scenario_id: str
    sources: List[ScenarioSource]


@dataclass
class ScenarioResult:
    """Scenario-level outcome: ordered per-source results + aggregate totals."""

    scenario_id: str
    source_results: list  # list[phase1_anpr.replay.SourceReplayResult]

    @property
    def totals(self) -> dict:
        """Sum the per-source status tallies across the whole scenario."""
        totals = {"observations": 0, "accepted": 0, "review": 0,
                  "abstained": 0, "alerts": 0}
        for sr in self.source_results:
            for key, value in sr.counts.items():
                totals[key] += value
        return totals


def _resolve_video_path(base_dir: Path, raw_video, source_id: str) -> Path:
    """Resolve a source's video_path relative to the scenario file directory.

    Relative paths are anchored at ``base_dir`` (the scenario YAML's folder);
    absolute paths are used as-is. pathlib keeps this platform-independent.
    """
    if not isinstance(raw_video, str) or not raw_video.strip():
        raise ScenarioError(f"source '{source_id}': video_path is required")
    path = Path(raw_video)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _validate_start_time(source_id: str, value) -> datetime:
    """Require a present, timezone-aware start_time (reuses Step 18 parsing).

    ``parse_video_start_time`` returns None for missing/blank values, so those
    are rejected explicitly here — a scenario source must always carry a time.
    """
    if value is None:
        raise ScenarioError(f"source '{source_id}': start_time is required")
    if isinstance(value, str) and not value.strip():
        raise ScenarioError(f"source '{source_id}': start_time must not be empty")
    try:
        parsed = parse_video_start_time(value)
    except ValueError as exc:
        raise ScenarioError(f"source '{source_id}': {exc}") from exc
    if parsed is None:  # defensive: blank/None already rejected above
        raise ScenarioError(f"source '{source_id}': start_time is required")
    return parsed


def load_scenario(scenario_path, city_graph) -> Scenario:
    """Load and fully validate a replay scenario against the city topology.

    ``city_graph`` is a Step 17 ``CityCameraGraph``; every source camera_id must
    resolve in it. Raises ``FileNotFoundError`` if the scenario file is missing
    and ``ScenarioError`` on any structural/referential/preflight problem. On
    success every source is guaranteed replayable, so processing can begin
    without risk of failing partway through.
    """
    path = Path(scenario_path)
    if not path.exists():
        raise FileNotFoundError(f"Scenario file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ScenarioError(
            "scenario must be a mapping with 'scenario_id' and 'sources'")

    scenario_id = data.get("scenario_id")
    if not isinstance(scenario_id, str) or not scenario_id.strip():
        raise ScenarioError("scenario_id is required and must be a non-empty string")

    raw_sources = data.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ScenarioError("scenario must define at least one source")

    base_dir = path.resolve().parent
    sources: List[ScenarioSource] = []
    seen_ids = set()
    for i, entry in enumerate(raw_sources):
        if not isinstance(entry, dict):
            raise ScenarioError(f"sources[{i}] must be a mapping")

        source_id = entry.get("source_id")
        if not isinstance(source_id, str) or not source_id.strip():
            raise ScenarioError(
                f"sources[{i}]: source_id is required and must be a non-empty, "
                "non-whitespace string")
        if source_id in seen_ids:
            raise ScenarioError(f"duplicate source_id: {source_id!r}")
        seen_ids.add(source_id)

        camera_id = entry.get("camera_id")
        if not isinstance(camera_id, str) or not camera_id.strip():
            raise ScenarioError(f"source '{source_id}': camera_id is required")
        if city_graph.get_camera(camera_id) is None:
            raise ScenarioError(
                f"source '{source_id}': camera_id {camera_id!r} is not defined "
                "in the city configuration")

        video_path = _resolve_video_path(base_dir, entry.get("video_path"), source_id)
        if not video_path.exists():
            raise ScenarioError(
                f"source '{source_id}': video file not found: {video_path}")

        start_time = _validate_start_time(source_id, entry.get("start_time"))

        sources.append(ScenarioSource(
            source_id=source_id, camera_id=camera_id,
            video_path=str(video_path), start_time=start_time))

    return Scenario(scenario_id=scenario_id, sources=sources)

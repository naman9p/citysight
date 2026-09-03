"""Phase 2 cross-camera trajectory reconstruction engine (Step 20).

Given a user-entered plate, reconstruct its OBSERVED trajectory across the city:
an ordered, duplicate-collapsed sequence of camera sightings (enriched with camera
GIS metadata from the Step 17 topology) and the transitions between consecutive
sightings, each carrying travel time and direct-link metadata.

Deliberately narrow:

  * read-only; no persistence of trajectories
  * ``accepted`` observations only (review/abstained excluded)
  * NO anomaly detection: no speed estimation, no implausibility flags, no
    classification, no shortest-path or inferred intermediate cameras. It only
    reports what was observed and whether a direct topology link exists.

Reuses existing building blocks rather than re-implementing them:

  * ``PlateNormalizer`` (Step 9) for plate normalization + validity
  * ``parse_video_start_time`` (Step 18) for tz-aware ISO-8601 parsing of both
    query bounds and stored observation timestamps
  * ``CityCameraGraph`` (Step 17) for camera metadata + directed links

CLI::

    python -m phase2_city.trajectory --plate "MH12AB1234" \\
        --start 2026-08-31T04:00:00Z --end 2026-08-31T06:00:00Z [--json]

Observations are read from the SQLite path in the Phase 1 ``config.yaml``
(``--db`` overrides it), so the CLI traces whatever the pipeline/replay wrote.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from phase1_anpr.normalization.plate_normalizer import PlateNormalizer
from phase1_anpr.persistence import SQLiteObservationRepository
from phase1_anpr.pipeline.anpr_pipeline import parse_video_start_time
from phase1_anpr.utils.config import DEFAULT_CONFIG_PATH, load_config
from phase2_city.config.loader import DEFAULT_CITY_CONFIG_PATH, load_city_config
from phase2_city.graph import CityCameraGraph
from phase2_city.models import CameraLink


DEFAULT_DUPLICATE_WINDOW_SECONDS = 30.0

# Safety cap on how many observation rows one query reads back. The repository
# returns newest-first, so hitting the cap drops the OLDEST sightings — the
# reconstructed Trajectory is flagged ``truncated`` when that happens.
DEFAULT_MAX_OBSERVATIONS = 500

# Only high-confidence, accepted observations contribute to an automatic
# trajectory. Operator-assisted inclusion of review observations is a later step.
_ACCEPTED_STATUS = "accepted"


class TrajectoryError(ValueError):
    """Base class for trajectory reconstruction errors."""


class TrajectoryQueryError(TrajectoryError):
    """Raised on invalid caller input (bad plate, bad/naive time bounds)."""


class TrajectoryDataError(TrajectoryError):
    """Raised when a persisted observation carries an unusable timestamp."""


@dataclass(frozen=True)
class TrajectorySighting:
    """One retained accepted observation, enriched with camera GIS metadata.

    Camera metadata comes from the ``CityCameraGraph``. If the observation
    references a camera that is no longer in the graph (historical / removed),
    ``camera_id`` is preserved and every metadata field is ``None``.
    """

    event_id: str
    plate_normalized: str
    camera_id: str
    timestamp: datetime          # tz-aware UTC
    confidence: float
    evidence_ref: Optional[str]
    camera_name: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    road_name: Optional[str]
    heading_deg: Optional[float]

    @property
    def camera_known(self) -> bool:
        """True iff this sighting's camera is present in the topology graph."""
        return self.camera_name is not None

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "plate_normalized": self.plate_normalized,
            "camera_id": self.camera_id,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
            "evidence_ref": self.evidence_ref,
            "camera_name": self.camera_name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "road_name": self.road_name,
            "heading_deg": self.heading_deg,
        }


@dataclass(frozen=True)
class TrajectoryTransition:
    """The observed movement between two consecutive retained sightings.

    ``link`` is the directed Step 17 topology link ``from -> to`` when one
    exists; ``None`` means those cameras are not directly connected, i.e. the
    vehicle crossed unobserved territory. No intermediate camera or route is
    inferred, and deliberately no speed is derived from distance/time — Step 20
    reports what was observed, nothing more.
    """

    from_event_id: str
    to_event_id: str
    from_camera_id: str
    to_camera_id: str
    departed_at: datetime        # tz-aware UTC
    arrived_at: datetime         # tz-aware UTC
    travel_time_seconds: float
    link: Optional[CameraLink] = None

    @property
    def direct_link(self) -> bool:
        """True iff the topology has a direct ``from -> to`` link."""
        return self.link is not None

    @property
    def distance_m(self) -> Optional[float]:
        return self.link.distance_m if self.link is not None else None

    @property
    def road_name(self) -> Optional[str]:
        return self.link.road_name if self.link is not None else None

    @property
    def travel_direction(self) -> Optional[str]:
        return self.link.travel_direction if self.link is not None else None

    def to_dict(self) -> dict:
        return {
            "from_event_id": self.from_event_id,
            "to_event_id": self.to_event_id,
            "from_camera_id": self.from_camera_id,
            "to_camera_id": self.to_camera_id,
            "departed_at": self.departed_at.isoformat(),
            "arrived_at": self.arrived_at.isoformat(),
            "travel_time_seconds": self.travel_time_seconds,
            "direct_link": self.direct_link,
            "distance_m": self.distance_m,
            "road_name": self.road_name,
            "travel_direction": self.travel_direction,
        }


@dataclass(frozen=True)
class Trajectory:
    """One plate's observed path: retained sightings + their transitions.

    ``sightings`` is chronological (oldest first) and duplicate-collapsed;
    ``transitions`` always has ``len(sightings) - 1`` entries (empty for 0 or 1
    sighting). ``accepted_count`` is how many accepted observations were read
    before collapsing, so the collapse is auditable. ``truncated`` is True when
    the reconstructor's observation cap was reached, meaning older sightings may
    be missing and the query window should be narrowed.
    """

    plate_normalized: str
    sightings: List[TrajectorySighting]
    transitions: List[TrajectoryTransition]
    accepted_count: int = 0
    truncated: bool = False

    @property
    def sighting_count(self) -> int:
        return len(self.sightings)

    @property
    def collapsed_duplicate_count(self) -> int:
        """Accepted observations dropped as duplicates of a retained sighting."""
        return max(0, self.accepted_count - len(self.sightings))

    @property
    def camera_sequence(self) -> list:
        """Camera ids in observed order (consecutive repeats are real revisits)."""
        return [s.camera_id for s in self.sightings]

    @property
    def first_seen(self) -> Optional[datetime]:
        return self.sightings[0].timestamp if self.sightings else None

    @property
    def last_seen(self) -> Optional[datetime]:
        return self.sightings[-1].timestamp if self.sightings else None

    @property
    def total_duration_seconds(self) -> Optional[float]:
        """Span from first to last sighting; None when it cannot span (< 2)."""
        if len(self.sightings) < 2:
            return None
        return (self.last_seen - self.first_seen).total_seconds()

    @property
    def unknown_camera_ids(self) -> list:
        """Sighted camera ids absent from the topology graph, in order."""
        seen = []
        for s in self.sightings:
            if not s.camera_known and s.camera_id not in seen:
                seen.append(s.camera_id)
        return seen

    def to_dict(self) -> dict:
        first, last = self.first_seen, self.last_seen
        return {
            "plate_normalized": self.plate_normalized,
            "sighting_count": self.sighting_count,
            "accepted_observation_count": self.accepted_count,
            "collapsed_duplicate_count": self.collapsed_duplicate_count,
            "truncated": self.truncated,
            "first_seen": first.isoformat() if first is not None else None,
            "last_seen": last.isoformat() if last is not None else None,
            "total_duration_seconds": self.total_duration_seconds,
            "camera_sequence": self.camera_sequence,
            "unknown_camera_ids": self.unknown_camera_ids,
            "sightings": [s.to_dict() for s in self.sightings],
            "transitions": [t.to_dict() for t in self.transitions],
        }


class TrajectoryReconstructor:
    """Rebuilds a plate's observed trajectory from persisted observations.

    Read-only: it queries the Phase 1 ``ObservationRepository`` and the Step 17
    ``CityCameraGraph`` and returns a ``Trajectory``; nothing is written back.
    """

    def __init__(self, observation_repo, city_graph, normalizer=None,
                 duplicate_window_seconds=DEFAULT_DUPLICATE_WINDOW_SECONDS,
                 max_observations=DEFAULT_MAX_OBSERVATIONS):
        window = float(duplicate_window_seconds)
        if window < 0:
            raise ValueError("duplicate_window_seconds must be >= 0, got "
                             f"{duplicate_window_seconds!r}")
        limit = int(max_observations)
        if limit < 1:
            raise ValueError(
                f"max_observations must be >= 1, got {max_observations!r}")
        self.observation_repo = observation_repo
        self.city_graph = city_graph
        self.normalizer = normalizer or PlateNormalizer()
        self.duplicate_window_seconds = window
        self.max_observations = limit

    @classmethod
    def from_config(cls, observation_repo, city_graph, config, normalizer=None):
        """Build from ``config['trajectory']`` (falls back to module defaults)."""
        config = config or {}
        cfg = config.get("trajectory") or {}
        return cls(
            observation_repo, city_graph,
            normalizer=normalizer or PlateNormalizer.from_config(config),
            duplicate_window_seconds=cfg.get(
                "duplicate_window_seconds", DEFAULT_DUPLICATE_WINDOW_SECONDS),
            max_observations=cfg.get(
                "max_observations", DEFAULT_MAX_OBSERVATIONS),
        )

    # --- query --------------------------------------------------------------

    def reconstruct(self, plate, *, start=None, end=None) -> Trajectory:
        """Reconstruct the observed trajectory for ``plate``.

        ``plate`` is free-form operator input (case/spacing/hyphens are
        normalized by the Step 9 normalizer). ``start``/``end`` are optional
        inclusive ISO-8601 (or ``datetime``) bounds that MUST carry timezone
        information. Raises ``TrajectoryQueryError`` on bad input and
        ``TrajectoryDataError`` if a stored observation timestamp is unusable.
        """
        normalized = self._normalize_query_plate(plate)
        start_dt = self._parse_bound("start", start)
        end_dt = self._parse_bound("end", end)
        if start_dt is not None and end_dt is not None and start_dt > end_dt:
            raise TrajectoryQueryError(
                f"start ({start_dt.isoformat()}) must not be after end "
                f"({end_dt.isoformat()})")

        rows = self.observation_repo.list_by_plate(
            normalized, self.max_observations, start=start_dt, end=end_dt)
        truncated = len(rows) >= self.max_observations

        sightings = [self._to_sighting(r) for r in rows
                     if r.get("status") == _ACCEPTED_STATUS]
        # The SQL bounds compare stored ISO strings (canonical observations are
        # UTC), so re-check the parsed instants: a row written with a non-UTC
        # offset must not slip into the window.
        sightings = [s for s in sightings
                     if self._within_bounds(s.timestamp, start_dt, end_dt)]
        sightings.sort(key=lambda s: (s.timestamp, s.event_id))

        retained = self._collapse_duplicates(sightings)
        return Trajectory(
            plate_normalized=normalized,
            sightings=retained,
            transitions=self._build_transitions(retained),
            accepted_count=len(sightings),
            truncated=truncated,
        )

    # --- input validation ---------------------------------------------------

    def _normalize_query_plate(self, plate) -> str:
        """Normalize + validate operator plate input (Step 9 rules, no guessing)."""
        if plate is None:
            raise TrajectoryQueryError("plate is required")
        result = self.normalizer.normalize(str(plate))
        if not result.normalized_text:
            raise TrajectoryQueryError(
                f"plate is empty after normalization: {plate!r}")
        if not result.is_valid:
            raise TrajectoryQueryError(
                f"{result.normalized_text!r} is not a valid Indian plate format")
        return result.normalized_text

    @staticmethod
    def _parse_bound(name: str, value):
        """Parse an optional inclusive tz-aware time bound (reuses Step 18)."""
        try:
            return parse_video_start_time(value)
        except ValueError as exc:
            raise TrajectoryQueryError(f"{name}: {exc}") from exc

    @staticmethod
    def _within_bounds(moment, start, end) -> bool:
        if start is not None and moment < start:
            return False
        if end is not None and moment > end:
            return False
        return True

    # --- row -> sighting ----------------------------------------------------

    def _to_sighting(self, row) -> TrajectorySighting:
        """Map a stored observation row to a camera-enriched sighting."""
        event_id = row.get("event_id")
        camera_id = row.get("camera_id")
        timestamp = self._parse_observation_timestamp(event_id, row.get("timestamp"))
        camera = self.city_graph.get_camera(camera_id) if camera_id else None
        return TrajectorySighting(
            event_id=event_id,
            plate_normalized=row.get("plate_normalized"),
            camera_id=camera_id,
            timestamp=timestamp,
            confidence=row.get("confidence"),
            evidence_ref=row.get("evidence_ref"),
            camera_name=camera.name if camera is not None else None,
            latitude=camera.latitude if camera is not None else None,
            longitude=camera.longitude if camera is not None else None,
            road_name=camera.road_name if camera is not None else None,
            heading_deg=camera.heading_deg if camera is not None else None,
        )

    @staticmethod
    def _parse_observation_timestamp(event_id, value) -> datetime:
        """Parse a persisted timestamp; corrupt/naive values are a data error."""
        try:
            parsed = parse_video_start_time(value)
        except ValueError as exc:
            raise TrajectoryDataError(
                f"observation {event_id!r} has an unusable timestamp "
                f"{value!r}: {exc}") from exc
        if parsed is None:
            raise TrajectoryDataError(
                f"observation {event_id!r} has no timestamp")
        return parsed

    # --- assembly -----------------------------------------------------------

    def _collapse_duplicates(self, sightings) -> List[TrajectorySighting]:
        """Collapse a burst of same-camera sightings into one camera visit.

        A sighting is dropped when it is at the SAME camera as the previous
        sighting of the current visit and within ``duplicate_window_seconds`` of
        it. The comparison chains along the burst (each dropped sighting extends
        the visit), so a vehicle held in front of one camera yields a single
        sighting. The earliest sighting of the visit is the one kept — first
        arrival is the meaningful event. A same-camera sighting after a longer
        gap is a genuine revisit and is retained.
        """
        retained: List[TrajectorySighting] = []
        previous = None
        for sighting in sightings:
            if (previous is not None
                    and sighting.camera_id == previous.camera_id
                    and (sighting.timestamp - previous.timestamp).total_seconds()
                    <= self.duplicate_window_seconds):
                previous = sighting
                continue
            retained.append(sighting)
            previous = sighting
        return retained

    def _build_transitions(self, sightings) -> List[TrajectoryTransition]:
        """Pair consecutive sightings; attach the direct link when one exists."""
        transitions: List[TrajectoryTransition] = []
        for departure, arrival in zip(sightings, sightings[1:]):
            link = self.city_graph.get_link(departure.camera_id, arrival.camera_id)
            transitions.append(TrajectoryTransition(
                from_event_id=departure.event_id,
                to_event_id=arrival.event_id,
                from_camera_id=departure.camera_id,
                to_camera_id=arrival.camera_id,
                departed_at=departure.timestamp,
                arrived_at=arrival.timestamp,
                travel_time_seconds=(
                    arrival.timestamp - departure.timestamp).total_seconds(),
                link=link,
            ))
        return transitions


# --- CLI ----------------------------------------------------------------------

def _sighting_line(index: int, sighting) -> str:
    name = sighting.camera_name or "(camera not in topology)"
    return (f"  {index}. {sighting.timestamp.isoformat()}  "
            f"{sighting.camera_id}  {name}  conf={sighting.confidence:.2f}")


def _transition_line(transition) -> str:
    if transition.direct_link:
        link = (f"direct link {transition.distance_m:.0f}m via "
                f"{transition.road_name}")
        if transition.travel_direction:
            link += f" ({transition.travel_direction})"
    else:
        link = "no direct link in topology"
    return (f"  {transition.from_camera_id} -> {transition.to_camera_id}  "
            f"travel={transition.travel_time_seconds:.1f}s  {link}")


def _print_trajectory(trajectory) -> None:
    print(f"plate         : {trajectory.plate_normalized}")
    print(f"accepted obs  : {trajectory.accepted_count}")
    print(f"sightings     : {trajectory.sighting_count} "
          f"({trajectory.collapsed_duplicate_count} duplicate(s) collapsed)")
    if not trajectory.sightings:
        print("no accepted sightings for this plate in the requested window")
        return
    duration = trajectory.total_duration_seconds
    print(f"first seen    : {trajectory.first_seen.isoformat()}")
    print(f"last seen     : {trajectory.last_seen.isoformat()}")
    print(f"duration      : "
          f"{'n/a' if duration is None else format(duration, '.1f') + 's'}")
    if trajectory.unknown_camera_ids:
        print(f"unknown cams  : {', '.join(trajectory.unknown_camera_ids)}")
    if trajectory.truncated:
        print("warning       : observation cap reached; older sightings may be "
              "missing (narrow the time window)")
    print("sightings:")
    for i, sighting in enumerate(trajectory.sightings, start=1):
        print(_sighting_line(i, sighting))
    if trajectory.transitions:
        print("transitions:")
        for transition in trajectory.transitions:
            print(_transition_line(transition))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="CitySight Phase 2 cross-camera trajectory reconstruction")
    parser.add_argument("--plate", required=True,
                        help="Plate to trace (free-form; normalized on entry)")
    parser.add_argument("--start", default=None,
                        help="Inclusive lower time bound (tz-aware ISO-8601)")
    parser.add_argument("--end", default=None,
                        help="Inclusive upper time bound (tz-aware ISO-8601)")
    parser.add_argument("--city-config", default=str(DEFAULT_CITY_CONFIG_PATH),
                        help="Step 17 city topology config")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH),
                        help="Phase 1 pipeline config (persistence/thresholds)")
    parser.add_argument("--db", default=None,
                        help="Observations SQLite path (overrides --config)")
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="Print the trajectory as JSON instead of a summary")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        cameras, links = load_city_config(args.city_config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    db_path = args.db or (config.get("persistence") or {}).get(
        "db_path", "outputs/observations/observations.db")
    repo = SQLiteObservationRepository(db_path)
    try:
        reconstructor = TrajectoryReconstructor.from_config(
            repo, CityCameraGraph(cameras, links), config)
        trajectory = reconstructor.reconstruct(
            args.plate, start=args.start, end=args.end)
    except ValueError as exc:  # TrajectoryError, or a bad configured threshold
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        repo.close()

    if args.as_json:
        print(json.dumps(trajectory.to_dict(), indent=2))
    else:
        _print_trajectory(trajectory)
    return 0


if __name__ == "__main__":
    sys.exit(main())

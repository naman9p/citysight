"""Step 20: Phase 2 cross-camera trajectory reconstruction.

In-memory SQLite observations plus an in-process city graph — no weights, no
video decoding, no HTTP. Covers query validation, accepted-only filtering,
duplicate collapse, camera enrichment, transitions/direct links, time windows,
and the CLI.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest
import yaml

from phase1_anpr.observation.observation_builder import PlateObservation
from phase1_anpr.persistence import SQLiteObservationRepository
from phase2_city.graph import CityCameraGraph
from phase2_city.models import Camera, CameraLink
from phase2_city.trajectory import (
    TrajectoryDataError,
    TrajectoryQueryError,
    TrajectoryReconstructor,
    main,
)

PLATE = "MH12AB1234"
T0 = datetime(2026, 8, 31, 10, 0, 0, tzinfo=timezone.utc)


def _at(seconds):
    """UTC ISO-8601 timestamp ``seconds`` after the base time."""
    return (T0 + timedelta(seconds=seconds)).isoformat()


def _obs(event_id, camera_id, timestamp, *, plate=PLATE, status="accepted",
         confidence=0.9, evidence_ref=None):
    return PlateObservation(
        event_id=event_id, camera_id=camera_id, track_id=1,
        timestamp=timestamp, plate_raw=plate, plate_normalized=plate,
        confidence=confidence, status=status, detector_confidence=0.8,
        ocr_confidence=0.85, quality_score=0.7, best_frame_number=5,
        plate_image_path=evidence_ref, model_version="phase1-anpr-0.1.0",
    )


def _graph():
    """CAM_01 -> CAM_02 -> CAM_03; CAM_01 -> CAM_03 deliberately absent."""
    cams = [
        Camera(camera_id="CAM_01", name="Junction 1", latitude=28.61,
               longitude=77.22, road_name="Ring Road", heading_deg=90.0),
        Camera(camera_id="CAM_02", name="Junction 2", latitude=28.62,
               longitude=77.24, road_name="Ring Road", heading_deg=90.0),
        Camera(camera_id="CAM_03", name="Junction 3", latitude=28.63,
               longitude=77.26, road_name="Link Road", heading_deg=45.0),
    ]
    links = [
        CameraLink(from_camera_id="CAM_01", to_camera_id="CAM_02",
                   distance_m=1800.0, road_name="Ring Road",
                   travel_direction="eastbound"),
        CameraLink(from_camera_id="CAM_02", to_camera_id="CAM_03",
                   distance_m=1200.0, road_name="Link Road"),
    ]
    return CityCameraGraph(cams, links)


@pytest.fixture
def repo():
    r = SQLiteObservationRepository(":memory:")
    yield r
    r.close()


def _reconstructor(repo, **kwargs):
    return TrajectoryReconstructor(repo, _graph(), **kwargs)


def _save(repo, *observations):
    for obs in observations:
        repo.save(obs)


# --- ordering / basic shape ---------------------------------------------------

def test_two_camera_trajectory_is_ordered_with_one_transition(repo):
    _save(repo, _obs("e1", "CAM_01", _at(0)), _obs("e2", "CAM_02", _at(600)))
    traj = _reconstructor(repo).reconstruct(PLATE)
    assert traj.camera_sequence == ["CAM_01", "CAM_02"]
    assert [s.event_id for s in traj.sightings] == ["e1", "e2"]
    assert len(traj.transitions) == 1
    assert traj.total_duration_seconds == 600.0


def test_sightings_chronological_regardless_of_insert_order(repo):
    _save(repo, _obs("e3", "CAM_03", _at(1200)), _obs("e1", "CAM_01", _at(0)),
          _obs("e2", "CAM_02", _at(600)))
    traj = _reconstructor(repo).reconstruct(PLATE)
    assert [s.event_id for s in traj.sightings] == ["e1", "e2", "e3"]
    assert traj.first_seen == T0
    assert traj.last_seen == T0 + timedelta(seconds=1200)


def test_single_sighting_has_no_transitions_and_no_duration(repo):
    _save(repo, _obs("e1", "CAM_01", _at(0)))
    traj = _reconstructor(repo).reconstruct(PLATE)
    assert traj.sighting_count == 1
    assert traj.transitions == []
    assert traj.total_duration_seconds is None


def test_unknown_plate_returns_empty_trajectory(repo):
    _save(repo, _obs("e1", "CAM_01", _at(0)))
    traj = _reconstructor(repo).reconstruct("DL01AB0001")
    assert traj.plate_normalized == "DL01AB0001"
    assert traj.sightings == []
    assert traj.transitions == []
    assert traj.first_seen is None
    assert traj.total_duration_seconds is None


# --- accepted-only ------------------------------------------------------------

def test_review_and_abstained_observations_excluded(repo):
    _save(repo,
          _obs("e1", "CAM_01", _at(0)),
          _obs("e2", "CAM_02", _at(600), status="review"),
          _obs("e3", "CAM_03", _at(1200), status="abstained"),
          _obs("e4", "CAM_03", _at(1800)))
    traj = _reconstructor(repo).reconstruct(PLATE)
    assert [s.event_id for s in traj.sightings] == ["e1", "e4"]
    assert traj.accepted_count == 2


# --- duplicate collapse -------------------------------------------------------

def test_same_camera_within_window_collapsed(repo):
    _save(repo, _obs("e1", "CAM_01", _at(0)), _obs("e2", "CAM_01", _at(10)))
    traj = _reconstructor(repo).reconstruct(PLATE)
    assert traj.sighting_count == 1
    assert traj.accepted_count == 2
    assert traj.collapsed_duplicate_count == 1
    assert traj.transitions == []


def test_collapse_keeps_earliest_sighting_of_the_visit(repo):
    _save(repo, _obs("e1", "CAM_01", _at(0), confidence=0.7),
          _obs("e2", "CAM_01", _at(5), confidence=0.99))
    traj = _reconstructor(repo).reconstruct(PLATE)
    assert [s.event_id for s in traj.sightings] == ["e1"]


def test_burst_chains_into_one_visit(repo):
    # 20s apart each: every hop is inside the 30s window, so the whole burst is
    # one camera visit even though it spans 60s.
    _save(repo, _obs("e1", "CAM_01", _at(0)), _obs("e2", "CAM_01", _at(20)),
          _obs("e3", "CAM_01", _at(40)), _obs("e4", "CAM_01", _at(60)),
          _obs("e5", "CAM_02", _at(160)))
    traj = _reconstructor(repo).reconstruct(PLATE)
    assert [s.event_id for s in traj.sightings] == ["e1", "e5"]
    assert traj.collapsed_duplicate_count == 3
    assert traj.transitions[0].travel_time_seconds == 160.0


def test_same_camera_after_window_is_a_revisit(repo):
    _save(repo, _obs("e1", "CAM_01", _at(0)), _obs("e2", "CAM_01", _at(3600)))
    traj = _reconstructor(repo).reconstruct(PLATE)
    assert traj.camera_sequence == ["CAM_01", "CAM_01"]
    transition = traj.transitions[0]
    assert transition.travel_time_seconds == 3600.0
    # A self-transition has no topology link (self-links are not modelled).
    assert transition.direct_link is False


def test_different_cameras_never_collapsed(repo):
    _save(repo, _obs("e1", "CAM_01", _at(0)), _obs("e2", "CAM_02", _at(5)))
    traj = _reconstructor(repo).reconstruct(PLATE)
    assert traj.camera_sequence == ["CAM_01", "CAM_02"]
    assert traj.collapsed_duplicate_count == 0


def test_zero_window_keeps_distinct_times_but_collapses_identical(repo):
    _save(repo, _obs("e1", "CAM_01", _at(0)), _obs("e2", "CAM_01", _at(1)))
    traj = _reconstructor(repo, duplicate_window_seconds=0).reconstruct(PLATE)
    assert [s.event_id for s in traj.sightings] == ["e1", "e2"]

    _save(repo, _obs("e3", "CAM_02", _at(500)), _obs("e4", "CAM_02", _at(500)))
    traj = _reconstructor(repo, duplicate_window_seconds=0).reconstruct(PLATE)
    assert [s.event_id for s in traj.sightings] == ["e1", "e2", "e3"]


def test_negative_window_rejected(repo):
    with pytest.raises(ValueError, match="duplicate_window_seconds"):
        _reconstructor(repo, duplicate_window_seconds=-1)


def test_zero_max_observations_rejected(repo):
    with pytest.raises(ValueError, match="max_observations"):
        _reconstructor(repo, max_observations=0)


# --- camera enrichment --------------------------------------------------------

def test_camera_metadata_enriched_from_graph(repo):
    _save(repo, _obs("e1", "CAM_02", _at(0), evidence_ref="e1.jpg"))
    sighting = _reconstructor(repo).reconstruct(PLATE).sightings[0]
    assert sighting.camera_known is True
    assert sighting.camera_name == "Junction 2"
    assert (sighting.latitude, sighting.longitude) == (28.62, 77.24)
    assert sighting.road_name == "Ring Road"
    assert sighting.heading_deg == 90.0
    assert sighting.evidence_ref == "e1.jpg"
    assert sighting.plate_normalized == PLATE


def test_unknown_camera_keeps_id_with_null_metadata(repo):
    _save(repo, _obs("e1", "CAM_99", _at(0)))
    traj = _reconstructor(repo).reconstruct(PLATE)
    sighting = traj.sightings[0]
    assert sighting.camera_id == "CAM_99"
    assert sighting.camera_known is False
    assert sighting.camera_name is None
    assert sighting.latitude is None and sighting.longitude is None
    assert sighting.road_name is None and sighting.heading_deg is None
    assert traj.unknown_camera_ids == ["CAM_99"]


# --- transitions / direct links ----------------------------------------------

def test_direct_link_metadata_attached(repo):
    _save(repo, _obs("e1", "CAM_01", _at(0)), _obs("e2", "CAM_02", _at(300)))
    transition = _reconstructor(repo).reconstruct(PLATE).transitions[0]
    assert transition.direct_link is True
    assert transition.distance_m == 1800.0
    assert transition.road_name == "Ring Road"
    assert transition.travel_direction == "eastbound"
    assert transition.travel_time_seconds == 300.0
    assert transition.departed_at == T0
    assert transition.arrived_at == T0 + timedelta(seconds=300)
    assert (transition.from_event_id, transition.to_event_id) == ("e1", "e2")


def test_direct_link_without_travel_direction(repo):
    _save(repo, _obs("e1", "CAM_02", _at(0)), _obs("e2", "CAM_03", _at(120)))
    transition = _reconstructor(repo).reconstruct(PLATE).transitions[0]
    assert transition.direct_link is True
    assert transition.road_name == "Link Road"
    assert transition.travel_direction is None


def test_missing_link_reports_no_direct_link(repo):
    # CAM_01 -> CAM_03 is not in the topology: an unobserved gap, not a route.
    _save(repo, _obs("e1", "CAM_01", _at(0)), _obs("e2", "CAM_03", _at(900)))
    transition = _reconstructor(repo).reconstruct(PLATE).transitions[0]
    assert transition.direct_link is False
    assert transition.distance_m is None
    assert transition.road_name is None
    assert transition.travel_direction is None
    assert transition.travel_time_seconds == 900.0


def test_reverse_direction_is_not_a_direct_link(repo):
    # Only CAM_01 -> CAM_02 exists; the reverse trip must not borrow it.
    _save(repo, _obs("e1", "CAM_02", _at(0)), _obs("e2", "CAM_01", _at(300)))
    transition = _reconstructor(repo).reconstruct(PLATE).transitions[0]
    assert (transition.from_camera_id, transition.to_camera_id) == ("CAM_02",
                                                                   "CAM_01")
    assert transition.direct_link is False
    assert transition.distance_m is None


def test_transition_count_is_one_less_than_sightings(repo):
    _save(repo, _obs("e1", "CAM_01", _at(0)), _obs("e2", "CAM_02", _at(600)),
          _obs("e3", "CAM_03", _at(1200)))
    traj = _reconstructor(repo).reconstruct(PLATE)
    assert traj.sighting_count == 3
    assert len(traj.transitions) == 2


# --- query validation ---------------------------------------------------------

def test_free_form_plate_input_normalized(repo):
    _save(repo, _obs("e1", "CAM_01", _at(0)))
    traj = _reconstructor(repo).reconstruct(" mh-12 ab 1234 ")
    assert traj.plate_normalized == PLATE
    assert traj.sighting_count == 1


def test_empty_plate_rejected(repo):
    with pytest.raises(TrajectoryQueryError, match="empty after normalization"):
        _reconstructor(repo).reconstruct("!!!")


def test_none_plate_rejected(repo):
    with pytest.raises(TrajectoryQueryError, match="plate is required"):
        _reconstructor(repo).reconstruct(None)


def test_invalid_plate_format_rejected(repo):
    with pytest.raises(TrajectoryQueryError, match="not a valid Indian plate"):
        _reconstructor(repo).reconstruct("ZZ99ZZ")


def test_unknown_state_code_rejected(repo):
    with pytest.raises(TrajectoryQueryError, match="not a valid Indian plate"):
        _reconstructor(repo).reconstruct("XX12AB1234")


def test_naive_start_bound_rejected(repo):
    with pytest.raises(TrajectoryQueryError, match="timezone"):
        _reconstructor(repo).reconstruct(PLATE, start="2026-08-31T10:00:00")


def test_malformed_end_bound_rejected(repo):
    with pytest.raises(TrajectoryQueryError, match="ISO-8601"):
        _reconstructor(repo).reconstruct(PLATE, end="yesterday")


def test_start_after_end_rejected(repo):
    with pytest.raises(TrajectoryQueryError, match="must not be after end"):
        _reconstructor(repo).reconstruct(PLATE, start=_at(600), end=_at(0))


# --- time window --------------------------------------------------------------

def test_window_bounds_are_inclusive(repo):
    _save(repo, _obs("e1", "CAM_01", _at(0)), _obs("e2", "CAM_02", _at(600)),
          _obs("e3", "CAM_03", _at(1200)))
    traj = _reconstructor(repo).reconstruct(PLATE, start=_at(0), end=_at(600))
    assert [s.event_id for s in traj.sightings] == ["e1", "e2"]


def test_window_excludes_sightings_outside_bounds(repo):
    _save(repo, _obs("e1", "CAM_01", _at(0)), _obs("e2", "CAM_02", _at(600)),
          _obs("e3", "CAM_03", _at(1200)))
    traj = _reconstructor(repo).reconstruct(PLATE, start=_at(1), end=_at(1199))
    assert [s.event_id for s in traj.sightings] == ["e2"]
    assert traj.transitions == []


def test_window_accepts_offset_and_z_bounds(repo):
    _save(repo, _obs("e1", "CAM_01", _at(0)))
    r = _reconstructor(repo)
    assert r.reconstruct(PLATE, start="2026-08-31T15:30:00+05:30",
                         end="2026-08-31T10:00:01Z").sighting_count == 1


def test_non_utc_stored_offset_outside_window_excluded(repo):
    # 15:00+05:30 is 09:30Z — before the window — even though the stored string
    # sorts inside it. The parsed instant decides.
    _save(repo, _obs("e1", "CAM_01", "2026-08-31T15:00:00+05:30"),
          _obs("e2", "CAM_02", _at(600)))
    traj = _reconstructor(repo).reconstruct(PLATE, start=_at(0), end=_at(36000))
    assert [s.event_id for s in traj.sightings] == ["e2"]


def test_repository_bounds_are_inclusive_and_optional(repo):
    _save(repo, _obs("e1", "CAM_01", _at(0)), _obs("e2", "CAM_02", _at(600)),
          _obs("e3", "CAM_03", _at(1200)))
    # Legacy positional call is unchanged (no bounds -> everything).
    assert len(repo.list_by_plate(PLATE, 50)) == 3
    bounded = repo.list_by_plate(PLATE, 50, start=T0 + timedelta(seconds=600),
                                 end=T0 + timedelta(seconds=1200))
    assert [r["event_id"] for r in bounded] == ["e3", "e2"]  # newest first
    assert len(repo.list_by_plate(PLATE, 50, start=T0 + timedelta(hours=1))) == 0

"""Step 21: trajectory HTTP API (POST /v1/trajectories) + serve.py wiring.

Real stdlib server on an ephemeral port, httpx client, in-memory SQLite —
no weights, no video decoding, no HTTP mock. Follows the same pattern as
test_api.py and test_watchlist.py.
"""

import threading
from datetime import datetime, timedelta, timezone

import httpx
import pytest
import yaml

from phase1_anpr.api import create_server
from phase1_anpr.observation.observation_builder import PlateObservation
from phase1_anpr.persistence import SQLiteObservationRepository
from phase1_anpr.serve import build_trajectory_reconstructor
from phase2_city.graph import CityCameraGraph
from phase2_city.models import Camera, CameraLink
from phase2_city.trajectory import TrajectoryReconstructor

PLATE = "MH12AB1234"
T0 = datetime(2026, 8, 31, 10, 0, 0, tzinfo=timezone.utc)


def _at(seconds):
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


def _seed_repo(repo):
    """Seed observations: CAM_01 at T0, CAM_02 at T0+600, CAM_03 at T0+1200."""
    repo.save(_obs("e1", "CAM_01", _at(0)))
    repo.save(_obs("e2", "CAM_02", _at(600)))
    repo.save(_obs("e3", "CAM_03", _at(1200)))


# --- fixtures -----------------------------------------------------------------

@pytest.fixture
def client():
    """Server WITH trajectory support enabled."""
    repo = SQLiteObservationRepository(":memory:")
    _seed_repo(repo)
    graph = _graph()
    reconstructor = TrajectoryReconstructor(repo, graph)
    server = create_server(repo, port=0,
                           trajectory_reconstructor=reconstructor)
    host, port = server.server_address
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    with httpx.Client(base_url=f"http://{host}:{port}") as c:
        yield c
    server.shutdown()
    repo.close()


@pytest.fixture
def client_no_trajectory():
    """Server WITHOUT trajectory support (no trajectory_reconstructor)."""
    repo = SQLiteObservationRepository(":memory:")
    server = create_server(repo, port=0)
    host, port = server.server_address
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    with httpx.Client(base_url=f"http://{host}:{port}") as c:
        yield c
    server.shutdown()
    repo.close()


# --- POST /v1/trajectories success -------------------------------------------

def test_trajectory_basic_response_shape(client):
    r = client.post("/v1/trajectories", json={"plate": PLATE})
    assert r.status_code == 200
    body = r.json()
    assert body["plate_normalized"] == PLATE
    assert body["sighting_count"] == 3
    assert body["accepted_observation_count"] == 3
    assert body["collapsed_duplicate_count"] == 0
    assert body["truncated"] is False
    assert body["first_seen"] is not None
    assert body["last_seen"] is not None
    assert body["total_duration_seconds"] == 1200.0
    assert body["camera_sequence"] == ["CAM_01", "CAM_02", "CAM_03"]
    assert isinstance(body["sightings"], list) and len(body["sightings"]) == 3
    assert isinstance(body["transitions"], list) and len(body["transitions"]) == 2


def test_trajectory_with_time_window(client):
    r = client.post("/v1/trajectories", json={
        "plate": PLATE,
        "start": _at(0),
        "end": _at(600),
    })
    assert r.status_code == 200
    body = r.json()
    ids = [s["event_id"] for s in body["sightings"]]
    assert ids == ["e1", "e2"]
    assert body["sighting_count"] == 2
    assert len(body["transitions"]) == 1


def test_trajectory_unknown_plate_empty(client):
    r = client.post("/v1/trajectories", json={"plate": "DL01AB0001"})
    assert r.status_code == 200
    body = r.json()
    assert body["plate_normalized"] == "DL01AB0001"
    assert body["sighting_count"] == 0
    assert body["sightings"] == []
    assert body["transitions"] == []
    assert body["first_seen"] is None
    assert body["total_duration_seconds"] is None


def test_trajectory_sightings_have_camera_metadata(client):
    r = client.post("/v1/trajectories", json={"plate": PLATE})
    assert r.status_code == 200
    sighting = r.json()["sightings"][0]
    assert sighting["camera_id"] == "CAM_01"
    assert sighting["camera_name"] == "Junction 1"
    assert sighting["latitude"] == 28.61
    assert sighting["longitude"] == 77.22
    assert sighting["road_name"] == "Ring Road"
    assert sighting["heading_deg"] == 90.0


def test_trajectory_transitions_have_link_metadata(client):
    r = client.post("/v1/trajectories", json={"plate": PLATE})
    assert r.status_code == 200
    transitions = r.json()["transitions"]

    # CAM_01 -> CAM_02: direct link exists
    t0 = transitions[0]
    assert t0["from_camera_id"] == "CAM_01"
    assert t0["to_camera_id"] == "CAM_02"
    assert t0["direct_link"] is True
    assert t0["distance_m"] == 1800.0
    assert t0["road_name"] == "Ring Road"
    assert t0["travel_direction"] == "eastbound"
    assert t0["travel_time_seconds"] == 600.0

    # CAM_02 -> CAM_03: direct link exists (no travel_direction on this link)
    t1 = transitions[1]
    assert t1["direct_link"] is True
    assert t1["distance_m"] == 1200.0
    assert t1["road_name"] == "Link Road"
    assert t1["travel_direction"] is None


# --- POST /v1/trajectories client errors (400) -------------------------------

def test_trajectory_invalid_plate_400(client):
    r = client.post("/v1/trajectories", json={"plate": "ZZ99ZZ"})
    assert r.status_code == 400
    assert "error" in r.json()


def test_trajectory_missing_plate_400(client):
    r = client.post("/v1/trajectories", json={})
    assert r.status_code == 400
    assert r.json()["error"] == "plate is required"


def test_trajectory_non_string_plate_400(client):
    r = client.post("/v1/trajectories", json={"plate": 123})
    assert r.status_code == 400
    assert r.json()["error"] == "plate must be a string"


def test_trajectory_non_object_body_400(client):
    r = client.post("/v1/trajectories",
                    content=b'["not", "an", "object"]',
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 400
    assert r.json()["error"] == "JSON object body required"


def test_trajectory_malformed_json_400(client):
    r = client.post("/v1/trajectories",
                    content=b'{broken',
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 400
    assert r.json()["error"] == "invalid JSON body"


def test_trajectory_naive_timestamp_400(client):
    r = client.post("/v1/trajectories", json={
        "plate": PLATE, "start": "2026-08-31T10:00:00"})
    assert r.status_code == 400
    assert "timezone" in r.json()["error"].lower()


def test_trajectory_malformed_iso_timestamp_400(client):
    r = client.post("/v1/trajectories", json={
        "plate": PLATE, "start": "not-a-date"})
    assert r.status_code == 400
    assert "error" in r.json()


def test_trajectory_start_after_end_400(client):
    r = client.post("/v1/trajectories", json={
        "plate": PLATE, "start": _at(600), "end": _at(0)})
    assert r.status_code == 400
    assert "must not be after end" in r.json()["error"]


# --- trajectory not enabled (404) --------------------------------------------

def test_trajectory_not_enabled_404(client_no_trajectory):
    r = client_no_trajectory.post("/v1/trajectories",
                                  json={"plate": PLATE})
    assert r.status_code == 404
    assert r.json()["error"] == "trajectory not enabled"


# --- existing endpoints still work with trajectory enabled --------------------

def test_existing_health_still_works(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_existing_observations_still_works(client):
    r = client.get("/observations")
    assert r.status_code == 200
    assert isinstance(r.json(), list) and len(r.json()) == 3


# --- serve.py build_trajectory_reconstructor ---------------------------------

def test_serve_missing_city_config_returns_none():
    repo = SQLiteObservationRepository(":memory:")
    try:
        result = build_trajectory_reconstructor(
            repo, "/nonexistent/path/city.yaml", {})
        assert result is None
    finally:
        repo.close()


def test_serve_malformed_city_config_fails(tmp_path):
    bad_config = tmp_path / "city.yaml"
    # cameras must be a list; a scalar triggers CityConfigError (a ValueError).
    bad_config.write_text("cameras: not-a-list\n", encoding="utf-8")
    repo = SQLiteObservationRepository(":memory:")
    try:
        with pytest.raises(ValueError):
            build_trajectory_reconstructor(repo, str(bad_config), {})
    finally:
        repo.close()


def test_serve_valid_city_config_returns_reconstructor(tmp_path):
    config_data = {
        "cameras": [
            {"camera_id": "CAM_A", "name": "A", "latitude": 28.0,
             "longitude": 77.0, "road_name": "Main Rd", "heading_deg": 0.0},
        ],
        "links": [],
    }
    config_file = tmp_path / "city.yaml"
    config_file.write_text(yaml.dump(config_data), encoding="utf-8")
    repo = SQLiteObservationRepository(":memory:")
    try:
        result = build_trajectory_reconstructor(repo, str(config_file), {})
        assert result is not None
        assert isinstance(result, TrajectoryReconstructor)
    finally:
        repo.close()

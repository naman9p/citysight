"""Step 22: camera topology API (GET /v1/cameras, /v1/cameras/{id},
/v1/cameras/{id}/links, /v1/links).

Real stdlib server on an ephemeral port, httpx client, in-memory data —
no weights, no video decoding, no HTTP mock. Follows the same pattern as
test_api.py and test_trajectory_api.py.
"""

import threading
from datetime import datetime, timedelta, timezone

import httpx
import pytest
import yaml

from phase1_anpr.api import create_server
from phase1_anpr.observation.observation_builder import PlateObservation
from phase1_anpr.persistence import SQLiteObservationRepository
from phase1_anpr.serve import build_city_graph, build_trajectory_reconstructor
from phase2_city.graph import CityCameraGraph
from phase2_city.models import Camera, CameraLink
from phase2_city.trajectory import TrajectoryReconstructor


# --- shared test data ----------------------------------------------------------

# Cameras are defined OUT of alphabetical order to prove sorting.
CAMERAS = [
    Camera(camera_id="CAM_03", name="Junction 3", latitude=28.63,
           longitude=77.26, road_name="Link Road", heading_deg=45.0,
           zone="north"),
    Camera(camera_id="CAM_01", name="Junction 1", latitude=28.61,
           longitude=77.22, road_name="Ring Road", heading_deg=90.0,
           zone="central", enabled=True),
    Camera(camera_id="CAM_02", name="Junction 2", latitude=28.62,
           longitude=77.24, road_name="Ring Road", heading_deg=90.0,
           zone="central"),
]

LINKS = [
    # Defined out of sorted order to prove sorting.
    CameraLink(from_camera_id="CAM_02", to_camera_id="CAM_03",
               distance_m=1200.0, road_name="Link Road",
               travel_direction="northeastbound"),
    CameraLink(from_camera_id="CAM_01", to_camera_id="CAM_02",
               distance_m=1800.0, road_name="Ring Road",
               travel_direction="eastbound"),
    CameraLink(from_camera_id="CAM_02", to_camera_id="CAM_01",
               distance_m=1800.0, road_name="Ring Road",
               travel_direction="westbound"),
]

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
    return CityCameraGraph(CAMERAS, LINKS)


def _seed_repo(repo):
    repo.save(_obs("e1", "CAM_01", _at(0)))
    repo.save(_obs("e2", "CAM_02", _at(600)))
    repo.save(_obs("e3", "CAM_03", _at(1200)))


# --- fixtures ------------------------------------------------------------------

@pytest.fixture
def client():
    """Server WITH topology AND trajectory support enabled."""
    repo = SQLiteObservationRepository(":memory:")
    _seed_repo(repo)
    graph = _graph()
    reconstructor = TrajectoryReconstructor(repo, graph)
    server = create_server(repo, port=0,
                           trajectory_reconstructor=reconstructor,
                           city_graph=graph)
    host, port = server.server_address
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    with httpx.Client(base_url=f"http://{host}:{port}") as c:
        yield c
    server.shutdown()
    repo.close()


@pytest.fixture
def client_no_topology():
    """Server WITHOUT topology (no city_graph, no trajectory)."""
    repo = SQLiteObservationRepository(":memory:")
    server = create_server(repo, port=0)
    host, port = server.server_address
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    with httpx.Client(base_url=f"http://{host}:{port}") as c:
        yield c
    server.shutdown()
    repo.close()


# === GET /v1/cameras ==========================================================

def test_list_cameras_returns_all(client):
    r = client.get("/v1/cameras")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
    assert len(body) == 3


def test_list_cameras_sorted_by_camera_id(client):
    """Cameras are returned sorted by camera_id regardless of insertion order."""
    r = client.get("/v1/cameras")
    ids = [c["camera_id"] for c in r.json()]
    assert ids == ["CAM_01", "CAM_02", "CAM_03"]


def test_list_cameras_response_shape(client):
    r = client.get("/v1/cameras")
    cam = r.json()[0]  # CAM_01 after sorting
    assert cam["camera_id"] == "CAM_01"
    assert cam["name"] == "Junction 1"
    assert cam["latitude"] == 28.61
    assert cam["longitude"] == 77.22
    assert cam["road_name"] == "Ring Road"
    assert cam["heading_deg"] == 90.0
    assert cam["zone"] == "central"
    assert cam["enabled"] is True


def test_list_cameras_not_enabled_404(client_no_topology):
    r = client_no_topology.get("/v1/cameras")
    assert r.status_code == 404
    assert r.json()["error"] == "camera topology not enabled"


# === GET /v1/cameras/{camera_id} ==============================================

def test_get_camera_ok(client):
    r = client.get("/v1/cameras/CAM_02")
    assert r.status_code == 200
    body = r.json()
    assert body["camera_id"] == "CAM_02"
    assert body["name"] == "Junction 2"
    assert body["latitude"] == 28.62
    assert body["longitude"] == 77.24


def test_get_camera_not_found(client):
    r = client.get("/v1/cameras/CAM_99")
    assert r.status_code == 404
    assert r.json()["error"] == "camera not found: CAM_99"


def test_get_camera_url_decoded(client):
    """URL-encoded camera ID is decoded before graph lookup."""
    r = client.get("/v1/cameras/CAM%5F01")  # CAM_01 with encoded underscore
    assert r.status_code == 200
    assert r.json()["camera_id"] == "CAM_01"


def test_get_camera_not_enabled_404(client_no_topology):
    r = client_no_topology.get("/v1/cameras/CAM_01")
    assert r.status_code == 404
    assert r.json()["error"] == "camera topology not enabled"


# === GET /v1/cameras/{camera_id}/links ========================================

def test_camera_links_response_shape(client):
    """CAM_02 has outgoing to CAM_01 and CAM_03, incoming from CAM_01."""
    r = client.get("/v1/cameras/CAM_02/links")
    assert r.status_code == 200
    body = r.json()
    assert body["camera_id"] == "CAM_02"
    assert isinstance(body["outgoing"], list)
    assert isinstance(body["incoming"], list)


def test_camera_links_outgoing_sorted(client):
    """CAM_02 outgoing: CAM_02->CAM_01 and CAM_02->CAM_03, sorted."""
    r = client.get("/v1/cameras/CAM_02/links")
    outgoing = r.json()["outgoing"]
    out_targets = [(lk["from_camera_id"], lk["to_camera_id"]) for lk in outgoing]
    assert out_targets == [("CAM_02", "CAM_01"), ("CAM_02", "CAM_03")]


def test_camera_links_incoming_sorted(client):
    """CAM_02 incoming: CAM_01->CAM_02, sorted."""
    r = client.get("/v1/cameras/CAM_02/links")
    incoming = r.json()["incoming"]
    assert len(incoming) == 1
    assert incoming[0]["from_camera_id"] == "CAM_01"
    assert incoming[0]["to_camera_id"] == "CAM_02"


def test_camera_links_leaf_no_outgoing(client):
    """CAM_03 has no outgoing links but has incoming."""
    r = client.get("/v1/cameras/CAM_03/links")
    body = r.json()
    assert body["outgoing"] == []
    assert len(body["incoming"]) == 1
    assert body["incoming"][0]["from_camera_id"] == "CAM_02"


def test_camera_links_root_no_incoming(client):
    """CAM_01 has outgoing to CAM_02 but no incoming links."""
    r = client.get("/v1/cameras/CAM_01/links")
    body = r.json()
    assert len(body["outgoing"]) == 1
    assert body["outgoing"][0]["to_camera_id"] == "CAM_02"
    # CAM_01 has incoming from CAM_02 (the westbound link).
    assert len(body["incoming"]) == 1
    assert body["incoming"][0]["from_camera_id"] == "CAM_02"


def test_camera_links_link_fields(client):
    """Verify full link serialization shape."""
    r = client.get("/v1/cameras/CAM_01/links")
    lk = r.json()["outgoing"][0]
    assert lk["from_camera_id"] == "CAM_01"
    assert lk["to_camera_id"] == "CAM_02"
    assert lk["distance_m"] == 1800.0
    assert lk["road_name"] == "Ring Road"
    assert lk["travel_direction"] == "eastbound"


def test_camera_links_unknown_camera_404(client):
    r = client.get("/v1/cameras/NOPE/links")
    assert r.status_code == 404
    assert r.json()["error"] == "camera not found: NOPE"


def test_camera_links_url_decoded(client):
    """URL-encoded camera ID in links endpoint."""
    r = client.get("/v1/cameras/CAM%5F02/links")
    assert r.status_code == 200
    assert r.json()["camera_id"] == "CAM_02"


def test_camera_links_not_enabled_404(client_no_topology):
    r = client_no_topology.get("/v1/cameras/CAM_01/links")
    assert r.status_code == 404
    assert r.json()["error"] == "camera topology not enabled"


# === GET /v1/links =============================================================

def test_list_links_returns_all(client):
    r = client.get("/v1/links")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
    assert len(body) == 3


def test_list_links_sorted_deterministically(client):
    """Links sorted by (from_camera_id, to_camera_id)."""
    r = client.get("/v1/links")
    keys = [(lk["from_camera_id"], lk["to_camera_id"]) for lk in r.json()]
    assert keys == [
        ("CAM_01", "CAM_02"),
        ("CAM_02", "CAM_01"),
        ("CAM_02", "CAM_03"),
    ]


def test_list_links_response_shape(client):
    r = client.get("/v1/links")
    lk = r.json()[0]
    expected_keys = {"from_camera_id", "to_camera_id", "distance_m",
                     "road_name", "travel_direction"}
    assert set(lk.keys()) == expected_keys


def test_list_links_not_enabled_404(client_no_topology):
    r = client_no_topology.get("/v1/links")
    assert r.status_code == 404
    assert r.json()["error"] == "camera topology not enabled"


# === backward compatibility ===================================================

def test_existing_health_still_works(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_existing_observations_still_works(client):
    r = client.get("/observations")
    assert r.status_code == 200
    assert isinstance(r.json(), list) and len(r.json()) == 3


def test_phase1_camera_observations_still_works(client):
    """Phase 1 un-prefixed /cameras/{id}/observations route."""
    r = client.get("/cameras/CAM_01/observations")
    assert r.status_code == 200
    ids = [o["event_id"] for o in r.json()]
    assert "e1" in ids


def test_trajectory_still_works_with_topology(client):
    """POST /v1/trajectories exercises the real trajectory endpoint."""
    r = client.post("/v1/trajectories", json={"plate": PLATE})
    assert r.status_code == 200
    body = r.json()
    assert body["plate_normalized"] == PLATE
    assert body["sighting_count"] == 3
    assert body["camera_sequence"] == ["CAM_01", "CAM_02", "CAM_03"]
    assert isinstance(body["sightings"], list) and len(body["sightings"]) == 3
    assert isinstance(body["transitions"], list) and len(body["transitions"]) == 2
    # Verify transition metadata comes from the shared graph
    t0 = body["transitions"][0]
    assert t0["from_camera_id"] == "CAM_01"
    assert t0["to_camera_id"] == "CAM_02"
    assert t0["direct_link"] is True
    assert t0["distance_m"] == 1800.0


def test_trajectory_not_enabled_when_no_topology(client_no_topology):
    r = client_no_topology.post("/v1/trajectories", json={"plate": PLATE})
    assert r.status_code == 404
    assert r.json()["error"] == "trajectory not enabled"


# === serve.py helpers ==========================================================

def test_build_city_graph_missing_returns_none():
    result = build_city_graph("/nonexistent/path/city.yaml")
    assert result is None


def test_build_city_graph_valid(tmp_path):
    config_data = {
        "cameras": [
            {"camera_id": "X1", "name": "X", "latitude": 28.0,
             "longitude": 77.0, "road_name": "Rd", "heading_deg": 0.0},
        ],
        "links": [],
    }
    cfg = tmp_path / "city.yaml"
    cfg.write_text(yaml.dump(config_data), encoding="utf-8")
    graph = build_city_graph(str(cfg))
    assert graph is not None
    assert isinstance(graph, CityCameraGraph)
    assert graph.camera_count == 1


def test_build_city_graph_malformed_raises(tmp_path):
    bad = tmp_path / "city.yaml"
    bad.write_text("cameras: not-a-list\n", encoding="utf-8")
    with pytest.raises(ValueError):
        build_city_graph(str(bad))


def test_build_trajectory_reconstructor_backward_compat():
    """Existing callers (no city_graph kwarg) still work."""
    repo = SQLiteObservationRepository(":memory:")
    try:
        result = build_trajectory_reconstructor(
            repo, "/nonexistent/path/city.yaml", {})
        assert result is None
    finally:
        repo.close()


def test_build_trajectory_reconstructor_with_graph():
    """When city_graph is provided, it is reused."""
    repo = SQLiteObservationRepository(":memory:")
    graph = _graph()
    try:
        result = build_trajectory_reconstructor(
            repo, "/nonexistent/path/city.yaml", {}, city_graph=graph)
        assert result is not None
        assert isinstance(result, TrajectoryReconstructor)
    finally:
        repo.close()

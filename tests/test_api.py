"""Tests for Step 13 read-only API. Drives a real stdlib server over httpx with an
in-memory SQLite repository. No network/Docker/models/external services."""

import threading

import httpx
import pytest

from phase1_anpr.api import create_server
from phase1_anpr.observation.observation_builder import PlateObservation
from phase1_anpr.persistence import SQLiteObservationRepository


def obs(event_id, camera_id="cam_01", plate="MH12AB1234", ts="2026-08-31T10:00:00",
        status="accepted"):
    abstained = status == "abstained"
    return PlateObservation(
        event_id=event_id, camera_id=camera_id, track_id=1, timestamp=ts,
        plate_raw=None if abstained else plate,
        plate_normalized=None if abstained else plate,
        confidence=0.2 if abstained else 0.9, status=status,
        detector_confidence=None if abstained else 0.8,
        ocr_confidence=None if abstained else 0.85,
        quality_score=None if abstained else 0.7,
        best_frame_number=None if abstained else 5,
        plate_image_path=None, model_version="phase1-anpr-0.1.0",
    )


@pytest.fixture
def client():
    repo = SQLiteObservationRepository(":memory:")
    repo.save(obs("e1", ts="2026-08-31T10:00:00"), format_type="standard", state_code="MH")
    repo.save(obs("e2", ts="2026-08-31T11:00:00"), format_type="standard", state_code="MH")
    repo.save(obs("e3", camera_id="cam_02", plate="KA05MN4321",
                  ts="2026-08-31T09:00:00"))
    repo.save(obs("abs1", status="abstained", ts="2026-08-31T12:00:00"))

    server = create_server(repo, port=0)
    host, port = server.server_address
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    with httpx.Client(base_url=f"http://{host}:{port}") as c:
        yield c
    server.shutdown()
    repo.close()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_observations_recent_newest_first(client):
    r = client.get("/observations")
    assert r.status_code == 200
    ids = [o["event_id"] for o in r.json()]
    assert ids[0] == "abs1"  # 12:00 newest
    assert ids.index("e2") < ids.index("e1")  # 11:00 before 10:00


def test_observations_limit(client):
    r = client.get("/observations", params={"limit": 2})
    assert r.status_code == 200 and len(r.json()) == 2


def test_invalid_limit_4xx(client):
    assert client.get("/observations", params={"limit": "abc"}).status_code == 400
    assert client.get("/observations", params={"limit": 0}).status_code == 400


def test_get_observation_ok_and_404(client):
    assert client.get("/observations/e1").json()["event_id"] == "e1"
    assert client.get("/observations/nope").status_code == 404


def test_abstained_has_no_plate_identity(client):
    row = client.get("/observations/abs1").json()
    assert row["status"] == "abstained"
    assert row["plate_raw"] is None and row["plate_normalized"] is None


def test_plate_search_exact_normalized_newest_first(client):
    # Query with separators/lowercase -> normalized to MH12AB1234.
    r = client.get("/plates/mh-12-ab-1234/observations")
    assert r.status_code == 200
    ids = [o["event_id"] for o in r.json()]
    assert ids == ["e2", "e1"]


def test_plate_search_empty_result(client):
    r = client.get("/plates/XX00YY0000/observations")
    assert r.status_code == 200 and r.json() == []


def test_plate_search_empty_query_4xx(client):
    assert client.get("/plates/---/observations").status_code == 400


def test_cameras_observations(client):
    r = client.get("/cameras/cam_02/observations")
    assert r.status_code == 200
    ids = [o["event_id"] for o in r.json()]
    assert ids == ["e3"]

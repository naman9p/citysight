"""Tests for Step 14 operator dashboard. Serves the real stdlib server over httpx
with an in-memory SQLite repository. No external network/Docker/models."""

import threading

import httpx
import pytest

from phase1_anpr.api import create_server
from phase1_anpr.observation.observation_builder import PlateObservation
from phase1_anpr.persistence import SQLiteObservationRepository


def obs(event_id, camera_id="cam_01", plate="MH12AB1234",
        ts="2026-08-31T10:00:00", status="accepted"):
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
    repo.save(obs("e1"))
    repo.save(obs("abs1", status="abstained", ts="2026-08-31T12:00:00"))
    server = create_server(repo, port=0)
    host, port = server.server_address
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with httpx.Client(base_url=f"http://{host}:{port}") as c:
        yield c
    server.shutdown()
    repo.close()


def test_dashboard_returns_html(client):
    r = client.get("/dashboard")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "<!doctype html>" in r.text.lower()
    assert "CitySight" in r.text


def test_dashboard_has_map_empty_notice(client):
    r = client.get("/dashboard")
    assert "No camera locations configured" in r.text


def test_dashboard_polls_existing_api(client):
    # The page must drive the existing API endpoints, not internal state.
    body = client.get("/dashboard").text
    assert "/observations?limit=50" in body
    assert "/plates/" in body
    # No SSE/WebSocket transport.
    assert "EventSource" not in body and "WebSocket" not in body


def test_dashboard_renders_safely_no_innerhtml(client):
    # Data must be injected via textContent, never innerHTML.
    body = client.get("/dashboard").text
    assert "textContent" in body
    assert "innerHTML" not in body


def test_dashboard_hides_abstained_plate_client_side(client):
    body = client.get("/dashboard").text
    # Guard exists: abstained rows resolve to a dash, not a plate identity.
    assert 'o.status === "abstained"' in body


def test_underlying_api_still_serves_data(client):
    # Feed data comes from the API; abstained row exposes no plate.
    data = client.get("/observations").json()
    ids = {o["event_id"] for o in data}
    assert {"e1", "abs1"} <= ids
    abs_row = client.get("/observations/abs1").json()
    assert abs_row["plate_normalized"] is None


def test_empty_repo_handled(client):
    # Search with no matches returns [] (dashboard shows "No observations").
    assert client.get("/plates/ZZ00ZZ0000/observations").json() == []

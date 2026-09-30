"""HTTP integration tests for SIH city analytics endpoints."""

import threading

import httpx
import pytest

from phase1_anpr.api import create_server
from phase1_anpr.observation.observation_builder import PlateObservation
from phase1_anpr.persistence import SQLiteObservationRepository
from phase2_city.graph import CityCameraGraph
from phase2_city.models import Camera, CameraLink


def _obs(event_id, camera_id, plate, timestamp):
    return PlateObservation(
        event_id=event_id,
        camera_id=camera_id,
        track_id=1,
        timestamp=timestamp,
        plate_raw=plate,
        plate_normalized=plate,
        confidence=0.95,
        status="accepted",
        detector_confidence=0.93,
        ocr_confidence=0.94,
        quality_score=0.90,
        best_frame_number=1,
        plate_image_path=None,
        model_version="test",
    )


@pytest.fixture
def client():
    repo = SQLiteObservationRepository(":memory:")
    repo.save(_obs("e1", "CAM_A", "GJ01AB1234", "2026-09-30T10:00:00+00:00"))
    repo.save(_obs("e2", "CAM_B", "GJ01AB1234", "2026-09-30T10:10:00+00:00"))
    cameras = [
        Camera("CAM_A", "A", 22.30, 73.18, "Road A", 90.0),
        Camera("CAM_B", "B", 22.31, 73.19, "Road B", 90.0),
    ]
    links = [
        CameraLink("CAM_A", "CAM_B", 1000.0, "Road AB", "eastbound"),
    ]
    graph = CityCameraGraph(cameras, links)
    server = create_server(repo, port=0, city_graph=graph)
    host, port = server.server_address
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with httpx.Client(base_url=f"http://{host}:{port}") as c:
        yield c
    server.shutdown()
    repo.close()


def test_traffic_analytics_endpoint(client):
    response = client.get("/v1/analytics/traffic")
    assert response.status_code == 200
    body = response.json()
    assert body["resource_type"] == "city_traffic_analytics"
    assert body["sample_observations"] == 2
    assert len(body["cameras"]) == 2


def test_origin_destination_endpoint(client):
    response = client.get("/v1/analytics/origin-destination")
    assert response.status_code == 200
    body = response.json()
    assert body["accepted_multi_camera_trajectories"] == 1
    assert body["origin_destination_pairs"][0]["origin_camera_id"] == "CAM_A"
    assert body["origin_destination_pairs"][0]["destination_camera_id"] == "CAM_B"


def test_route_anomaly_endpoint(client):
    response = client.get("/v1/alerts/route-anomalies")
    assert response.status_code == 200
    body = response.json()
    assert body["resource_type"] == "city_route_anomaly_alerts"
    assert body["evaluated_transitions"] == 1
    assert body["alert_count"] == 0

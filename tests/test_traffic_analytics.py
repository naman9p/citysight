"""Tests for SIH PS traffic analytics, OD aggregation and route anomaly MVP."""

from phase1_anpr.observation.observation_builder import PlateObservation
from phase1_anpr.persistence import SQLiteObservationRepository
from phase2_city.graph import CityCameraGraph
from phase2_city.models import Camera, CameraLink
from phase2_city.traffic_analytics import (
    origin_destination_summary,
    route_anomaly_alerts,
    traffic_summary,
)


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
        detector_confidence=0.92,
        ocr_confidence=0.94,
        quality_score=0.90,
        best_frame_number=1,
        plate_image_path=None,
        model_version="test",
    )


def _graph():
    cameras = [
        Camera("CAM_A", "A", 22.30, 73.18, "Road A", 90.0, "central"),
        Camera("CAM_B", "B", 22.31, 73.19, "Road B", 90.0, "central"),
        Camera("CAM_C", "C", 22.32, 73.20, "Road C", 90.0, "north"),
    ]
    links = [
        CameraLink("CAM_A", "CAM_B", 1000.0, "Road AB", "eastbound"),
        CameraLink("CAM_B", "CAM_C", 1000.0, "Road BC", "eastbound"),
    ]
    return CityCameraGraph(cameras, links)


def _seed(repo):
    repo.save(_obs("e1", "CAM_A", "GJ01AB1234", "2026-09-30T10:00:00+00:00"))
    repo.save(_obs("e2", "CAM_B", "GJ01AB1234", "2026-09-30T10:10:00+00:00"))
    repo.save(_obs("e3", "CAM_C", "GJ01AB1234", "2026-09-30T10:20:00+00:00"))
    repo.save(_obs("e4", "CAM_A", "GJ02CD5678", "2026-09-30T11:00:00+00:00"))
    repo.save(_obs("e5", "CAM_C", "GJ02CD5678", "2026-09-30T11:01:00+00:00"))


def test_traffic_summary_is_heatmap_ready():
    repo = SQLiteObservationRepository(":memory:")
    try:
        _seed(repo)
        result = traffic_summary(repo, _graph())
        assert result["sample_observations"] == 5
        assert result["camera_count"] == 3
        by_id = {item["camera_id"]: item for item in result["cameras"]}
        assert by_id["CAM_A"]["observation_count"] == 2
        assert by_id["CAM_A"]["latitude"] == 22.30
        assert 0.0 <= by_id["CAM_C"]["heatmap_weight"] <= 1.0
    finally:
        repo.close()


def test_origin_destination_aggregation():
    repo = SQLiteObservationRepository(":memory:")
    try:
        _seed(repo)
        result = origin_destination_summary(repo, _graph())
        assert result["accepted_multi_camera_trajectories"] == 2
        pairs = {
            (item["origin_camera_id"], item["destination_camera_id"]): item["count"]
            for item in result["origin_destination_pairs"]
        }
        assert pairs[("CAM_A", "CAM_C")] == 2
    finally:
        repo.close()


def test_route_anomaly_detects_topology_gap():
    repo = SQLiteObservationRepository(":memory:")
    try:
        _seed(repo)
        result = route_anomaly_alerts(repo, _graph(), max_speed_kph=160.0)
        assert result["evaluated_transitions"] == 3
        assert result["alert_count"] == 1
        alert = result["alerts"][0]
        assert alert["plate_normalized"] == "GJ02CD5678"
        assert alert["anomaly_type"] == "topology_gap"
        assert alert["from_camera_id"] == "CAM_A"
        assert alert["to_camera_id"] == "CAM_C"
    finally:
        repo.close()

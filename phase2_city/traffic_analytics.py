"""Lightweight city traffic analytics built from persisted ANPR observations.

The module intentionally reuses CitySight's existing observation repository and
camera topology. It provides three deterministic, read-only MVP capabilities:
camera-level traffic density summaries (with heatmap-ready GIS points),
origin-destination aggregation from accepted plate trajectories, and route
anomaly alerts based on topology gaps or implausible direct-link speeds.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime


DEFAULT_ANALYTICS_LIMIT = 500
DEFAULT_MAX_SPEED_KPH = 160.0


class TrafficAnalyticsError(ValueError):
    """Raised when analytics input data or policy is invalid."""


def _parse_timestamp(value) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise TrafficAnalyticsError("observation timestamp is missing")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise TrafficAnalyticsError(
            f"invalid observation timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        raise TrafficAnalyticsError(
            f"observation timestamp must include timezone: {value!r}")
    return parsed


def _rows(repository, limit: int) -> list[dict]:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise TrafficAnalyticsError("limit must be an integer >= 1")
    return list(repository.list_recent(limit))


def _accepted_plate_rows(repository, limit: int) -> list[dict]:
    return [
        row for row in _rows(repository, limit)
        if row.get("status") == "accepted" and row.get("plate_normalized")
    ]


def _plate_sequences(repository, limit: int):
    grouped = defaultdict(list)
    for row in _accepted_plate_rows(repository, limit):
        item = dict(row)
        item["_timestamp"] = _parse_timestamp(row.get("timestamp"))
        grouped[row["plate_normalized"]].append(item)

    for plate, items in grouped.items():
        items.sort(key=lambda row: (row["_timestamp"], row.get("event_id") or ""))
        collapsed = []
        for row in items:
            if collapsed and collapsed[-1].get("camera_id") == row.get("camera_id"):
                continue
            collapsed.append(row)
        if collapsed:
            yield plate, collapsed


def traffic_summary(repository, city_graph, *, limit=DEFAULT_ANALYTICS_LIMIT) -> dict:
    """Return camera traffic counts plus GIS points usable by a heatmap layer."""
    rows = _rows(repository, limit)
    total_by_camera = Counter()
    accepted_by_camera = Counter()
    status_counts = Counter()
    hourly_counts = Counter()

    for row in rows:
        camera_id = row.get("camera_id")
        if camera_id:
            total_by_camera[camera_id] += 1
            if row.get("status") == "accepted":
                accepted_by_camera[camera_id] += 1
        status_counts[row.get("status") or "unknown"] += 1
        try:
            ts = _parse_timestamp(row.get("timestamp"))
        except TrafficAnalyticsError:
            continue
        hourly_counts[
            ts.replace(minute=0, second=0, microsecond=0).isoformat()
        ] += 1

    max_count = max(total_by_camera.values(), default=0)
    cameras = []
    for camera in sorted(city_graph.list_cameras(), key=lambda item: item.camera_id):
        count = total_by_camera.get(camera.camera_id, 0)
        cameras.append({
            "camera_id": camera.camera_id,
            "camera_name": camera.name,
            "road_name": camera.road_name,
            "zone": camera.zone,
            "latitude": camera.latitude,
            "longitude": camera.longitude,
            "heading_deg": camera.heading_deg,
            "observation_count": count,
            "accepted_count": accepted_by_camera.get(camera.camera_id, 0),
            "heatmap_weight": (count / max_count) if max_count else 0.0,
        })

    return {
        "resource_type": "city_traffic_analytics",
        "sample_limit": limit,
        "sample_observations": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "camera_count": len(cameras),
        "cameras": cameras,
        "hourly_flow": [
            {"hour": hour, "observation_count": count}
            for hour, count in sorted(hourly_counts.items())
        ],
        "heatmap_semantics": (
            "heatmap_weight is normalized observed-track volume within the "
            "queried sample; it is not a road-capacity congestion estimate"
        ),
    }


def origin_destination_summary(
        repository, city_graph, *, limit=DEFAULT_ANALYTICS_LIMIT) -> dict:
    """Aggregate accepted plate trajectories into OD pairs and transitions."""
    od_counts = Counter()
    transition_counts = Counter()
    trajectory_count = 0

    for _plate, sequence in _plate_sequences(repository, limit):
        if len(sequence) < 2:
            continue
        trajectory_count += 1
        origin = sequence[0].get("camera_id")
        destination = sequence[-1].get("camera_id")
        if origin and destination and origin != destination:
            od_counts[(origin, destination)] += 1

        for first, second in zip(sequence, sequence[1:]):
            a, b = first.get("camera_id"), second.get("camera_id")
            if a and b and a != b:
                transition_counts[(a, b)] += 1

    def pair_payload(pair, count):
        origin, destination = pair
        origin_camera = city_graph.get_camera(origin)
        destination_camera = city_graph.get_camera(destination)
        return {
            "origin_camera_id": origin,
            "destination_camera_id": destination,
            "origin_name": None if origin_camera is None else origin_camera.name,
            "destination_name": (
                None if destination_camera is None else destination_camera.name),
            "count": count,
        }

    return {
        "resource_type": "city_origin_destination_analytics",
        "sample_limit": limit,
        "accepted_multi_camera_trajectories": trajectory_count,
        "origin_destination_pairs": [
            pair_payload(pair, count)
            for pair, count in sorted(
                od_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
        "camera_transitions": [
            {
                "from_camera_id": pair[0],
                "to_camera_id": pair[1],
                "count": count,
            }
            for pair, count in sorted(
                transition_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
    }


def route_anomaly_alerts(
        repository,
        city_graph,
        *,
        limit=DEFAULT_ANALYTICS_LIMIT,
        max_speed_kph=DEFAULT_MAX_SPEED_KPH,
) -> dict:
    """Generate explainable anomaly alerts from accepted cross-camera sightings."""
    try:
        speed_limit = float(max_speed_kph)
    except (TypeError, ValueError) as exc:
        raise TrafficAnalyticsError("max_speed_kph must be numeric") from exc
    if speed_limit <= 0:
        raise TrafficAnalyticsError("max_speed_kph must be > 0")

    alerts = []
    evaluated_transitions = 0

    for plate, sequence in _plate_sequences(repository, limit):
        for first, second in zip(sequence, sequence[1:]):
            from_camera = first.get("camera_id")
            to_camera = second.get("camera_id")
            if not from_camera or not to_camera or from_camera == to_camera:
                continue

            evaluated_transitions += 1
            elapsed = (
                second["_timestamp"] - first["_timestamp"]
            ).total_seconds()
            link = city_graph.get_link(from_camera, to_camera)

            base = {
                "plate_normalized": plate,
                "from_event_id": first.get("event_id"),
                "to_event_id": second.get("event_id"),
                "from_camera_id": from_camera,
                "to_camera_id": to_camera,
                "departed_at": first["_timestamp"].isoformat(),
                "arrived_at": second["_timestamp"].isoformat(),
                "elapsed_seconds": elapsed,
            }

            if elapsed <= 0:
                alerts.append({
                    **base,
                    "anomaly_type": "non_positive_travel_time",
                    "severity": "high",
                    "reason": (
                        "arrival timestamp is not after departure timestamp"),
                })
                continue

            if link is None:
                alerts.append({
                    **base,
                    "anomaly_type": "topology_gap",
                    "severity": "medium",
                    "reason": (
                        "no direct camera-topology link exists for transition"),
                })
                continue

            speed_kph = (float(link.distance_m) / elapsed) * 3.6
            if speed_kph > speed_limit:
                alerts.append({
                    **base,
                    "anomaly_type": "implausible_speed",
                    "severity": "high",
                    "distance_m": float(link.distance_m),
                    "estimated_speed_kph": speed_kph,
                    "threshold_kph": speed_limit,
                    "reason": (
                        "direct-link travel speed exceeds configured threshold"),
                })

    return {
        "resource_type": "city_route_anomaly_alerts",
        "sample_limit": limit,
        "maximum_speed_kph": speed_limit,
        "evaluated_transitions": evaluated_transitions,
        "alert_count": len(alerts),
        "alerts": alerts,
        "alert_semantics": (
            "rule-based operational alerts from accepted observations; "
            "they are not confirmed violations or identity assertions"
        ),
    }


__all__ = [
    "DEFAULT_ANALYTICS_LIMIT",
    "DEFAULT_MAX_SPEED_KPH",
    "TrafficAnalyticsError",
    "traffic_summary",
    "origin_destination_summary",
    "route_anomaly_alerts",
]

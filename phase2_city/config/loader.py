"""Loader for the static city topology config (Step 17).

Reads ``city.yaml``, validates its shape, and builds ``Camera`` / ``CameraLink``
objects. Validation performed here (beyond the models' own field checks):

* duplicate camera IDs are rejected
* duplicate directed links are rejected
* links referencing an undefined camera are rejected

Loading and persisting are separate concerns: ``load_city_config`` returns
domain objects; ``load_into_repository`` is an explicit helper that saves them
(cameras first, then links) into a ``CityRepository``.
"""

from pathlib import Path

import yaml

from phase2_city.models import Camera, CameraLink, CameraValidationError

DEFAULT_CITY_CONFIG_PATH = Path("phase2_city/config/city.yaml")


class CityConfigError(ValueError):
    """Raised when the city config is malformed or internally inconsistent."""


def load_city_config(config_path=DEFAULT_CITY_CONFIG_PATH):
    """Load and validate a city config file.

    Returns ``(cameras, links)`` as lists of domain objects. Raises
    ``FileNotFoundError`` if missing and ``CityConfigError`` on any structural
    or referential problem.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"City config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise CityConfigError("city config must be a mapping with 'cameras'/'links'")

    raw_cameras = data.get("cameras") or []
    raw_links = data.get("links") or []
    if not isinstance(raw_cameras, list) or not isinstance(raw_links, list):
        raise CityConfigError("'cameras' and 'links' must be lists")

    cameras = []
    seen_camera_ids = set()
    for entry in raw_cameras:
        if not isinstance(entry, dict):
            raise CityConfigError(f"camera entry must be a mapping: {entry!r}")
        try:
            camera = Camera(
                camera_id=entry.get("camera_id"),
                name=entry.get("name"),
                latitude=entry.get("latitude"),
                longitude=entry.get("longitude"),
                road_name=entry.get("road_name"),
                heading_deg=entry.get("heading_deg"),
                zone=entry.get("zone"),
                enabled=entry.get("enabled", True),
            )
        except (CameraValidationError, TypeError) as exc:
            raise CityConfigError(f"invalid camera entry {entry!r}: {exc}") from exc
        if camera.camera_id in seen_camera_ids:
            raise CityConfigError(f"duplicate camera_id: {camera.camera_id}")
        seen_camera_ids.add(camera.camera_id)
        cameras.append(camera)

    links = []
    seen_link_keys = set()
    for entry in raw_links:
        if not isinstance(entry, dict):
            raise CityConfigError(f"link entry must be a mapping: {entry!r}")
        try:
            link = CameraLink(
                from_camera_id=entry.get("from_camera_id"),
                to_camera_id=entry.get("to_camera_id"),
                distance_m=entry.get("distance_m"),
                road_name=entry.get("road_name"),
                travel_direction=entry.get("travel_direction"),
            )
        except (CameraValidationError, TypeError) as exc:
            raise CityConfigError(f"invalid link entry {entry!r}: {exc}") from exc
        if link.key in seen_link_keys:
            raise CityConfigError(f"duplicate directed link: {link.key}")
        for cid in (link.from_camera_id, link.to_camera_id):
            if cid not in seen_camera_ids:
                raise CityConfigError(
                    f"link {link.key} references undefined camera '{cid}'"
                )
        seen_link_keys.add(link.key)
        links.append(link)

    return cameras, links


def load_into_repository(repo, config_path=DEFAULT_CITY_CONFIG_PATH):
    """Load a city config and persist it into ``repo`` (cameras then links).

    Returns the ``(cameras, links)`` that were saved.
    """
    cameras, links = load_city_config(config_path)
    for camera in cameras:
        repo.save_camera(camera)
    for link in links:
        repo.save_link(link)
    return cameras, links

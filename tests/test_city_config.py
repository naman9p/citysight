"""Step 17 tests: city.yaml loading and validation."""

import textwrap

import pytest

from phase2_city import SQLiteCityRepository
from phase2_city.config import (
    CityConfigError,
    DEFAULT_CITY_CONFIG_PATH,
    load_city_config,
    load_into_repository,
)


def write_cfg(tmp_path, text):
    p = tmp_path / "city.yaml"
    p.write_text(textwrap.dedent(text), encoding="utf-8")
    return p


VALID = """
    cameras:
      - camera_id: CAM_01
        name: Junction 1
        latitude: 28.61
        longitude: 77.22
        road_name: Demo Road
        heading_deg: 90.0
        zone: demo
        enabled: true
      - camera_id: CAM_02
        name: Junction 2
        latitude: 28.62
        longitude: 77.24
        road_name: Demo Road
        heading_deg: 90.0
    links:
      - from_camera_id: CAM_01
        to_camera_id: CAM_02
        distance_m: 1800
        road_name: Demo Road
        travel_direction: eastbound
"""


def test_shipped_city_yaml_loads():
    cameras, links = load_city_config(DEFAULT_CITY_CONFIG_PATH)
    assert len(cameras) == 4
    assert len(links) == 4


def test_valid_config_loads(tmp_path):
    cameras, links = load_city_config(write_cfg(tmp_path, VALID))
    assert [c.camera_id for c in cameras] == ["CAM_01", "CAM_02"]
    assert links[0].key == ("CAM_01", "CAM_02")


DUP_CAMERA = """
    cameras:
      - camera_id: CAM_01
        name: Junction 1
        latitude: 28.61
        longitude: 77.22
        road_name: Demo Road
        heading_deg: 90.0
      - camera_id: CAM_01
        name: Dup
        latitude: 28.63
        longitude: 77.25
        road_name: Demo Road
        heading_deg: 10.0
    links: []
"""

DUP_LINK = """
    cameras:
      - camera_id: CAM_01
        name: Junction 1
        latitude: 28.61
        longitude: 77.22
        road_name: Demo Road
        heading_deg: 90.0
      - camera_id: CAM_02
        name: Junction 2
        latitude: 28.62
        longitude: 77.24
        road_name: Demo Road
        heading_deg: 90.0
    links:
      - from_camera_id: CAM_01
        to_camera_id: CAM_02
        distance_m: 1800
        road_name: Demo Road
      - from_camera_id: CAM_01
        to_camera_id: CAM_02
        distance_m: 900
        road_name: Demo Road
"""

UNDEFINED_LINK = """
    cameras:
      - camera_id: CAM_01
        name: Junction 1
        latitude: 28.61
        longitude: 77.22
        road_name: Demo Road
        heading_deg: 90.0
      - camera_id: CAM_02
        name: Junction 2
        latitude: 28.62
        longitude: 77.24
        road_name: Demo Road
        heading_deg: 90.0
    links:
      - from_camera_id: CAM_02
        to_camera_id: CAM_99
        distance_m: 900
        road_name: Demo Road
"""


def test_duplicate_camera_ids_rejected(tmp_path):
    with pytest.raises(CityConfigError, match="duplicate camera_id"):
        load_city_config(write_cfg(tmp_path, DUP_CAMERA))


def test_duplicate_directed_links_rejected(tmp_path):
    with pytest.raises(CityConfigError, match="duplicate directed link"):
        load_city_config(write_cfg(tmp_path, DUP_LINK))


def test_link_to_undefined_camera_rejected(tmp_path):
    with pytest.raises(CityConfigError, match="undefined camera"):
        load_city_config(write_cfg(tmp_path, UNDEFINED_LINK))


def test_malformed_coordinates_rejected(tmp_path):
    cfg = """
    cameras:
      - camera_id: CAM_01
        name: Bad
        latitude: 999
        longitude: 77.22
        road_name: Demo Road
        heading_deg: 90.0
    links: []
"""
    with pytest.raises(CityConfigError):
        load_city_config(write_cfg(tmp_path, cfg))


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_city_config(tmp_path / "nope.yaml")


def test_load_into_repository(tmp_path):
    repo = SQLiteCityRepository(":memory:")
    cameras, links = load_into_repository(repo, write_cfg(tmp_path, VALID))
    assert len(repo.list_cameras()) == 2
    assert len(repo.list_links()) == 1
    repo.close()

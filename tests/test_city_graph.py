"""Step 17 tests: CityCameraGraph direct topology."""

from phase2_city import Camera, CameraLink, CityCameraGraph, SQLiteCityRepository


def cam(cid):
    return Camera(camera_id=cid, name=f"Cam {cid}", latitude=28.6, longitude=77.2,
                  road_name="Demo Road", heading_deg=90.0)


def link(a, b):
    return CameraLink(from_camera_id=a, to_camera_id=b, distance_m=1000.0,
                      road_name="Demo Road")


def build():
    cameras = [cam("CAM_01"), cam("CAM_02"), cam("CAM_03")]
    links = [link("CAM_01", "CAM_02"), link("CAM_02", "CAM_03")]
    return CityCameraGraph(cameras, links)


def test_get_camera():
    g = build()
    assert g.get_camera("CAM_02").camera_id == "CAM_02"
    assert g.get_camera("NOPE") is None


def test_outgoing_links():
    g = build()
    out = g.get_outgoing_links("CAM_02")
    assert [l.to_camera_id for l in out] == ["CAM_03"]
    assert g.get_outgoing_links("CAM_03") == []


def test_incoming_links():
    g = build()
    inc = g.get_incoming_links("CAM_02")
    assert [l.from_camera_id for l in inc] == ["CAM_01"]
    assert g.get_incoming_links("CAM_01") == []


def test_direct_connectivity_direction_matters():
    g = build()
    assert g.are_directly_connected("CAM_01", "CAM_02") is True
    assert g.are_directly_connected("CAM_02", "CAM_01") is False


def test_unknown_camera_is_clear():
    g = build()
    assert g.get_outgoing_links("UNKNOWN") == []
    assert g.get_incoming_links("UNKNOWN") == []
    assert g.are_directly_connected("UNKNOWN", "CAM_01") is False


def test_counts_and_listing():
    g = build()
    assert g.camera_count == 3
    assert g.link_count == 2
    assert len(g.list_cameras()) == 3
    assert len(g.list_links()) == 2


def test_from_repository():
    repo = SQLiteCityRepository(":memory:")
    repo.save_camera(cam("CAM_01"))
    repo.save_camera(cam("CAM_02"))
    repo.save_link(link("CAM_01", "CAM_02"))
    g = CityCameraGraph.from_repository(repo)
    assert g.camera_count == 2
    assert g.are_directly_connected("CAM_01", "CAM_02") is True
    repo.close()
